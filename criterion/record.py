"""
Module 4: Record (the Preserver)

Furqan reading order: 4 of 4 (kernel)
Dependency depth: 3 — depends on capture, divergence, and verdict

Preserves everything append-only: the source, the output,
the divergences, and the verdict. Nothing is overwritten.
Nothing is silently corrected. The record is the evidence.

The Record module stores to a local JSONL file by default.
Each line is one complete assessment — self-contained,
timestamped, and hash-chained to the previous entry.

What the chain does and does not prove
--------------------------------------
* Unkeyed (default): any edit to an entry, and any reordering or
  removal of an entry from the middle, is detected. It does NOT stop
  someone with write access from recomputing every hash, and it cannot
  see that entries were deleted from the END of the file.
* Keyed (set CRITERION_RECORD_KEY, or pass key=): each entry carries an
  HMAC-SHA256 over its id. Rewriting the chain now needs the key.
* Anchored: `verify_chain(expected_head=..., expected_count=...)` checks
  the ledger against a head hash / length you stored somewhere the
  writer cannot reach. This is what detects tail truncation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from criterion._version import __version__
from criterion.capture import CapturedPair
from criterion.divergence import DivergenceReport
from criterion.verdict import Verdict

try:  # POSIX advisory locking; absent on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


class RecordCorruptError(ValueError):
    """The record file's last line is unreadable; appending would fork the chain."""


# Entry-id algorithm versions.
#   1 — legacy (<= 0.3.1): fields concatenated with no delimiter. Ambiguous:
#       a digit can be moved across a field boundary without changing the
#       hash. Still verifiable so old ledgers keep working; never written.
#   2 — canonical JSON over every integrity-protected field, domain-separated
#       (0.4.0 to 0.6.0). The field list is frozen below: fields added later
#       are not part of a scheme-2 id, so those ledgers keep verifying.
#   3 — as 2, over the current field list: adds the entry kind, per-sentence
#       support and reviewer decisions (0.7.0).
HASH_SCHEME = 3
_DOMAIN = b"criterion-record-v2\n"
_DOMAIN_V3 = b"criterion-record-v3\n"
_KEY_ENV = "CRITERION_RECORD_KEY"


# ---------------------------------------------------------------------------
# Record entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecordEntry:
    """One complete Criterion assessment, ready for storage.

    Self-contained: everything needed to understand, verify,
    and reproduce the assessment is in this single record.
    """

    # Identity
    entry_id: str              # SHA-256 of the content fields
    recorded_at: str           # ISO-8601 UTC timestamp
    predecessor_hash: str      # Hash of the previous entry (append-only chain)

    # Content
    source_hash: str           # SHA-256 of the source text
    output_hash: str           # SHA-256 of the output text
    source_text: str | None    # The full source text (None if store_text=False)
    output_text: str | None    # The full output text (None if store_text=False)

    # Assessment
    trust_score: float
    trust_level: str
    total_divergences: int
    divergences: list[dict[str, Any]]
    bond_divergences: list[dict[str, Any]]
    broken_bonds: int
    summary: str

    # Provenance
    criterion_version: str
    mode: str = ""                                   # verification mode used
    detector_config: dict[str, Any] | None = None    # which detectors ran, which models
    hash_scheme: int = 1                             # absent in legacy entries
    mac: str | None = None                           # HMAC-SHA256(key, entry_id), if keyed

    # 0.7.0 (hash scheme 3)
    kind: str = "verdict"                            # "verdict" or "review"
    # One row per judged output sentence: index, text_hash, evidence_hash,
    # support, models, confidence. Protected by the entry id.
    sentences: list[dict[str, Any]] | None = None
    # The sentences and the source passage each was best supported by, in the
    # same order. Like source_text these are bound by hash, not by the id, so
    # they can be withheld or redacted without breaking the chain.
    sentence_texts: list[dict[str, Any]] | None = None
    # For kind == "review": {"of": verdict entry id, "sentence": index or
    # None, "decision": ..., "reviewer": ..., "note": ...}.
    review: dict[str, Any] | None = None


_ENTRY_FIELDS = tuple(f.name for f in fields(RecordEntry))
# Not covered by entry_id: the id itself, the MAC over it, and the raw texts
# (bound through source_hash/output_hash so they can be withheld or redacted
# without breaking the chain).
_UNHASHED = {"entry_id", "mac", "source_text", "output_text", "sentence_texts"}
# Frozen: exactly the fields a scheme-2 id was computed over.
_V2_FIELDS = (
    "recorded_at", "predecessor_hash", "source_hash", "output_hash",
    "trust_score", "trust_level", "total_divergences", "divergences",
    "bond_divergences", "broken_bonds", "summary", "criterion_version",
    "mode", "detector_config", "hash_scheme",
)
_V3_ONLY = ("kind", "sentences", "sentence_texts", "review")


def _serialize_divergence(d: Any) -> dict[str, Any]:
    """Serialize a Divergence to a plain dict."""
    result = {
        "type": d.dtype.value,
        "severity": d.severity.value,
        "explanation": d.explanation,
        "confidence": d.confidence,
        "suggested_correction": d.suggested_correction,
        "verification_method": d.verification_method,
    }
    if getattr(d, "code", ""):
        result["code"] = d.code
    if d.source_atom is not None:
        result["source_text"] = d.source_atom.text
        result["source_index"] = d.source_atom.index
        result["source_type"] = d.source_atom.atom_type.value
    if d.output_atom is not None:
        result["output_text"] = d.output_atom.text
        result["output_index"] = d.output_atom.index
        result["output_type"] = d.output_atom.atom_type.value
    return result


def _serialize_bond_divergence(bd: Any) -> dict[str, Any]:
    """Serialize a BondDivergence to a plain dict."""
    return {
        "bond_type": bd.bond.bond_type.value,
        "source_atom_text": bd.source_atom.text[:100],
        "target_atom_text": bd.target_atom.text[:100],
        "explanation": bd.explanation,
        "severity": bd.severity.value,
    }


def _hash_content(*parts: str) -> str:
    """Legacy (scheme 1) entry id: undelimited concatenation. Verify-only."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def _entry_id_v2(data: dict[str, Any]) -> str:
    """Scheme 2 entry id: SHA-256 over canonical JSON of the frozen field list."""
    protected = {k: data.get(k) for k in _V2_FIELDS}
    blob = json.dumps(protected, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(_DOMAIN + blob.encode("utf-8")).hexdigest()


def _entry_id_v3(data: dict[str, Any]) -> str:
    """Scheme 3 entry id: SHA-256 over canonical JSON of every protected field."""
    protected = {k: data.get(k) for k in _ENTRY_FIELDS if k not in _UNHASHED}
    blob = json.dumps(protected, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(_DOMAIN_V3 + blob.encode("utf-8")).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_key(key: bytes | str | None) -> bytes | None:
    if key is None:
        key = os.environ.get(_KEY_ENV) or None
    if key is None:
        return None
    return key.encode("utf-8") if isinstance(key, str) else key


def _mac(key: bytes, entry_id: str) -> str:
    return hmac.new(key, entry_id.encode("ascii"), hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_DEFAULT_RECORD_PATH = Path(".criterion/record.jsonl")


def _last_line(f) -> str:
    """Return the last non-blank line of an open binary file, reading from the end."""
    f.seek(0, os.SEEK_END)
    end = f.tell()
    buf = b""
    pos = end
    while pos > 0:
        step = min(65536, pos)
        pos -= step
        f.seek(pos)
        buf = f.read(step) + buf
        stripped = buf.rstrip()
        if b"\n" in stripped:
            return stripped.rsplit(b"\n", 1)[1].decode("utf-8")
    return buf.strip().decode("utf-8")


def _last_hash_from(f) -> str:
    last = _last_line(f)
    if not last:
        return "genesis"
    try:
        entry_id = json.loads(last)["entry_id"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise RecordCorruptError(
            "last line of the record is not a valid entry; refusing to append "
            "(appending would silently restart the chain at genesis)"
        ) from e
    if not isinstance(entry_id, str):
        raise RecordCorruptError("last record entry has a non-string entry_id")
    return entry_id


def _get_last_hash(path: Path) -> str:
    """Read the hash of the last entry in the record file.

    Returns "genesis" if the file is empty or does not exist.
    Raises RecordCorruptError if the last line cannot be parsed.
    """
    if not path.exists():
        return "genesis"
    with open(path, "rb") as f:
        return _last_hash_from(f)


def _entry_to_dict(entry: RecordEntry) -> dict[str, Any]:
    data = {name: getattr(entry, name) for name in _ENTRY_FIELDS}
    if data["mac"] is None:
        del data["mac"]
    if entry.hash_scheme < 3:
        # Older entries are written back exactly as they were read.
        for name in _V3_ONLY:
            del data[name]
    return data


def _append(path: Path, data: dict[str, Any], mac_key: bytes | None) -> RecordEntry:
    """Chain `data` onto the ledger under an exclusive lock and write it."""
    with open(path, "a+b") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            data["predecessor_hash"] = _last_hash_from(f)
            data["entry_id"] = _entry_id_v3(data)
            data["mac"] = _mac(mac_key, data["entry_id"]) if mac_key else None
            entry = RecordEntry(**data)
            line = json.dumps(_entry_to_dict(entry), ensure_ascii=False) + "\n"
            f.seek(0, os.SEEK_END)
            f.write(line.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preserve(
    pair: CapturedPair,
    report: DivergenceReport | None,
    verdict: Verdict,
    record_path: Path | str | None = None,
    *,
    mode: str = "",
    detector_config: dict[str, Any] | None = None,
    store_text: bool = True,
    key: bytes | str | None = None,
    sentences: Sequence[dict[str, Any]] | None = None,
) -> RecordEntry:
    """Preserve a complete Criterion assessment to the append-only record.

    This is the fourth and final step in the kernel pipeline.
    After verdict, record. Nothing is lost.

    The read-last-hash + append sequence runs under an exclusive file
    lock, so concurrent writers cannot fork the chain.

    Args:
        pair: The captured source-output pair.
        report: The divergence report (may be None for degenerate inputs).
        verdict: The trust verdict.
        record_path: Where to store the record. Defaults to .criterion/record.jsonl
        mode: The verification mode the verdict was produced under.
        detector_config: Which detectors ran and with which models.
        store_text: If False, store only the SHA-256 of source and output,
            not the texts themselves (for sensitive material).
        key: HMAC key for a keyed chain. Defaults to $CRITERION_RECORD_KEY.
        sentences: Per-sentence support rows from the verdict (text, support,
            models, confidence, evidence). Recorded so a reviewer can later be
            shown exactly what the machine saw and decided.

    Returns:
        The RecordEntry that was written.
    """
    path = Path(record_path) if record_path else _DEFAULT_RECORD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    mac_key = _resolve_key(key)

    rows: list[dict[str, Any]] | None = None
    texts: list[dict[str, Any]] | None = None
    if sentences:
        rows, texts = [], []
        for i, s in enumerate(sentences):
            evidence = s.get("evidence") or ""
            rows.append({
                "index": i,
                "text_hash": _sha(s["text"]),
                "evidence_hash": _sha(evidence) if evidence else None,
                "support": s.get("support"),
                "models": s.get("models"),
                "confidence": s.get("confidence"),
            })
            texts.append({"text": s["text"], "evidence": evidence or None})
        if not store_text:
            texts = None

    if detector_config is None and report is not None:
        detector_config = dict(getattr(report, "capabilities", {}) or {}) or None

    data: dict[str, Any] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_hash": pair.source.sha256,
        "output_hash": pair.output.sha256,
        "source_text": pair.source.raw_text if store_text else None,
        "output_text": pair.output.raw_text if store_text else None,
        "trust_score": verdict.trust_score,
        "trust_level": verdict.trust_level.value,
        "total_divergences": verdict.total_divergences,
        "divergences": [_serialize_divergence(d) for d in verdict.divergences],
        "bond_divergences": [_serialize_bond_divergence(bd) for bd in verdict.bond_divergences],
        "broken_bonds": len(verdict.bond_divergences),
        "summary": verdict.summary,
        "criterion_version": __version__,
        "mode": mode,
        "detector_config": detector_config,
        "hash_scheme": HASH_SCHEME,
        "kind": "verdict",
        "sentences": rows,
        "sentence_texts": texts,
        "review": None,
    }
    return _append(path, data, mac_key)


DECISIONS = ("supported", "unsupported", "unclear")


def preserve_review(
    target: RecordEntry,
    *,
    sentence: int | None,
    decision: str,
    reviewer: str,
    note: str = "",
    record_path: Path | str | None = None,
    key: bytes | str | None = None,
) -> RecordEntry:
    """Append a reviewer's decision about a recorded verdict.

    The decision is a new entry chained after everything before it; the
    verdict it concerns is never altered. It names that verdict by entry id
    and repeats its source and output hashes, so it cannot be re-pointed at
    different text.

    What this proves: that the decision was recorded in this position and has
    not been changed since. It does not prove who made it — `reviewer` is
    whatever name was given. With a record key, it proves the entry was
    written by someone holding the key.
    """
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("a reviewer name is required")
    if target.kind != "verdict":
        raise ValueError("only a verdict can be reviewed, not another review")
    if sentence is not None:
        count = len(target.sentences or [])
        if isinstance(sentence, bool) or not isinstance(sentence, int) or not 0 <= sentence < count:
            raise ValueError(f"sentence {sentence!r} is not one of the {count} recorded for this verdict")
    path = Path(record_path) if record_path else _DEFAULT_RECORD_PATH
    where = "the output" if sentence is None else f"sentence {sentence}"
    data: dict[str, Any] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_hash": target.source_hash,
        "output_hash": target.output_hash,
        "source_text": None,
        "output_text": None,
        "trust_score": 0.0,
        "trust_level": "review",
        "total_divergences": 0,
        "divergences": [],
        "bond_divergences": [],
        "broken_bonds": 0,
        "summary": f"{reviewer.strip()} judged {where} of {target.entry_id[:12]} {decision}.",
        "criterion_version": __version__,
        "mode": target.mode,
        "detector_config": None,
        "hash_scheme": HASH_SCHEME,
        "kind": "review",
        "sentences": None,
        "sentence_texts": None,
        "review": {
            "of": target.entry_id,
            "sentence": sentence,
            "decision": decision,
            "reviewer": reviewer.strip(),
            "note": note or "",
        },
    }
    return _append(path, data, _resolve_key(key))


def read_record(record_path: Path | str | None = None) -> list[RecordEntry]:
    """Read all entries from the record file.

    Returns entries in chronological order (oldest first).
    """
    path = Path(record_path) if record_path else _DEFAULT_RECORD_PATH
    if not path.exists():
        return []

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError("record line is not a JSON object")
            entries.append(RecordEntry(**data))
    return entries


def _recompute_entry_id(entry: RecordEntry) -> str:
    """Recompute an entry's ID from its content fields.

    If this doesn't match the stored entry_id, the content was tampered with.
    """
    if entry.hash_scheme == 3:
        return _entry_id_v3({name: getattr(entry, name) for name in _ENTRY_FIELDS})
    if entry.hash_scheme == 2:
        if any(getattr(entry, name) != RecordEntry.__dataclass_fields__[name].default for name in _V3_ONLY):
            # A scheme-2 id does not cover these fields; accepting them would
            # let unprotected content ride on a valid hash.
            raise ValueError("scheme-2 entry carries fields its hash does not protect")
        return _entry_id_v2(_entry_to_dict(entry))
    if entry.hash_scheme != 1:
        raise ValueError(f"unknown hash_scheme {entry.hash_scheme!r}")
    divs_json = json.dumps(entry.divergences, sort_keys=True, ensure_ascii=False)
    bond_divs_json = json.dumps(entry.bond_divergences, sort_keys=True, ensure_ascii=False)
    return _hash_content(
        entry.source_hash,
        entry.output_hash,
        entry.recorded_at,
        entry.predecessor_hash,
        str(entry.trust_score),
        entry.trust_level,
        str(entry.total_divergences),
        divs_json,
        bond_divs_json,
        str(entry.broken_bonds),
        entry.summary,
        entry.criterion_version,
    )


def _verify_text_hashes(entry: RecordEntry) -> str | None:
    """Verify that stored text matches stored hashes. Returns error or None."""
    for label, text, stored in (
        ("source", entry.source_text, entry.source_hash),
        ("output", entry.output_text, entry.output_hash),
    ):
        if text is None:
            continue  # text withheld (store_text=False); the hash still binds it
        if not isinstance(text, str):
            return f"{label} text is not a string"
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != stored:
            return f"{label} text hash mismatch: stored {str(stored)[:12]}, computed {actual[:12]}"
    if entry.sentence_texts is not None:
        rows = entry.sentences or []
        if not isinstance(entry.sentence_texts, list) or len(entry.sentence_texts) != len(rows):
            return "sentence texts do not line up with the recorded sentences"
        for row, t in zip(rows, entry.sentence_texts):
            if not isinstance(t, dict) or not isinstance(t.get("text"), str):
                return "sentence text is not a string"
            if _sha(t["text"]) != row.get("text_hash"):
                return f"sentence {row.get('index')} text hash mismatch"
            evidence = t.get("evidence")
            if evidence is not None and (not isinstance(evidence, str) or _sha(evidence) != row.get("evidence_hash")):
                return f"sentence {row.get('index')} evidence hash mismatch"
    return None


def _verify_kind(entry: RecordEntry, earlier: dict[str, RecordEntry]) -> str | None:
    """A review must point at an earlier verdict of the same text."""
    if entry.kind == "verdict":
        return "a verdict entry carries a review" if entry.review is not None else None
    if entry.kind != "review":
        return f"unknown entry kind {entry.kind!r}"
    r = entry.review
    if not isinstance(r, dict) or r.get("decision") not in DECISIONS or not r.get("reviewer"):
        return "malformed review"
    target = earlier.get(r.get("of"))
    if target is None or target.kind != "verdict":
        return "review does not refer to an earlier verdict in this record"
    if (target.source_hash, target.output_hash) != (entry.source_hash, entry.output_hash):
        return "review is bound to different text than the verdict it names"
    n = r.get("sentence")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or not 0 <= n < len(target.sentences or [])):
        return "review names a sentence the verdict does not have"
    return None


def chain_head(record_path: Path | str | None = None) -> tuple[str, int]:
    """Return (head entry id, entry count) — the values to anchor externally."""
    entries = read_record(record_path)
    return (entries[-1].entry_id if entries else "genesis", len(entries))


def verify_chain(
    record_path: Path | str | None = None,
    *,
    key: bytes | str | None = None,
    expected_head: str | None = None,
    expected_count: int | None = None,
) -> tuple[bool, str]:
    """Verify the append-only hash chain is intact.

    Checks:
    1. Predecessor links form a valid chain from genesis
    2. Entry IDs match recomputed content hashes (tamper detection)
    3. Source/output text hashes match the stored text
    4. With a key (argument or $CRITERION_RECORD_KEY): every entry's MAC
    5. With expected_head / expected_count: the ledger ends where your
       external anchor says it should (detects truncation and rewrite)

    Returns (True, "chain intact, ...") if the chain is valid, or
    (False, "description of break") if tampered.
    """
    path = Path(record_path) if record_path else _DEFAULT_RECORD_PATH
    if not path.exists():
        return False, f"record file not found: {path}"
    try:
        entries = read_record(record_path)
    except (json.JSONDecodeError, TypeError, KeyError, UnicodeDecodeError) as e:
        return False, f"malformed record: {e}"
    mac_key = _resolve_key(key)

    if entries and entries[0].predecessor_hash != "genesis":
        return False, f"first entry predecessor is {entries[0].predecessor_hash!r}, expected 'genesis'"

    seen: dict[str, RecordEntry] = {}
    for i, entry in enumerate(entries):
        # Check predecessor chain
        if i > 0 and entry.predecessor_hash != entries[i - 1].entry_id:
            return False, (
                f"entry {i} predecessor {entry.predecessor_hash!r} "
                f"does not match entry {i-1} id {entries[i-1].entry_id!r}"
            )

        # Recompute entry ID from content — detects tampered fields
        try:
            recomputed = _recompute_entry_id(entry)
        except (ValueError, TypeError) as e:
            return False, f"entry {i}: {e}"
        if not isinstance(entry.entry_id, str) or not hmac.compare_digest(recomputed, entry.entry_id):
            return False, (
                f"entry {i} content hash mismatch: stored {str(entry.entry_id)[:12]}..., "
                f"recomputed {recomputed[:12]}... — content was modified"
            )

        # Verify source/output text matches their stored hashes
        text_err = _verify_text_hashes(entry)
        if text_err:
            return False, f"entry {i}: {text_err}"

        kind_err = _verify_kind(entry, seen)
        if kind_err:
            return False, f"entry {i}: {kind_err}"
        seen[entry.entry_id] = entry

        if mac_key is not None:
            if not entry.mac:
                return False, f"entry {i} has no MAC but a record key was supplied"
            if not hmac.compare_digest(_mac(mac_key, entry.entry_id), str(entry.mac)):
                return False, f"entry {i} MAC mismatch — entry was not written with this key"

    head = entries[-1].entry_id if entries else "genesis"
    if expected_count is not None and len(entries) != expected_count:
        return False, f"expected {expected_count} entries, found {len(entries)} — record truncated or extended"
    if expected_head is not None and head != expected_head:
        return False, f"head is {head[:12]}..., expected {expected_head[:12]}... — record truncated or rewritten"

    if not entries:
        return True, "empty record"
    return True, f"chain intact, {len(entries)} entries verified"
