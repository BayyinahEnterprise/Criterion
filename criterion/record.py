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
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from criterion.capture import CapturedPair
from criterion.divergence import DivergenceReport
from criterion.verdict import Verdict


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
    source_text: str           # The full source text
    output_text: str           # The full output text

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
    """Hash content fields to produce a stable entry ID."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_DEFAULT_RECORD_PATH = Path(".criterion/record.jsonl")


def _get_last_hash(path: Path) -> str:
    """Read the hash of the last entry in the record file.

    Returns "genesis" if the file is empty or does not exist.
    """
    if not path.exists():
        return "genesis"

    last_line = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if not last_line:
        return "genesis"

    try:
        entry = json.loads(last_line)
        return entry.get("entry_id", "genesis")
    except (json.JSONDecodeError, KeyError):
        return "genesis"


def _write_entry(entry: RecordEntry, path: Path) -> None:
    """Append one entry to the record file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    record_dict = {
        "entry_id": entry.entry_id,
        "recorded_at": entry.recorded_at,
        "predecessor_hash": entry.predecessor_hash,
        "source_hash": entry.source_hash,
        "output_hash": entry.output_hash,
        "source_text": entry.source_text,
        "output_text": entry.output_text,
        "trust_score": entry.trust_score,
        "trust_level": entry.trust_level,
        "total_divergences": entry.total_divergences,
        "divergences": entry.divergences,
        "bond_divergences": entry.bond_divergences,
        "broken_bonds": entry.broken_bonds,
        "summary": entry.summary,
        "criterion_version": entry.criterion_version,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preserve(
    pair: CapturedPair,
    report: DivergenceReport,
    verdict: Verdict,
    record_path: Path | str | None = None,
) -> RecordEntry:
    """Preserve a complete Criterion assessment to the append-only record.

    This is the fourth and final step in the kernel pipeline.
    After verdict, record. Nothing is lost.

    Args:
        pair: The captured source-output pair.
        report: The divergence report.
        verdict: The trust verdict.
        record_path: Where to store the record. Defaults to .criterion/record.jsonl

    Returns:
        The RecordEntry that was written.
    """
    path = Path(record_path) if record_path else _DEFAULT_RECORD_PATH

    now = datetime.now(timezone.utc).isoformat()
    predecessor = _get_last_hash(path)

    divs_serialized = [_serialize_divergence(d) for d in verdict.divergences]
    bond_divs_serialized = [_serialize_bond_divergence(bd) for bd in verdict.bond_divergences]

    entry_id = _hash_content(
        pair.source.sha256,
        pair.output.sha256,
        now,
        predecessor,
    )

    entry = RecordEntry(
        entry_id=entry_id,
        recorded_at=now,
        predecessor_hash=predecessor,
        source_hash=pair.source.sha256,
        output_hash=pair.output.sha256,
        source_text=pair.source.raw_text,
        output_text=pair.output.raw_text,
        trust_score=verdict.trust_score,
        trust_level=verdict.trust_level.value,
        total_divergences=verdict.total_divergences,
        divergences=divs_serialized,
        bond_divergences=bond_divs_serialized,
        broken_bonds=verdict.broken_bonds,
        summary=verdict.summary,
        criterion_version="0.1.0",
    )

    _write_entry(entry, path)
    return entry


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
            entries.append(RecordEntry(**data))
    return entries


def verify_chain(record_path: Path | str | None = None) -> tuple[bool, str]:
    """Verify the append-only hash chain is intact.

    Returns (True, "ok") if the chain is valid, or
    (False, "description of break") if tampered.
    """
    entries = read_record(record_path)
    if not entries:
        return True, "empty record"

    if entries[0].predecessor_hash != "genesis":
        return False, f"first entry predecessor is {entries[0].predecessor_hash!r}, expected 'genesis'"

    for i in range(1, len(entries)):
        if entries[i].predecessor_hash != entries[i - 1].entry_id:
            return False, (
                f"entry {i} predecessor {entries[i].predecessor_hash!r} "
                f"does not match entry {i-1} id {entries[i-1].entry_id!r}"
            )

    return True, f"chain intact, {len(entries)} entries"
