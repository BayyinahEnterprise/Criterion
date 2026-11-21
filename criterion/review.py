"""
Criterion Review — put a person where the machine is weakest, and keep score.

Verdicts are wrong often enough that a person has to look at some of them.
This module answers three questions from the record alone:

    queue    which sentences should a person look at, and in what order?
    decide   what did they conclude?  (appended to the same hash chain)
    stats    how often did the machine and the reviewers agree?

Order of the queue. A reviewer's time is worth most where the machine's
verdict is worth least. Measured on LLM-AggreFact, verdicts were right about
59% of the time when the two grounding models disagreed ("unsure"), 69% when
they agreed narrowly ("probable") and 89% when they agreed clearly
("confident"). So the queue runs unsure, then probable, then confident, and
within each from the score nearest the threshold outward.

What is queued by default: every sentence the machine did not clear, and
every sentence it cleared without being confident. Confident, supported
sentences are left out — there are too many of them — except for a
spot-check sample, which is what makes the machine's error rate on *your*
data measurable. The sample is chosen by hash, not by a random number, so
the same record always yields the same sample.

Nothing here changes a verdict. A decision is a new entry that names the
verdict it concerns; the verdict stays as the machine wrote it, so the
record shows both what the machine said and what a person decided.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from criterion.record import DECISIONS, RecordEntry, preserve_review, read_record

_TIER_ORDER = {"unsure": 0, None: 1, "probable": 2, "confident": 3}
_DEFAULT_GROUNDED = 0.5


@dataclass(frozen=True)
class ReviewItem:
    """One sentence (or one whole output) a reviewer is asked to judge."""

    entry_id: str
    sentence: int | None        # None: the output as a whole
    text: str | None            # None when the record was written without text
    evidence: str | None        # the source passage that supported it best
    support: float | None       # averaged model score, 0-1
    models: dict[str, float] | None
    confidence: str | None      # "confident" | "probable" | "unsure" | None
    machine: str                # "supported" | "unsupported"
    reason: str                 # why it is in the queue
    recorded_at: str
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _grounded(entry: RecordEntry) -> float:
    thresholds = (entry.detector_config or {}).get("thresholds") or {}
    value = thresholds.get("grounded")
    return float(value) if isinstance(value, (int, float)) else _DEFAULT_GROUNDED


def _sampled(entry_id: str, sentence: int | None, fraction: float) -> bool:
    """Deterministic spot-check sample: the same record gives the same sample."""
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    digest = hashlib.sha256(f"criterion-spot-check\n{entry_id}\n{sentence}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64 < fraction


def _decisions(entries: list[RecordEntry]) -> dict[tuple[str, int | None], RecordEntry]:
    """Latest decision per (verdict, sentence). A later review supersedes an earlier one."""
    latest: dict[tuple[str, int | None], RecordEntry] = {}
    for e in entries:
        if e.kind == "review" and isinstance(e.review, dict):
            latest[(e.review.get("of"), e.review.get("sentence"))] = e
    return latest


def _items(entry: RecordEntry) -> list[ReviewItem]:
    """Every reviewable unit of one verdict, with the machine's call on each."""
    common = dict(entry_id=entry.entry_id, recorded_at=entry.recorded_at, mode=entry.mode)
    rows = entry.sentences or []
    if not rows:
        # Nothing was scored sentence by sentence (heuristics only, or a
        # verdict decided by deterministic findings): review the whole output.
        cleared = entry.trust_level in ("faithful", "adapted")
        return [ReviewItem(
            sentence=None, text=entry.output_text, evidence=None, support=None,
            models=None, confidence=None,
            machine="supported" if cleared else "unsupported",
            reason="", **common,
        )]
    grounded = _grounded(entry)
    texts = entry.sentence_texts or [{}] * len(rows)
    out = []
    for row, t in zip(rows, texts):
        support = row.get("support")
        out.append(ReviewItem(
            sentence=row["index"], text=t.get("text"), evidence=t.get("evidence"),
            support=support, models=row.get("models"), confidence=row.get("confidence"),
            machine="supported" if isinstance(support, (int, float)) and support >= grounded else "unsupported",
            reason="", **common,
        ))
    return out


def _reason(item: ReviewItem, spot_check: float) -> str | None:
    """Why this item is queued, or None if it is not."""
    if item.confidence == "unsure":
        return "the two models disagree"
    if item.machine == "unsupported":
        return "the machine found it unsupported"
    if item.confidence == "probable":
        return "the models agree only narrowly"
    if item.confidence is None:
        return "only one model judged it"
    if _sampled(item.entry_id, item.sentence, spot_check):
        return "spot check of a confident verdict"
    return None


def queue(
    record_path: Path | str | None = None,
    *,
    spot_check: float = 0.05,
    include_reviewed: bool = False,
    everything: bool = False,
) -> list[ReviewItem]:
    """The sentences a person should look at, most doubtful first.

    Args:
        record_path: The record to read. Defaults to .criterion/record.jsonl.
        spot_check: Share (0-1) of confident, supported sentences to include
            anyway. Without some, the error rate of the machine's confident
            verdicts can never be measured. Default 5%.
        include_reviewed: Also list sentences that already have a decision.
        everything: Queue every recorded sentence, whatever its label.
    """
    if not 0 <= spot_check <= 1:
        raise ValueError("spot_check must be between 0 and 1")
    entries = read_record(record_path)
    decided = _decisions(entries)
    grounded_of = {e.entry_id: _grounded(e) for e in entries if e.kind == "verdict"}
    picked: list[ReviewItem] = []
    for entry in entries:
        if entry.kind != "verdict":
            continue
        for item in _items(entry):
            if not include_reviewed and (item.entry_id, item.sentence) in decided:
                continue
            reason = _reason(item, 1.0 if everything else spot_check)
            if reason is None:
                continue
            picked.append(ReviewItem(**{**item.to_dict(), "reason": reason}))

    def priority(item: ReviewItem) -> tuple:
        distance = abs(item.support - grounded_of[item.entry_id]) if item.support is not None else 0.0
        return (_TIER_ORDER.get(item.confidence, 1), distance, item.recorded_at, item.sentence or 0)

    return sorted(picked, key=priority)


def _find(entries: list[RecordEntry], entry_id: str) -> RecordEntry:
    """A verdict by full id or unambiguous prefix (at least 8 characters)."""
    if not isinstance(entry_id, str) or len(entry_id) < 8:
        raise ValueError("give at least the first 8 characters of the entry id")
    matches = [e for e in entries if e.entry_id.startswith(entry_id)]
    if not matches:
        raise ValueError(f"no entry in this record starts with {entry_id!r}")
    if len(matches) > 1:
        raise ValueError(f"{entry_id!r} matches {len(matches)} entries; give more characters")
    return matches[0]


def decide(
    entry_id: str,
    decision: str,
    reviewer: str,
    *,
    sentence: int | None = None,
    note: str = "",
    record_path: Path | str | None = None,
    key: bytes | str | None = None,
) -> RecordEntry:
    """Record a reviewer's decision about one sentence of a recorded verdict.

    `decision` is "supported", "unsupported" or "unclear". The decision is
    appended to the record; the verdict is not modified. Deciding again later
    appends again, and the later decision is the one `stats` counts — the
    earlier one stays in the record.
    """
    target = _find(read_record(record_path), entry_id)
    if sentence is None and target.sentences:
        raise ValueError(
            f"this verdict has {len(target.sentences)} recorded sentences; say which one with sentence="
        )
    return preserve_review(
        target, sentence=sentence, decision=decision, reviewer=reviewer,
        note=note, record_path=record_path, key=key,
    )


def stats(record_path: Path | str | None = None) -> dict[str, Any]:
    """How the machine's verdicts compare with the reviewers' decisions.

    For each confidence label: how many sentences were reviewed and on how
    many the reviewer agreed with the machine. "unclear" decisions are
    counted apart; they are neither agreement nor disagreement.

    Read the confident row with care. Reviewed sentences are not a random
    sample of all sentences — doubtful ones are queued first — so only the
    spot-checked share says anything about confident verdicts in general.
    """
    entries = read_record(record_path)
    decided = _decisions(entries)
    tiers: dict[str, dict[str, int]] = {}
    total = reviewed = 0
    for entry in entries:
        if entry.kind != "verdict":
            continue
        for item in _items(entry):
            total += 1
            review = decided.get((item.entry_id, item.sentence))
            label = item.confidence or "single model"
            row = tiers.setdefault(label, {
                "sentences": 0, "reviewed": 0, "agreed": 0, "disagreed": 0, "unclear": 0,
                "machine_supported_human_unsupported": 0, "machine_unsupported_human_supported": 0,
            })
            row["sentences"] += 1
            if review is None:
                continue
            reviewed += 1
            row["reviewed"] += 1
            human = review.review["decision"]
            if human == "unclear":
                row["unclear"] += 1
            elif human == item.machine:
                row["agreed"] += 1
            else:
                row["disagreed"] += 1
                row[f"machine_{item.machine}_human_{human}"] += 1
    for row in tiers.values():
        judged = row["agreed"] + row["disagreed"]
        row["agreement"] = round(row["agreed"] / judged, 4) if judged else None
    order = ["unsure", "probable", "confident", "single model"]
    return {
        "sentences": total,
        "reviewed": reviewed,
        "pending_in_default_queue": len(queue(record_path)),
        "by_confidence": {k: tiers[k] for k in order if k in tiers},
        "reviewers": sorted({e.review["reviewer"] for e in decided.values()}),
    }


__all__ = ["DECISIONS", "ReviewItem", "queue", "decide", "stats"]
