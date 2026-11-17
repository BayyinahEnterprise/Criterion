"""
Public API for Criterion.

This is the user-facing module. Everything else is internal.

Three modes for three use cases:

    "rag"     — RAG verification: is the LLM answer grounded in retrieved docs?
                Strict on additions (fabricated content), lenient on omissions
                (the answer doesn't need to include everything).

    "summary" — Summary verification: does the summary faithfully represent the source?
                Lenient on omissions and compression (that's the job of a summary),
                strict on additions and strengthening (those change meaning).

    "strict"  — Strict comparison: flag any deviation from the source.
                Used for contracts, legal, medical, regulatory content.
                Every change matters.

    None      — Auto-detect: chooses the right mode based on the content.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from criterion.capture import capture, CapturedPair
from criterion.divergence import (
    find_divergences,
    Divergence,
    BondDivergence,
    DivergenceReport,
    DivergenceType,
    Severity,
)
from criterion.verdict import render_verdict, Verdict, TrustLevel
from criterion.record import preserve, RecordEntry


class Mode(Enum):
    """Verification mode — determines how strict Criterion is."""
    RAG = "rag"
    SUMMARY = "summary"
    STRICT = "strict"
    AUTO = "auto"


@dataclass(frozen=True)
class VerifyResult:
    """The result of a Criterion verification.

    This is the primary return type. Everything a user needs is here.
    """

    # Trust assessment
    trust_score: float
    trust_level: str           # "faithful", "adapted", "divergent", "unreliable"
    is_trustworthy: bool       # True if faithful or adapted

    # What went wrong
    divergences: tuple[Divergence, ...]
    broken_bonds: tuple[BondDivergence, ...]
    total_divergences: int

    # Counts by severity
    critical_count: int
    major_count: int
    minor_count: int

    # Human-readable summary
    summary: str

    # Actionable: what to fix and how to verify
    corrections: tuple[str, ...]  # suggested_correction for each divergence

    # The mode that was used
    mode: str

    # Record entry (if recording is enabled)
    record_id: str | None

    # Raw objects for advanced use
    _verdict: Verdict
    _report: DivergenceReport
    _pair: CapturedPair

    def to_dict(self) -> dict:
        """Serialize to a dictionary for JSON output."""
        return {
            "trust_score": self.trust_score,
            "trust_level": self.trust_level,
            "is_trustworthy": self.is_trustworthy,
            "total_divergences": self.total_divergences,
            "critical": self.critical_count,
            "major": self.major_count,
            "minor": self.minor_count,
            "broken_bonds": len(self.broken_bonds),
            "summary": self.summary,
            "mode": self.mode,
            "record_id": self.record_id,
            "divergences": [
                {
                    "type": d.dtype.value,
                    "severity": d.severity.value,
                    "explanation": d.explanation,
                    "confidence": d.confidence,
                    "suggested_correction": d.suggested_correction,
                    "verification_method": d.verification_method,
                    "source_text": d.source_atom.text if d.source_atom else None,
                    "output_text": d.output_atom.text if d.output_atom else None,
                }
                for d in self.divergences
            ],
        }


def _auto_detect_mode(source: str, output: str) -> Mode:
    """Guess the right mode from content characteristics."""
    src_len = len(source.split())
    out_len = len(output.split())

    if out_len == 0 or src_len == 0:
        return Mode.STRICT

    ratio = out_len / src_len

    # If output is much shorter than source, it's likely a summary
    if ratio < 0.4:
        return Mode.SUMMARY

    # If source looks like retrieved documents (long, multi-paragraph)
    # and output is a focused answer, it's RAG
    if src_len > 200 and ratio < 0.6:
        return Mode.RAG

    # Default to strict for similar-length content
    return Mode.STRICT


def verify(
    source: str,
    output: str,
    mode: str | Mode | None = None,
    record_path: Path | str | None = None,
    record: bool = True,
) -> VerifyResult:
    """Verify an AI-generated output against its source.

    This is the primary entry point for Criterion.

    Args:
        source: The original source material (retrieved docs, article,
                original text, etc.)
        output: The AI-generated output to verify (answer, summary,
                revised text, etc.)
        mode: Verification mode. One of "rag", "summary", "strict",
              "auto", or None (auto-detect). See Mode enum for details.
        record_path: Path for the append-only record. Set record=False
                     to skip recording.
        record: Whether to write to the append-only record.

    Returns:
        A VerifyResult with trust assessment, divergences, and
        suggested corrections.

    Examples::

        # Basic usage — auto-detects mode
        result = criterion.verify(
            source="The earth orbits the sun in 365.25 days.",
            output="The earth orbits the sun in about a year.",
        )

        # RAG verification
        result = criterion.verify(
            source=retrieved_documents,
            output=llm_answer,
            mode="rag",
        )

        # Summary verification
        result = criterion.verify(
            source=long_article,
            output=ai_summary,
            mode="summary",
        )

        # CI gate — exit code 0 if trustworthy
        result = criterion.verify(source=src, output=out, mode="strict")
        sys.exit(0 if result.is_trustworthy else 1)
    """
    # Resolve mode
    if mode is None or mode == "auto":
        resolved_mode = _auto_detect_mode(source, output)
    elif isinstance(mode, str):
        resolved_mode = Mode(mode)
    else:
        resolved_mode = mode

    # Run the pipeline
    pair = capture(source, output)
    report = find_divergences(pair)
    verdict = render_verdict(report)

    # Apply mode-specific filtering
    filtered_divs = _apply_mode(verdict.divergences, resolved_mode)
    filtered_bond_divs = verdict.bond_divergences

    # Re-score with filtered divergences if mode changed anything
    if len(filtered_divs) != len(verdict.divergences):
        from criterion.verdict import _compute_trust_score, _classify_trust, TrustLevel
        score = _compute_trust_score(filtered_divs, filtered_bond_divs)
        crit = sum(1 for d in filtered_divs if d.severity == Severity.CRITICAL)
        maj = sum(1 for d in filtered_divs if d.severity == Severity.MAJOR)
        level = _classify_trust(score, crit, maj, len(filtered_bond_divs))
        is_trust = level in (TrustLevel.FAITHFUL, TrustLevel.ADAPTED)
    else:
        score = verdict.trust_score
        level = verdict.trust_level
        is_trust = verdict.is_trustworthy
        crit = verdict.critical_count
        maj = verdict.major_count

    # Record
    record_id = None
    if record:
        entry = preserve(pair, report, verdict, record_path)
        record_id = entry.entry_id

    minor = sum(1 for d in filtered_divs if d.severity == Severity.MINOR)

    return VerifyResult(
        trust_score=score,
        trust_level=level.value,
        is_trustworthy=is_trust,
        divergences=filtered_divs,
        broken_bonds=filtered_bond_divs,
        total_divergences=len(filtered_divs) + len(filtered_bond_divs),
        critical_count=crit,
        major_count=maj,
        minor_count=minor,
        summary=verdict.summary,
        corrections=tuple(
            d.suggested_correction for d in filtered_divs
            if d.severity in (Severity.CRITICAL, Severity.MAJOR)
        ),
        mode=resolved_mode.value,
        record_id=record_id,
        _verdict=verdict,
        _report=report,
        _pair=pair,
    )


def _apply_mode(
    divergences: tuple[Divergence, ...],
    mode: Mode,
) -> tuple[Divergence, ...]:
    """Filter divergences based on the verification mode."""

    if mode == Mode.STRICT:
        # Keep everything
        return divergences

    if mode == Mode.SUMMARY:
        # In summary mode, omissions and compression are expected.
        # Only keep: additions (fabricated content), strengthening
        # (overclaiming), and broken bonds (which come through separately).
        result = []
        for d in divergences:
            if d.dtype == DivergenceType.OMISSION:
                # Downgrade all omissions to cosmetic in summary mode
                continue
            if d.dtype in (DivergenceType.CONCEALING, DivergenceType.SURFACING):
                if d.severity in (Severity.MINOR, Severity.COSMETIC):
                    continue  # Expected in summaries
            result.append(d)
        return tuple(result)

    if mode == Mode.RAG:
        # In RAG mode, the answer doesn't need to include everything
        # from the retrieved docs. Only flag:
        # - Additions (hallucinated content not in docs)
        # - Strengthening (overclaiming from docs)
        # - Weakening that changes meaning (not just hedging)
        result = []
        for d in divergences:
            if d.dtype == DivergenceType.OMISSION:
                continue  # Expected — answer is selective
            if d.dtype in (DivergenceType.CONCEALING, DivergenceType.DEFERRING):
                continue  # Answer can compress and reorder
            if d.dtype == DivergenceType.SURFACING and d.severity <= Severity.MINOR:
                continue  # Minor rewording is fine
            result.append(d)
        return tuple(result)

    return divergences


def verify_batch(
    pairs: Sequence[tuple[str, str]],
    mode: str | Mode | None = None,
    record_path: Path | str | None = None,
    record: bool = True,
) -> list[VerifyResult]:
    """Verify multiple source-output pairs.

    Args:
        pairs: List of (source, output) tuples.
        mode: Verification mode applied to all pairs.
        record_path: Shared record path for all verifications.
        record: Whether to write to the append-only record.

    Returns:
        List of VerifyResult, one per pair.
    """
    return [
        verify(src, out, mode=mode, record_path=record_path, record=record)
        for src, out in pairs
    ]
