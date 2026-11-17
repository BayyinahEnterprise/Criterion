"""
Pipeline: the four kernel modules composed.

    source_text + output_text
        → capture()
        → find_divergences()
        → render_verdict()
        → preserve()
        → RecordEntry

This is the complete Criterion pipeline in one function call.
"""

from __future__ import annotations

from pathlib import Path

from criterion.capture import CapturedPair, capture
from criterion.divergence import DivergenceReport, find_divergences
from criterion.verdict import Verdict, render_verdict
from criterion.record import RecordEntry, preserve


def criterion(
    source_text: str,
    output_text: str,
    record_path: Path | str | None = None,
) -> tuple[Verdict, RecordEntry]:
    """Run the full Criterion pipeline.

    Args:
        source_text: The original source material.
        output_text: The AI-generated output to verify.
        record_path: Where to store the record. Defaults to .criterion/record.jsonl

    Returns:
        A (Verdict, RecordEntry) tuple.

    Example::

        from criterion.pipeline import criterion

        verdict, record = criterion(
            source_text="The contract expires on March 15, 2027.",
            output_text="The contract expires in early 2027.",
        )
        print(verdict.summary)

    This is the mode-less kernel. For RAG / summary / strict policies,
    capability reporting and `require_nli`, use `criterion.verify`.
    """
    pair: CapturedPair = capture(source_text, output_text)
    report: DivergenceReport = find_divergences(pair)
    verdict: Verdict = render_verdict(report)
    entry: RecordEntry = preserve(pair, report, verdict, record_path)
    return verdict, entry
