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

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from criterion.capture import capture, CapturedPair
from criterion.divergence import (
    agreement,
    find_divergences,
    Thresholds,
    Divergence,
    BondDivergence,
    DivergenceReport,
    DivergenceType,
    Severity,
)
from criterion.verdict import (
    render_verdict, Verdict, TrustLevel,
    _compute_trust_score, _classify_trust, _compute_direction,
)
from criterion.record import preserve, RecordEntry


class CapabilityError(RuntimeError):
    """A detector the caller required (e.g. NLI) did not run."""


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


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

    # Which detectors actually ran: {"nli": bool, "semantic": bool, ...}.
    # If "nli" is False, meaning-level contradictions and fabricated
    # claims were NOT checked and a clean verdict means much less.
    capabilities: dict[str, Any] = field(default_factory=dict)

    # Per output sentence NLI judged: {"text", "support"}; support is the
    # best entailment (0-1) any relevant source passage gives that sentence.
    sentence_support: tuple[dict[str, Any], ...] = ()
    # The weakest sentence's support — one number to threshold or chart.
    # None when NLI did not run or judged nothing (e.g. fragment answers).
    groundedness: float | None = None
    # The thresholds this verdict was decided with.
    thresholds: dict[str, float] = field(default_factory=dict)
    # How far to trust the verdict, when two grounding models scored it:
    #   "confident"  they agree on every sentence, each well clear of the threshold
    #   "probable"   they agree on every sentence, at least one of them narrowly
    #   "unsure"     they disagree on at least one sentence — the verdict there
    #                is no better than a coin flip and a person should look
    #   None         only one model ran (or nothing was scored)
    # The verdict itself is decided by the models' average either way; this
    # says how much weight it will bear. Each sentence's own label and both
    # scores are in sentence_support.
    confidence: str | None = None
    unsure_count: int = 0

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
            "capabilities": dict(self.capabilities),
            "groundedness": self.groundedness,
            "sentence_support": list(self.sentence_support),
            "thresholds": dict(self.thresholds),
            "confidence": self.confidence,
            "unsure_count": self.unsure_count,
            "divergences": [
                {
                    "type": d.dtype.value,
                    "code": d.code,
                    "alignment": d.alignment,
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
    *,
    require_nli: bool | None = None,
    store_text: bool = True,
    thresholds: Thresholds | dict[str, float] | None = None,
    question: str | None = None,
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
        require_nli: Default True: raise CapabilityError when the NLI
                     model did not run, because without it Criterion
                     cannot see contradictions or fabricated claims.
                     Pass False (or set CRITERION_REQUIRE_NLI=0) to run
                     heuristics only; a clean result is then reported as
                     "unverified" and is never trustworthy.
        thresholds: A Thresholds object or a dict such as
                    {"fabrication": 0.05}. `fabrication` is the main
                    precision/recall dial: an unsupported sentence blocks
                    trust on its own only when its best source support is
                    below it. Higher flags more; lower flags less.
        store_text: If False, the record keeps only SHA-256 hashes of
                    source and output, not the texts (sensitive data).

    Returns:
        A VerifyResult with trust assessment, divergences, suggested
        corrections, and `capabilities` (which detectors actually ran).

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
    if not isinstance(source, str) or not isinstance(output, str):
        raise TypeError("source and output must be strings")
    if require_nli is None:
        env = os.environ.get("CRITERION_REQUIRE_NLI", "").strip().lower()
        require_nli = env not in ("0", "false", "no", "off")
    if isinstance(thresholds, dict):
        thresholds = Thresholds(**thresholds)
    elif thresholds is None:
        thresholds = Thresholds()

    # Resolve mode
    if mode is None or mode == "auto" or mode == Mode.AUTO:
        resolved_mode = _auto_detect_mode(source, output)
    elif isinstance(mode, str):
        resolved_mode = Mode(mode)
    else:
        resolved_mode = mode

    # Empty input invariants — a trust layer cannot call empty output "faithful"
    source_empty = not source.strip()
    output_empty = not output.strip()
    if source_empty or output_empty:
        if source_empty and output_empty:
            reason = "Both source and output are empty"
        elif output_empty:
            reason = "Output is empty. No content to verify against source."
        else:
            reason = "Source is empty. Output cannot be verified as grounded."
        return _degenerate_result(
            resolved_mode, reason, source, output,
            # Nothing to record when there is no input at all.
            do_record=record and not (source_empty and output_empty),
            record_path=record_path, store_text=store_text,
        )

    # Run the pipeline. The mode is passed down so that detection leniency
    # is the caller's decision, never a side effect of the output's length.
    pair = capture(source, output)
    if question is not None and not isinstance(question, str):
        raise TypeError("question must be a string")
    report = find_divergences(
        pair, mode=resolved_mode.value, thresholds=thresholds, question=question,
    )
    capabilities = dict(report.capabilities)
    # "Model-checked" means the grounding scorer ran. An NLI model alone does
    # not count: it is several points weaker at this job, and falling back to
    # it without saying so would be exactly the silent downgrade this guards
    # against. (CRITERION_SCORER=nli selects it as the scorer explicitly.)
    model_checked = bool(capabilities.get("scorer"))
    capabilities["model_checked"] = model_checked
    if require_nli and not model_checked:
        raise CapabilityError(
            "The grounding model did not run "
            f"({capabilities.get('scorer_error') or 'unavailable'}), so contradictions and "
            "fabricated claims cannot be checked. Install it with "
            "`pip install criterion-ai[nli]`, or pass require_nli=False "
            "(CLI: --allow-degraded) to run heuristics only; clean results are "
            "then reported as 'unverified'."
        )
    verdict = render_verdict(report)

    # Apply mode-specific filtering
    filtered_divs = _apply_mode(verdict.divergences, resolved_mode)
    if resolved_mode == Mode.RAG:
        # A RAG answer is selective, so dropped evidence does not block.
        # It is still reported: the reader should know the answer kept a
        # claim and left its support behind.
        filtered_divs += tuple(
            Divergence(
                dtype=DivergenceType.CONCEALING,
                # Informational: it tells the reader what the answer left
                # behind, and must not lower the trust level by itself.
                severity=Severity.COSMETIC,
                source_atom=bd.source_atom,
                output_atom=None,
                explanation=bd.explanation,
                confidence=0.3,
                suggested_correction=f"Consider citing: {bd.source_atom.text}",
                verification_method="Check whether the answer's claim still stands without this evidence",
                code="evidence_not_carried",
            )
            for bd in verdict.bond_divergences
        )
        filtered_bond_divs: tuple[BondDivergence, ...] = ()
    else:
        filtered_bond_divs = verdict.bond_divergences

    # Score what is actually returned
    score = _compute_trust_score(filtered_divs, filtered_bond_divs)
    crit = sum(1 for d in filtered_divs if d.severity == Severity.CRITICAL)
    maj = sum(1 for d in filtered_divs if d.severity == Severity.MAJOR)
    minor = sum(1 for d in filtered_divs if d.severity == Severity.MINOR)
    level = _classify_trust(score, crit, maj, len(filtered_bond_divs))
    if not model_checked and level in (TrustLevel.FAITHFUL, TrustLevel.ADAPTED):
        # Heuristics found nothing blocking, but the detector that looks for
        # contradictions and fabrications never ran. That is not "faithful".
        level = TrustLevel.UNVERIFIED
    is_trust = level in (TrustLevel.FAITHFUL, TrustLevel.ADAPTED)
    summary = _build_filtered_summary(level, score, filtered_divs, filtered_bond_divs)
    if not model_checked:
        summary += " [NLI did not run: contradictions and fabricated claims were not checked.]"
    out_text = {a.index: a.text for a in pair.output.atoms}
    def _sentence(i: int, sc: float) -> dict[str, Any]:
        row: dict[str, Any] = {"text": out_text[i], "support": sc}
        models = report.support_models.get(i)
        if models:
            row["models"] = dict(models)
            row["confidence"] = agreement(models, thresholds.grounded, thresholds.confident)
        return row

    sentence_support = tuple(
        _sentence(i, sc) for i, sc in report.support.items() if i in out_text
    )
    groundedness = min((x["support"] for x in sentence_support), default=None)
    labels = [x["confidence"] for x in sentence_support if x.get("confidence")]
    unsure_count = labels.count("unsure")
    confidence = (
        None if not labels or not model_checked
        else "unsure" if unsure_count
        else "probable" if "probable" in labels
        else "confident"
    )
    if unsure_count:
        summary += (
            f" [Unsure: the two grounding models disagree on {unsure_count} "
            f"sentence{'s' if unsure_count != 1 else ''}; a person should check "
            f"{'them' if unsure_count != 1 else 'it'}.]"
        )
    threshold_dict = {
        "grounded": thresholds.grounded, "contradiction": thresholds.contradiction,
        "fabrication": thresholds.fabrication, "equivalence": thresholds.equivalence,
        "realign": thresholds.realign, "verbatim": thresholds.verbatim,
        "confident": thresholds.confident,
    }

    # Record — store the FILTERED verdict so record matches what was returned
    record_id = None
    if record:
        filtered_verdict = Verdict(
            trust_score=score,
            trust_level=level,
            total_divergences=len(filtered_divs) + len(filtered_bond_divs),
            critical_count=crit,
            major_count=maj,
            minor_count=minor,
            cosmetic_count=sum(1 for d in filtered_divs if d.severity == Severity.COSMETIC),
            broken_bonds=len(filtered_bond_divs),
            direction=_compute_direction(filtered_divs),
            summary=summary,
            divergences=filtered_divs,
            bond_divergences=filtered_bond_divs,
        )
        entry = preserve(
            pair, report, filtered_verdict, record_path,
            mode=resolved_mode.value,
            detector_config=_detector_config(capabilities, threshold_dict),
            store_text=store_text,
        )
        record_id = entry.entry_id

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
        summary=summary,
        corrections=tuple(
            d.suggested_correction for d in filtered_divs
            if d.severity in (Severity.CRITICAL, Severity.MAJOR)
        ),
        mode=resolved_mode.value,
        record_id=record_id,
        _verdict=verdict,
        _report=report,
        _pair=pair,
        capabilities=capabilities,
        sentence_support=sentence_support,
        groundedness=groundedness,
        thresholds=threshold_dict,
        confidence=confidence,
        unsure_count=unsure_count,
    )


def _detector_config(
    capabilities: dict[str, Any], thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """What produced this verdict: detectors that ran, models, thresholds."""
    config: dict[str, Any] = {
        "scorer": bool(capabilities.get("scorer")),
        "scorer_error": capabilities.get("scorer_error"),
        "nli": bool(capabilities.get("nli")),
        "semantic": bool(capabilities.get("semantic")),
        "nli_error": capabilities.get("nli_error"),
        "semantic_error": capabilities.get("semantic_error"),
        "thresholds": dict(thresholds) if thresholds else None,
    }
    try:
        from criterion.grounding import scorer_info as _scorer_info
        config["scorer_model"] = _scorer_info()
    except ImportError:
        config["scorer_model"] = None
    try:
        from criterion.nli import model_info as _nli_info
        config["nli_model"] = _nli_info()
    except ImportError:
        config["nli_model"] = None
    try:
        from criterion.semantic import model_info as _sem_info
        config["semantic_model"] = _sem_info()
    except ImportError:
        config["semantic_model"] = None
    return config


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
        # Keep: additions, strengthening, weakening, critical findings.
        # Drop: omissions, minor compression/surfacing.
        # Exception: numeric OMISSIONS (numbers dropped, not changed)
        # are expected compression in summaries.
        result = []
        for d in divergences:
            # Omissions are ALWAYS expected in summaries, even critical ones
            if d.dtype == DivergenceType.OMISSION:
                continue
            # Numbers dropped (not changed) are compression. Keyed off the
            # finding code, not the wording of the explanation.
            if d.code == "numeric_dropped":
                continue
            # Keep critical findings (numeric changes, negation inversions, etc.)
            if d.severity == Severity.CRITICAL:
                result.append(d)
                continue
            # Keep strengthening/weakening — these change meaning
            if d.dtype in (DivergenceType.STRENGTHENING, DivergenceType.WEAKENING):
                result.append(d)
                continue
            # Keep major additions — these are fabrications
            if d.dtype == DivergenceType.ADDITION and d.severity in (Severity.CRITICAL, Severity.MAJOR):
                result.append(d)
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
        # - Any CRITICAL divergence (numeric change, negation inversion)
        result = []
        for d in divergences:
            # RAG answers are SELECTIVE — omissions are always expected,
            # even critical ones (the answer picks what's relevant)
            if d.dtype == DivergenceType.OMISSION:
                continue
            # Keep critical non-omission findings (numeric changes, negation inversions)
            if d.severity == Severity.CRITICAL:
                result.append(d)
                continue
            if d.dtype in (DivergenceType.CONCEALING, DivergenceType.DEFERRING):
                continue  # Answer can compress and reorder
            if d.dtype == DivergenceType.SURFACING and d.severity in (Severity.MINOR, Severity.COSMETIC):
                continue  # Minor rewording is fine
            result.append(d)
        return tuple(result)

    return divergences


def verify_batch(
    pairs: Sequence[tuple[str, str]],
    mode: str | Mode | None = None,
    record_path: Path | str | None = None,
    record: bool = True,
    **kwargs: Any,
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
        verify(src, out, mode=mode, record_path=record_path, record=record, **kwargs)
        for src, out in pairs
    ]


def _build_filtered_summary(
    level, score, divergences, bond_divergences,
) -> str:
    """Build a summary from the filtered (mode-applied) findings."""
    n = len(divergences) + len(bond_divergences)
    if n == 0:
        return f"The output faithfully represents the source. Trust: {level.value} ({score:.0%})."
    parts = [f"{n} divergence{'s' if n != 1 else ''} detected."]
    crit = sum(1 for d in divergences if d.severity == Severity.CRITICAL)
    maj = sum(1 for d in divergences if d.severity == Severity.MAJOR)
    if crit:
        parts.append(f"{crit} critical.")
    if maj:
        parts.append(f"{maj} major.")
    if bond_divergences:
        parts.append(f"{len(bond_divergences)} broken bond{'s' if len(bond_divergences) != 1 else ''}.")
    parts.append(f"Trust: {level.value} ({score:.0%}).")
    return " ".join(parts)


def _degenerate_result(
    mode: Mode, reason: str, source: str, output: str,
    *, do_record: bool, record_path, store_text: bool,
) -> VerifyResult:
    """Empty source and/or output: never trustworthy, and still recorded.

    An empty output is exactly the kind of event an audit trail must
    show, so it is written to the record like any other verdict.
    """
    record_id = None
    if do_record:
        pair = capture(source, output)
        verdict = Verdict(
            trust_score=0.0, trust_level=TrustLevel.UNRELIABLE,
            total_divergences=0, critical_count=0, major_count=0,
            minor_count=0, cosmetic_count=0, broken_bonds=0,
            direction=_compute_direction(()), summary=reason,
            divergences=(), bond_divergences=(),
        )
        entry = preserve(
            pair, None, verdict, record_path,
            mode=mode.value, detector_config=_detector_config({}),
            store_text=store_text,
        )
        record_id = entry.entry_id
    return VerifyResult(
        trust_score=0.0,
        trust_level="unreliable",
        is_trustworthy=False,
        divergences=(),
        broken_bonds=(),
        total_divergences=0,
        critical_count=0,
        major_count=0,
        minor_count=0,
        summary=reason,
        corrections=(),
        mode=mode.value,
        record_id=record_id,
        _verdict=None,
        _report=None,
        _pair=None,
    )
