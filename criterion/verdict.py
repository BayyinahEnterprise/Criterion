"""
Module 3: Verdict (the Reckoner)

Furqan reading order: 3 of 4 (kernel)
Dependency depth: 2 — depends on capture and divergence

Scores the trustworthiness of an AI-generated output based on:
    - Atom-level divergences (content changes)
    - Bond-level divergences (broken structural relationships)
    - The semantic type of affected atoms (claim vs evidence vs observation)
    - The direction of change (inflating vs deflating)

Produces a Verdict: a trust score, a classification, and a
human-readable summary with actionable detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from criterion.divergence import (
    BondDivergence,
    Divergence,
    DivergenceReport,
    DivergenceType,
    Severity,
)


# ---------------------------------------------------------------------------
# Trust classification
# ---------------------------------------------------------------------------

class TrustLevel(Enum):
    FAITHFUL = "faithful"
    ADAPTED = "adapted"
    DIVERGENT = "divergent"
    UNRELIABLE = "unreliable"
    # No problem was found, but the NLI model did not run, so contradictions
    # and fabricated claims were never checked. Never counts as trustworthy.
    UNVERIFIED = "unverified"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHT = {
    Severity.CRITICAL: 1.0,
    Severity.MAJOR: 0.5,
    Severity.MINOR: 0.15,
    Severity.COSMETIC: 0.02,
}

_BOND_BREAK_WEIGHT = 0.8  # A broken evidential bond is almost as bad as a critical finding


@dataclass(frozen=True)
class DirectionBalance:
    inflating_count: int
    deflating_count: int
    inflating_weight: float
    deflating_weight: float

    @property
    def net_direction(self) -> str:
        if self.inflating_weight > self.deflating_weight * 1.2:
            return "inflating"
        elif self.deflating_weight > self.inflating_weight * 1.2:
            return "deflating"
        return "balanced"

    @property
    def bias_ratio(self) -> float:
        total = self.inflating_weight + self.deflating_weight
        if total == 0:
            return 0.5
        return max(self.inflating_weight, self.deflating_weight) / total


@dataclass(frozen=True)
class Verdict:
    """The final trust assessment."""

    trust_score: float
    trust_level: TrustLevel
    total_divergences: int
    critical_count: int
    major_count: int
    minor_count: int
    cosmetic_count: int
    broken_bonds: int
    direction: DirectionBalance
    summary: str
    divergences: tuple[Divergence, ...]
    bond_divergences: tuple[BondDivergence, ...]

    @property
    def is_trustworthy(self) -> bool:
        return self.trust_level in (TrustLevel.FAITHFUL, TrustLevel.ADAPTED)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _compute_trust_score(
    divergences: tuple[Divergence, ...],
    bond_divergences: tuple[BondDivergence, ...],
) -> float:
    if not divergences and not bond_divergences:
        return 1.0

    total_penalty = 0.0
    for d in divergences:
        weight = _SEVERITY_WEIGHT[d.severity]
        total_penalty += weight * d.confidence

    # Broken bonds are penalized heavily
    total_penalty += len(bond_divergences) * _BOND_BREAK_WEIGHT

    return round(1.0 / (1.0 + total_penalty), 4)


def _classify_trust(score: float, critical: int, major: int, broken_bonds: int) -> TrustLevel:
    if critical >= 2 or broken_bonds >= 2:
        return TrustLevel.UNRELIABLE
    if critical > 0 or broken_bonds > 0:
        return TrustLevel.DIVERGENT
    # Without critical/bond issues, need substantial MAJOR divergences to flag
    if major == 0 and score >= 0.90:
        return TrustLevel.FAITHFUL
    if score >= 0.90:
        return TrustLevel.FAITHFUL
    if score >= 0.70:
        return TrustLevel.ADAPTED
    if score >= 0.40:
        return TrustLevel.DIVERGENT
    return TrustLevel.UNRELIABLE


def _compute_direction(divergences: tuple[Divergence, ...]) -> DirectionBalance:
    inf_count = inf_weight = 0
    def_count = def_weight = 0

    for d in divergences:
        w = _SEVERITY_WEIGHT[d.severity] * d.confidence
        if d.direction == "inflating":
            inf_count += 1
            inf_weight += w
        else:
            def_count += 1
            def_weight += w

    return DirectionBalance(
        inflating_count=inf_count,
        deflating_count=def_count,
        inflating_weight=round(inf_weight, 4),
        deflating_weight=round(def_weight, 4),
    )


def _build_summary(
    trust_level: TrustLevel,
    trust_score: float,
    report: DivergenceReport,
    direction: DirectionBalance,
) -> str:
    parts = []
    n = report.total

    if n == 0:
        return (
            f"The output faithfully represents the source. "
            f"No divergences detected across {report.source_atoms_matched} "
            f"aligned atoms."
        )

    parts.append(
        f"{n} divergence{'s' if n != 1 else ''} detected "
        f"across {report.source_atoms_matched} aligned atoms."
    )

    by_type = report.by_type
    type_parts = []
    for dt in DivergenceType:
        count = len(by_type.get(dt, []))
        if count:
            type_parts.append(f"{count} {dt.value}")
    if type_parts:
        parts.append("Types: " + ", ".join(type_parts) + ".")

    if report.broken_bonds_count:
        parts.append(
            f"{report.broken_bonds_count} broken bond{'s' if report.broken_bonds_count != 1 else ''}: "
            f"evidence dropped while claims retained."
        )

    if direction.net_direction != "balanced":
        verb = (
            "added, strengthened, or elevated"
            if direction.net_direction == "inflating"
            else "removed, weakened, or buried"
        )
        parts.append(f"Direction: {direction.net_direction} — the AI {verb} content.")

    parts.append(f"Trust: {trust_level.value} ({trust_score:.0%}).")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_verdict(report: DivergenceReport) -> Verdict:
    """Render a trust verdict from a divergence report.

    Args:
        report: A DivergenceReport from the divergence module.

    Returns:
        A Verdict with trust score, classification, broken bonds, and summary.
    """
    divs = report.divergences
    bond_divs = report.bond_divergences

    by_sev = report.by_severity
    critical = len(by_sev.get(Severity.CRITICAL, []))
    major = len(by_sev.get(Severity.MAJOR, []))
    minor = len(by_sev.get(Severity.MINOR, []))
    cosmetic = len(by_sev.get(Severity.COSMETIC, []))

    score = _compute_trust_score(divs, bond_divs)
    level = _classify_trust(score, critical, major, len(bond_divs))
    direction = _compute_direction(divs)
    summary = _build_summary(level, score, report, direction)

    return Verdict(
        trust_score=score,
        trust_level=level,
        total_divergences=report.total,
        critical_count=critical,
        major_count=major,
        minor_count=minor,
        cosmetic_count=cosmetic,
        broken_bonds=len(bond_divs),
        direction=direction,
        summary=summary,
        divergences=divs,
        bond_divergences=bond_divs,
    )
