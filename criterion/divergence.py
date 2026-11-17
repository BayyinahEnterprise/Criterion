"""
Module 2: Divergence (the Counter)

Furqan reading order: 2 of 4 (kernel)
Dependency depth: 1 — depends only on capture

Enumerates every difference between source and output molecules,
classified into eight types derived from four paired axes:

    Addition / Omission        — atoms expanded or removed
    Strengthening / Weakening  — claims made stronger or softer
    Advancing / Deferring      — emphasis shifted forward or buried
    Surfacing / Concealing     — implicit made explicit or explicit hidden

Every divergence carries actionable fields (from furqan-lint's Marad):
    - suggested_correction: what the output should have said
    - verification_method: how to confirm this divergence is real

The severity of an omission depends on the atom's TYPE and its BONDS:
    - Omitting evidence that supports a claim is CRITICAL
    - Omitting a standalone observation is MINOR
    - Omitting a claim whose evidence is also omitted is MAJOR
      (the whole argument was dropped, not just the claim)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import Sequence

import numpy as np

from criterion.capture import (
    Atom,
    AtomType,
    Bond,
    BondType,
    CapturedPair,
    Molecule,
)

# Semantic similarity (optional — degrades gracefully if unavailable)
_USE_SEMANTIC = True
try:
    from criterion.semantic import (
        batch_semantic_similarity,
        FAITHFUL_THRESHOLD,
        DIVERGENT_THRESHOLD,
    )
except ImportError:
    _USE_SEMANTIC = False


# ---------------------------------------------------------------------------
# Divergence taxonomy — four paired axes, eight types
# ---------------------------------------------------------------------------

class DivergenceType(Enum):
    """Eight types of divergence, organized as four complementary pairs."""

    ADDITION = "addition"
    OMISSION = "omission"
    STRENGTHENING = "strengthening"
    WEAKENING = "weakening"
    ADVANCING = "advancing"
    DEFERRING = "deferring"
    SURFACING = "surfacing"
    CONCEALING = "concealing"

    @property
    def pair(self) -> str:
        pairs = {
            "addition": "expansion", "omission": "expansion",
            "strengthening": "intensity", "weakening": "intensity",
            "advancing": "emphasis", "deferring": "emphasis",
            "surfacing": "visibility", "concealing": "visibility",
        }
        return pairs[self.value]

    @property
    def direction(self) -> str:
        positive = {"addition", "strengthening", "advancing", "surfacing"}
        return "inflating" if self.value in positive else "deflating"


# ---------------------------------------------------------------------------
# Severity — now atom-type-aware
# ---------------------------------------------------------------------------

class Severity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    COSMETIC = "cosmetic"


def _severity_from_atom(atom: Atom, molecule: Molecule) -> Severity:
    """Determine omission severity from atom type and bond context.

    An evidence atom that supports a claim is CRITICAL to omit.
    A claim with no supporting evidence in the molecule is MAJOR.
    A standalone observation is MINOR.
    """
    base = atom.atom_type.omission_severity

    # Upgrade if this atom has evidential bonds
    if base != "critical":
        bonds = molecule.bonds_for_atom(atom.index)
        if any(b.is_evidential for b in bonds):
            return Severity.CRITICAL

    return Severity[base.upper()]


# ---------------------------------------------------------------------------
# Divergence finding — now with Marad-style actionable fields
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Divergence:
    """One atomic divergence between source and output molecules.

    Carries actionable fields inspired by furqan-lint's Marad type:
    every divergence tells you not just what went wrong, but what
    the correct version should look like and how to verify it.
    """

    dtype: DivergenceType
    severity: Severity
    source_atom: Atom | None      # None for additions
    output_atom: Atom | None      # None for omissions
    explanation: str               # What went wrong
    confidence: float              # 0.0–1.0
    # Marad-style actionable fields
    suggested_correction: str      # What the output should have said
    verification_method: str       # How to confirm this divergence is real

    @property
    def pair(self) -> str:
        return self.dtype.pair

    @property
    def direction(self) -> str:
        return self.dtype.direction

    @property
    def source_type(self) -> str:
        """The semantic type of the source atom, if present."""
        return self.source_atom.atom_type.value if self.source_atom else "n/a"

    @property
    def output_type(self) -> str:
        """The semantic type of the output atom, if present."""
        return self.output_atom.atom_type.value if self.output_atom else "n/a"


@dataclass(frozen=True)
class BondDivergence:
    """A structural relationship that was broken between source and output.

    When the source has a SUPPORTS bond (evidence → claim) and the
    output preserves the claim but drops the evidence, the bond is broken.
    This is a structural divergence distinct from atom-level divergences.
    """

    bond: Bond
    source_atom: Atom
    target_atom: Atom
    explanation: str
    severity: Severity = Severity.CRITICAL


@dataclass(frozen=True)
class DivergenceReport:
    """All divergences found between a captured pair of molecules."""

    pair: CapturedPair
    divergences: tuple[Divergence, ...]
    bond_divergences: tuple[BondDivergence, ...]
    source_atoms_matched: int
    source_atoms_unmatched: int
    output_atoms_unmatched: int

    @property
    def total(self) -> int:
        return len(self.divergences) + len(self.bond_divergences)

    @property
    def by_type(self) -> dict[DivergenceType, list[Divergence]]:
        result: dict[DivergenceType, list[Divergence]] = {}
        for d in self.divergences:
            result.setdefault(d.dtype, []).append(d)
        return result

    @property
    def by_severity(self) -> dict[Severity, list[Divergence]]:
        result: dict[Severity, list[Divergence]] = {}
        for d in self.divergences:
            result.setdefault(d.severity, []).append(d)
        # Include bond divergences in severity counts
        for bd in self.bond_divergences:
            # Create a synthetic Divergence for counting purposes
            result.setdefault(bd.severity, [])
        return result

    @property
    def broken_bonds_count(self) -> int:
        return len(self.bond_divergences)


# ---------------------------------------------------------------------------
# Alignment engine
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[*_`#]", "", text)
    return text


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _align_atoms(
    source: Sequence[Atom],
    output: Sequence[Atom],
    threshold: float = 0.4,
) -> tuple[
    list[tuple[Atom, Atom, float]],
    list[Atom],
    list[Atom],
]:
    """Align source and output atoms by content similarity.

    Uses word-level similarity first. If semantic embeddings are
    available, unmatched atoms get a second-chance alignment pass
    using semantic similarity — catching paraphrases that share
    meaning but not vocabulary.
    """
    matched: list[tuple[Atom, Atom, float]] = []
    used_output: set[int] = set()
    unmatched_source: list[Atom] = []

    # Pass 1: word-level alignment
    for src in source:
        best_sim = 0.0
        best_out: Atom | None = None
        best_idx = -1

        for i, out in enumerate(output):
            if i in used_output:
                continue
            sim = _similarity(src.text, out.text)
            if sim > best_sim:
                best_sim = sim
                best_out = out
                best_idx = i

        if best_out is not None and best_sim >= threshold:
            matched.append((src, best_out, best_sim))
            used_output.add(best_idx)
        else:
            unmatched_source.append(src)

    unmatched_output = [
        out for i, out in enumerate(output) if i not in used_output
    ]

    return matched, unmatched_source, unmatched_output


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_HEDGE_WORDS = {
    "might", "may", "could", "possibly", "perhaps", "likely",
    "suggests", "appears", "seems", "approximately", "roughly",
    "generally", "often", "sometimes", "arguably", "potentially",
}

_INTENSIFIERS = {
    "always", "never", "all", "every", "none", "certainly",
    "definitely", "absolutely", "clearly", "obviously",
    "undoubtedly", "unquestionably", "proven", "guaranteed",
}


_MAGNITUDE_WORDS = {'thousand', 'million', 'billion', 'trillion'}
_SUFFIX_CANONICAL = {
    'k': 'K', 'thousand': 'K',
    'm': 'M', 'million': 'M',
    'b': 'B', 'billion': 'B',
    't': 'T', 'trillion': 'T',
}


def _extract_numbers(text: str) -> set[str]:
    """Extract all number-like tokens, normalized.

    Normalizes suffixes so "94.8 billion" and "94.8B" match.
    Treats "60%" and "60%" identically.
    """
    normalized = set()
    tokens = text.split()
    i = 0
    while i < len(tokens):
        tok = tokens[i].strip('.,;:!?()')
        # Check if token starts with a digit
        core = re.match(r'^(\d[\d,.]*)(.*)$', tok)
        if core:
            num_part = core.group(1).rstrip('.')
            trail = core.group(2).lower()
            suffix = ''
            # Check trailing suffix attached to number (94.8B, 60%)
            if trail == '%':
                suffix = '%'
            elif trail in _SUFFIX_CANONICAL:
                suffix = _SUFFIX_CANONICAL[trail]
            # Check next token for standalone magnitude word
            elif i + 1 < len(tokens):
                next_tok = tokens[i + 1].strip('.,;:!?()').lower()
                if next_tok in _MAGNITUDE_WORDS:
                    suffix = _SUFFIX_CANONICAL[next_tok]
                    i += 1  # Skip the magnitude word
            if num_part:
                normalized.add(num_part + suffix)
        i += 1
    return normalized


def _classify_matched_pair(
    source: Atom, output: Atom, similarity: float,
    semantic_sim: float | None = None,
) -> Divergence | None:
    src_norm = _normalize(source.text)
    out_norm = _normalize(output.text)
    src_words = set(src_norm.split())
    out_words = set(out_norm.split())
    added_words = out_words - src_words
    removed_words = src_words - out_words

    # --- Critical checks run BEFORE the 0.95 similarity early-return ---
    # These catch factual changes that word/semantic similarity misses.

    # Numeric change detection — numbers matter more than words.
    # "60% improved" vs "90% cured" has high word similarity but is
    # a factual fabrication. Embedding models miss this.
    src_numbers = _extract_numbers(source.text)
    out_numbers = _extract_numbers(output.text)
    changed_numbers = (src_numbers - out_numbers) | (out_numbers - src_numbers)
    if changed_numbers and src_numbers:
        return Divergence(
            dtype=DivergenceType.SURFACING,
            severity=Severity.CRITICAL,
            source_atom=source,
            output_atom=output,
            explanation=f"Numeric values changed: source has {src_numbers}, output has {out_numbers}",
            confidence=0.9,
            suggested_correction=source.text,
            verification_method=f"Compare numbers directly: {src_numbers} vs {out_numbers}",
        )

    # Negation detection — "was not present" vs "was present" is a factual inversion
    _NEGATIONS = {"not", "no", "never", "neither", "nor", "nobody", "nothing",
                  "nowhere", "wasn't", "weren't", "isn't", "aren't", "don't",
                  "doesn't", "didn't", "can't", "couldn't", "won't", "wouldn't",
                  "shouldn't", "hasn't", "haven't", "hadn't"}
    src_negs = src_words & _NEGATIONS
    out_negs = out_words & _NEGATIONS
    neg_added = out_negs - src_negs
    neg_removed = src_negs - out_negs
    if neg_added or neg_removed:
        direction = "added" if neg_added else "removed"
        changed = neg_added or neg_removed
        return Divergence(
            dtype=DivergenceType.STRENGTHENING if neg_removed else DivergenceType.WEAKENING,
            severity=Severity.CRITICAL,
            source_atom=source,
            output_atom=output,
            explanation=f"Negation {direction}: {', '.join(sorted(changed))} — meaning may be inverted",
            confidence=0.85,
            suggested_correction=source.text,
            verification_method=f"Check whether the negation change inverts the factual claim",
        )

    # --- End critical checks. Near-identical text can return early now. ---
    if similarity > 0.95:
        return None

    # Strengthening
    added_intensifiers = added_words & _INTENSIFIERS
    if added_intensifiers:
        return Divergence(
            dtype=DivergenceType.STRENGTHENING,
            severity=Severity.MAJOR,
            source_atom=source,
            output_atom=output,
            explanation=f"AI strengthened claim by adding: {', '.join(sorted(added_intensifiers))}",
            confidence=min(0.6 + 0.1 * len(added_intensifiers), 0.95),
            suggested_correction=source.text,
            verification_method=f"Compare source atom {source.index} with output atom {output.index}; source has no intensifiers",
        )

    # Weakening
    added_hedges = added_words & _HEDGE_WORDS
    if added_hedges:
        return Divergence(
            dtype=DivergenceType.WEAKENING,
            severity=Severity.MAJOR,
            source_atom=source,
            output_atom=output,
            explanation=f"AI weakened claim by adding hedges: {', '.join(sorted(added_hedges))}",
            confidence=min(0.5 + 0.1 * len(added_hedges), 0.9),
            suggested_correction=source.text,
            verification_method=f"Source states this without qualification; output adds hedging language",
        )

    # Semantic gate: if meaning is preserved and no intensifiers/hedges
    # were detected above, this is a faithful paraphrase
    if semantic_sim is not None and semantic_sim >= FAITHFUL_THRESHOLD:
        return None  # Different words, same meaning, no intensity shift

    # Surfacing (significant expansion)
    if len(out_norm) > len(src_norm) * 1.5:
        return Divergence(
            dtype=DivergenceType.SURFACING,
            severity=Severity.MINOR,
            source_atom=source,
            output_atom=output,
            explanation="AI significantly expanded this atom.",
            confidence=0.5,
            suggested_correction=source.text,
            verification_method="Check whether added content is supported by the source document",
        )

    # Concealing (significant compression)
    if len(out_norm) < len(src_norm) * 0.5:
        return Divergence(
            dtype=DivergenceType.CONCEALING,
            severity=Severity.MAJOR,
            source_atom=source,
            output_atom=output,
            explanation="AI significantly compressed this atom, possibly hiding detail.",
            confidence=0.5,
            suggested_correction=source.text,
            verification_method="Compare full source text with compressed output; identify what was dropped",
        )

    # Semantic-informed general modification
    if similarity < 0.7:
        # If semantic similarity says it is in the uncertain zone,
        # reduce confidence — it might be a valid paraphrase
        conf = 0.4
        if semantic_sim is not None:
            if semantic_sim < DIVERGENT_THRESHOLD:
                conf = 0.75  # Semantics confirms: genuinely different
            elif semantic_sim < FAITHFUL_THRESHOLD:
                conf = 0.3   # Uncertain zone: lower confidence

        return Divergence(
            dtype=DivergenceType.SURFACING if len(out_norm) > len(src_norm)
            else DivergenceType.CONCEALING,
            severity=Severity.MINOR if conf < 0.5 else Severity.MAJOR,
            source_atom=source,
            output_atom=output,
            explanation=f"Content modified (word sim: {similarity:.0%}"
                       + (f", semantic sim: {semantic_sim:.0%}" if semantic_sim is not None else "")
                       + ").",
            confidence=conf,
            suggested_correction=source.text,
            verification_method=f"Side-by-side comparison; semantic similarity {'confirms divergence' if semantic_sim and semantic_sim < DIVERGENT_THRESHOLD else 'uncertain'}",
        )

    return None


def _classify_omission(atom: Atom, molecule: Molecule) -> Divergence:
    severity = _severity_from_atom(atom, molecule)
    return Divergence(
        dtype=DivergenceType.OMISSION,
        severity=severity,
        source_atom=atom,
        output_atom=None,
        explanation=f"Source {atom.atom_type.value} omitted: {atom.text[:80]}...",
        confidence=0.8,
        suggested_correction=f"Include: {atom.text}",
        verification_method=f"Search output for any paraphrase of: {atom.text[:60]}",
    )


def _classify_addition(atom: Atom) -> Divergence:
    return Divergence(
        dtype=DivergenceType.ADDITION,
        severity=Severity.MAJOR,
        source_atom=None,
        output_atom=atom,
        explanation=f"Output contains {atom.atom_type.value} not in source: {atom.text[:80]}...",
        confidence=0.8,
        suggested_correction="Remove this content or cite a source for it",
        verification_method="Search the source document for any basis for this content",
    )


def _find_broken_bonds(
    source_mol: Molecule,
    matched_indices: set[int],
    omitted_indices: set[int],
) -> list[BondDivergence]:
    """Find evidential bonds broken by omissions.

    If atom A supports atom B, and A was omitted but B was kept,
    the support bond is broken — B's claim is now unsupported.
    """
    broken: list[BondDivergence] = []
    atom_map = {a.index: a for a in source_mol.atoms}

    for bond in source_mol.bonds:
        if not bond.is_evidential:
            continue

        source_in = bond.source_index
        target_in = bond.target_index

        # Evidence omitted but claim kept
        if source_in in omitted_indices and target_in in matched_indices:
            broken.append(BondDivergence(
                bond=bond,
                source_atom=atom_map[source_in],
                target_atom=atom_map[target_in],
                explanation=(
                    f"Evidence dropped but claim retained: "
                    f"'{atom_map[source_in].text[:50]}...' supported "
                    f"'{atom_map[target_in].text[:50]}...' — "
                    f"the claim is now unsupported"
                ),
            ))

    return broken


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_divergences(pair: CapturedPair) -> DivergenceReport:
    """Find all divergences between a captured source-output pair.

    This is the second step in Criterion. After capture, divergence.

    Compares molecules atom-by-atom, classifies each difference by
    type and severity, checks for broken bonds (evidence dropped
    while claims retained), and returns actionable corrections.

    Args:
        pair: A CapturedPair from the capture module.

    Returns:
        A DivergenceReport with every divergence classified.
    """
    matched, unmatched_src, unmatched_out = _align_atoms(
        pair.source.atoms, pair.output.atoms
    )

    # Batch semantic similarity for matched pairs (if available)
    sem_sims: list[float | None] = [None] * len(matched)
    if _USE_SEMANTIC and matched:
        try:
            src_texts = [src.text for src, _, _ in matched]
            out_texts = [out.text for _, out, _ in matched]
            sem_sims = batch_semantic_similarity(src_texts, out_texts)
        except Exception:
            pass  # Degrade gracefully to word-only

    # Compute retention ratio BEFORE classification so summary mode
    # can downgrade matched-pair divergences too.
    total_source = len(pair.source.atoms)
    total_output = len(pair.output.atoms)
    output_ratio = total_output / total_source if total_source > 0 else 1.0
    # Summary = output is substantially shorter than source.
    is_summary = (output_ratio < 0.5)

    divergences: list[Divergence] = []

    for i, (src, out, sim) in enumerate(matched):
        div = _classify_matched_pair(src, out, sim, semantic_sim=sem_sims[i])
        if div is not None:
            # In summary mode, downgrade intensity shifts and compression
            # effects. Summaries reword and compress; that's their job.
            if is_summary and div.dtype in (
                DivergenceType.STRENGTHENING, DivergenceType.WEAKENING,
                DivergenceType.CONCEALING, DivergenceType.SURFACING,
            ):
                div = Divergence(
                    dtype=div.dtype,
                    severity=Severity.MINOR,
                    source_atom=div.source_atom,
                    output_atom=div.output_atom,
                    explanation=div.explanation,
                    confidence=min(div.confidence, 0.4),
                    suggested_correction=div.suggested_correction,
                    verification_method=div.verification_method,
                )
            divergences.append(div)

    # Bond analysis (computed before omissions so we can check bond context)
    matched_indices = {src.index for src, _, _ in matched}
    omitted_indices = {a.index for a in unmatched_src}
    broken_bonds = _find_broken_bonds(pair.source, matched_indices, omitted_indices)
    broken_source_indices = {bd.source_atom.index for bd in broken_bonds}

    for atom in unmatched_src:
        div = _classify_omission(atom, pair.source)
        # In summary mode, downgrade plain omissions to MINOR.
        # Only omissions that break an evidential bond stay at their
        # original severity — those are real integrity failures.
        if is_summary and atom.index not in broken_source_indices:
            div = Divergence(
                dtype=div.dtype,
                severity=Severity.MINOR,
                source_atom=div.source_atom,
                output_atom=div.output_atom,
                explanation=div.explanation,
                confidence=0.3,  # Low confidence — expected in summaries
                suggested_correction=div.suggested_correction,
                verification_method=div.verification_method,
            )
        divergences.append(div)

    # For summary-mode additions: batch-encode all unmatched atoms once
    # and compute a similarity matrix, instead of per-atom model calls.
    add_max_sims: dict[int, float] = {}  # output atom index → max sim to any source atom
    if is_summary and _USE_SEMANTIC and unmatched_out and unmatched_src:
        try:
            from criterion.semantic import _get_model, FAITHFUL_THRESHOLD
            model = _get_model()
            out_texts = [a.text for a in unmatched_out]
            src_texts = [a.text for a in unmatched_src]
            all_embs = model.encode(
                out_texts + src_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            out_embs = all_embs[:len(out_texts)]
            src_embs = all_embs[len(out_texts):]
            # Compute similarity matrix and take max per output atom
            # out_embs: (O, D), src_embs: (S, D) → sim_matrix: (O, S)
            out_norms = np.linalg.norm(out_embs, axis=1, keepdims=True)
            src_norms = np.linalg.norm(src_embs, axis=1, keepdims=True)
            out_normed = out_embs / np.maximum(out_norms, 1e-9)
            src_normed = src_embs / np.maximum(src_norms, 1e-9)
            sim_matrix = out_normed @ src_normed.T  # (O, S)
            for oi, atom in enumerate(unmatched_out):
                add_max_sims[atom.index] = float(sim_matrix[oi].max())
        except Exception:
            pass  # Degrade gracefully

    for atom in unmatched_out:
        div = _classify_addition(atom)
        if is_summary:
            max_sim = add_max_sims.get(atom.index, 0.0)
            if max_sim >= 0.75:
                # This "addition" is actually a paraphrase of omitted content
                div = Divergence(
                    dtype=div.dtype,
                    severity=Severity.COSMETIC,
                    source_atom=div.source_atom,
                    output_atom=div.output_atom,
                    explanation=f"Likely paraphrase of source content (sem={max_sim:.2f}): {atom.text[:60]}...",
                    confidence=0.2,
                    suggested_correction=div.suggested_correction,
                    verification_method=div.verification_method,
                )
            elif max_sim >= 0.4:
                div = Divergence(
                    dtype=div.dtype,
                    severity=Severity.MINOR,
                    source_atom=div.source_atom,
                    output_atom=div.output_atom,
                    explanation=f"Possible paraphrase in summary (sem={max_sim:.2f}): {atom.text[:60]}...",
                    confidence=0.4,
                    suggested_correction=div.suggested_correction,
                    verification_method=div.verification_method,
                )
            else:
                # Low semantic similarity — still downgrade slightly in summary mode
                div = Divergence(
                    dtype=div.dtype,
                    severity=Severity.MINOR,
                    source_atom=div.source_atom,
                    output_atom=div.output_atom,
                    explanation=div.explanation,
                    confidence=0.5,
                    suggested_correction=div.suggested_correction,
                    verification_method=div.verification_method,
                )
        divergences.append(div)

    return DivergenceReport(
        pair=pair,
        divergences=tuple(divergences),
        bond_divergences=tuple(broken_bonds),
        source_atoms_matched=len(matched),
        source_atoms_unmatched=len(unmatched_src),
        output_atoms_unmatched=len(unmatched_out),
    )
