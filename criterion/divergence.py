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

import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Sequence

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
        encode_texts as _encode_texts,
        FAITHFUL_THRESHOLD,
        DIVERGENT_THRESHOLD,
    )
except ImportError:
    _USE_SEMANTIC = False
    FAITHFUL_THRESHOLD = 0.75
    DIVERGENT_THRESHOLD = 0.35

# NLI contradiction detection (optional — degrades gracefully)
_USE_NLI = True
try:
    from criterion.nli import classify_batch_probs as _nli_probs
except ImportError:
    _USE_NLI = False

# Grounding scorer — decides whether each output sentence is supported
_USE_SCORER = True
try:
    from criterion import grounding as _grounding_mod
except ImportError:
    _USE_SCORER = False


def _scorer_support(source_text: str, claims: list[str]) -> list[tuple[float, str]]:
    """Indirection so tests can stub the scorer."""
    return _grounding_mod.support(source_text, claims)


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
    # Stable machine-readable finding code (e.g. "numeric_change",
    # "ungrounded"). Policy decisions key off this, never off the
    # human-readable explanation text.
    code: str = ""
    # Character similarity (0-1) between the aligned source and output
    # sentences, when the finding comes from an aligned pair.
    alignment: float | None = None

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
    # Which detectors actually ran for THIS report. A missing or failed
    # model is recorded here instead of being silently swallowed.
    capabilities: dict[str, Any] = field(default_factory=dict)
    # Output atom index -> best source support (NLI entailment, 0-1) for
    # every output sentence NLI judged. Absent for fragments judged by
    # containment and whenever NLI did not run.
    support: dict[int, float] = field(default_factory=dict)

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
# Text hygiene
# ---------------------------------------------------------------------------

_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "ʼ": "'", "‛": "'",
    "“": '"', "”": '"', "−": "-",
})


def _clean(text: str) -> str:
    """Canonical form used by every check and every model call.

    NFKC folds compatibility forms (fullwidth digits, ligatures); format
    characters (zero-width spaces/joiners, bidi controls, soft hyphens)
    are removed so they cannot split a token and hide it from a detector;
    typographic quotes are straightened so "isn’t" and "isn't" are the
    same word.
    """
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _normalize(text: str) -> str:
    text = _clean(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[*_`#]", "", text)
    return text


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


_CONTRACTIONS = (
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"(\w)n't\b"), r"\1 not"),
)
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


def _words(text: str) -> list[str]:
    """Lowercased word tokens with contractions expanded, punctuation dropped."""
    norm = _normalize(text)
    for pattern, repl in _CONTRACTIONS:
        norm = pattern.sub(repl, norm)
    return _WORD.findall(norm)


_SCRIPTS = ("LATIN", "CYRILLIC", "GREEK")


def _hidden_content(output_text: str, source_raw: str) -> str | None:
    """Describe invisible or look-alike characters the output introduced.

    Returns None when the output is clean. Characters that also occur in
    the source are not reported, and neither are joiners between emoji or
    non-Latin letters (ZWNJ/ZWJ are legitimate in Persian, Indic scripts
    and emoji). Bidi overrides and tag characters are always reported.
    """
    invisible_set = set()
    for pos, ch in enumerate(output_text):
        if unicodedata.category(ch) != "Cf" or ch in source_raw:
            continue
        code = ord(ch)
        always = 0x202A <= code <= 0x202E or 0x2066 <= code <= 0x2069 or 0xE0000 <= code <= 0xE007F
        prev_ch = output_text[pos - 1] if pos else ""
        next_ch = output_text[pos + 1] if pos + 1 < len(output_text) else ""
        # ZWJ/ZWNJ between emoji or non-Latin letters is ordinary typography;
        # next to ASCII letters or digits it has no legitimate use.
        beside_ascii = any(c.isascii() and c.isalnum() for c in (prev_ch, next_ch))
        if always or beside_ascii:
            invisible_set.add(f"U+{code:04X}")
    invisible = sorted(invisible_set)
    mixed: list[str] = []
    for token in re.findall(r"[^\W\d_]+", output_text):
        scripts = set()
        for ch in token:
            name = unicodedata.name(ch, "")
            for script in _SCRIPTS:
                if name.startswith(script):
                    scripts.add(script)
        if len(scripts) > 1 and token not in source_raw:
            mixed.append(token)
    parts = []
    if invisible:
        parts.append(f"invisible characters {', '.join(invisible)}")
    if mixed:
        parts.append(f"mixed-script look-alike words {mixed[:3]}")
    return "; ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Alignment engine
# ---------------------------------------------------------------------------

class InputTooLargeError(ValueError):
    """Fuzzy alignment would exceed the comparison budget.

    Alignment is quadratic in the number of sentences that are not
    verbatim copies. Rather than hang (a 2,000 x 2,000 rewrite takes
    minutes), Criterion refuses and says so.
    """


MAX_FUZZY_COMPARISONS = int(os.environ.get("CRITERION_MAX_COMPARISONS", "1000000"))

def _align_atoms(
    source: Sequence[Atom],
    output: Sequence[Atom],
    threshold: float = 0.5,
) -> tuple[
    list[tuple[Atom, Atom, float]],
    list[Atom],
    list[Atom],
]:
    """Align source and output atoms by character-level similarity.

    Pass 1 pairs atoms whose normalised text is identical (linear time).
    Pass 2 scores the remainder and assigns best-first, so a source atom
    early in the document cannot steal the output atom that a later
    source atom matches better. Cheap upper bounds (real_quick_ratio,
    quick_ratio) prune most pairs before the quadratic-cost ratio().

    Atoms that stay unmatched are not assumed to be omissions/additions:
    find_divergences() grounds them with NLI when it is available.
    """
    src_norm = [_normalize(a.text) for a in source]
    out_norm = [_normalize(a.text) for a in output]

    pairs: dict[int, tuple[int, float]] = {}   # source pos -> (output pos, sim)
    used_out: set[int] = set()

    # Pass 1: exact matches
    by_text: dict[str, list[int]] = {}
    for j, text in enumerate(out_norm):
        by_text.setdefault(text, []).append(j)
    for i, text in enumerate(src_norm):
        bucket = by_text.get(text)
        if bucket:
            j = bucket.pop(0)
            pairs[i] = (j, 1.0)
            used_out.add(j)

    # Pass 2: fuzzy matches, best-first
    candidates: list[tuple[float, int, int]] = []
    matcher = SequenceMatcher(None)
    rest_src = [i for i in range(len(source)) if i not in pairs]
    rest_out = len(output) - len(used_out)
    if len(rest_src) * rest_out > MAX_FUZZY_COMPARISONS:
        raise InputTooLargeError(
            f"{len(rest_src)} source x {rest_out} output sentences still need fuzzy "
            f"alignment ({len(rest_src) * rest_out:,} comparisons; limit "
            f"{MAX_FUZZY_COMPARISONS:,}). Verify the document in sections, or raise "
            f"CRITERION_MAX_COMPARISONS."
        )
    for j in range(len(output)):
        if j in used_out:
            continue
        matcher.set_seq2(out_norm[j])
        for i in rest_src:
            matcher.set_seq1(src_norm[i])
            if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                continue
            sim = matcher.ratio()
            if sim >= threshold:
                candidates.append((-sim, i, j))
    candidates.sort()
    for neg_sim, i, j in candidates:
        if i in pairs or j in used_out:
            continue
        pairs[i] = (j, -neg_sim)
        used_out.add(j)

    matched = [
        (source[i], output[pairs[i][0]], pairs[i][1])
        for i in range(len(source)) if i in pairs
    ]
    unmatched_source = [source[i] for i in range(len(source)) if i not in pairs]
    unmatched_output = [out for j, out in enumerate(output) if j not in used_out]
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

_NEGATIONS = {
    "not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",
}

_NEG_PREFIXES = ("un", "in", "im", "il", "ir", "dis", "non")
_ANTONYM_PAIRS = {
    ("innocent", "guilty"), ("true", "false"), ("alive", "dead"),
    ("present", "absent"), ("accept", "reject"), ("allow", "deny"),
    ("approve", "reject"), ("confirm", "deny"), ("increase", "decrease"),
    ("rise", "fall"), ("success", "failure"), ("win", "lose"),
    ("gain", "loss"), ("profit", "loss"), ("positive", "negative"),
    ("include", "exclude"), ("open", "closed"), ("open", "close"),
    ("start", "stop"), ("begin", "end"), ("buy", "sell"),
    ("safe", "dangerous"), ("good", "bad"),
}
_ANTONYM_SET = _ANTONYM_PAIRS | {(b, a) for a, b in _ANTONYM_PAIRS}

_MAGNITUDE = {
    "k": 10**3, "thousand": 10**3,
    "m": 10**6, "mm": 10**6, "million": 10**6,
    "b": 10**9, "bn": 10**9, "billion": 10**9,
    "t": 10**12, "trillion": 10**12,
}
_NUMBER = re.compile(
    r"(?<![\w.,])"                                   # not inside a word or number
    r"(?P<sign>-)?"
    r"(?:[$€£¥¢₹₽]\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:"
    r"(?P<pct>\s?%|\s?per\s?cent\b)"
    r"|(?P<suf>k|mm|m|bn|b|t)\b"
    r"|\s(?P<mag>thousand|million|billion|trillion)\b"
    r")?",
    re.IGNORECASE,
)
_SIGN_CONTEXT = re.compile(r"(?:^|[\s(\[=:$€£¥¢₹₽])$")


def _extract_numbers(text: str) -> set[str]:
    """Extract every number in the text as a canonical value string.

    Canonical means the VALUE, not the spelling, so equivalent forms
    compare equal:  1,500,000 == 1.5 million == 1.5M;  10% == 10 percent;
    $1,000 == 1000;  2.50 == 2.5. Currency symbols are dropped (a
    currency swap is left to NLI). A leading minus is kept when it is a
    sign rather than a range dash, so "-5%" and "5%" differ while
    "5-10" yields 5 and 10.
    """
    cleaned = _clean(text)
    found: set[str] = set()
    for m in _NUMBER.finditer(cleaned):
        try:
            value = Decimal(m.group("num").replace(",", ""))
        except InvalidOperation:
            continue
        scale = m.group("suf") or m.group("mag")
        if scale:
            value *= _MAGNITUDE[scale.lower()]
        if m.group("sign") and _SIGN_CONTEXT.search(cleaned[:m.start()]):
            value = -value
        canonical = format(value.normalize(), "f")
        if canonical in ("-0", ""):
            canonical = "0"
        found.add(canonical + ("%" if m.group("pct") else ""))
    return found


def _fmt_numbers(numbers: set[str]) -> str:
    return "{" + ", ".join(sorted(numbers)) + "}" if numbers else "none"


def _classify_matched_pair(
    source: Atom, output: Atom, similarity: float,
    semantic_sim: float | None = None,
    nli_label: str | None = None,
    nli_equivalent: bool = False,
) -> Divergence | None:
    """Deterministic checks on one aligned pair.

    nli_label is the forward (source ⇒ output) NLI label, if known.
    nli_equivalent means NLI found entailment in BOTH directions with
    high confidence; lexical negation/antonym heuristics then stand down
    ("is not untrue" vs "is true"). Numeric checks never stand down.
    """
    src_tokens = _words(source.text)
    out_tokens = _words(output.text)
    src_norm = " ".join(src_tokens)
    out_norm = " ".join(out_tokens)
    src_words = set(src_tokens)
    out_words = set(out_tokens)
    added_words = out_words - src_words
    removed_words = src_words - out_words

    # --- Critical checks run BEFORE the 0.95 similarity early-return ---
    # These catch factual changes that word/semantic similarity misses.

    # Numeric change detection — numbers matter more than words.
    src_numbers = _extract_numbers(source.text)
    out_numbers = _extract_numbers(output.text)
    numbers_differ = src_numbers != out_numbers
    numbers_added = out_numbers - src_numbers
    # If NLI says entailment, only flag when numbers were REPLACED
    # (both sides have numbers but they differ), not just removed
    # (a faithful paraphrase like "365 days" → "about a year")
    if numbers_differ and nli_label == "entailment":
        if not (src_numbers and out_numbers and numbers_added):
            numbers_differ = False
    if numbers_differ:
        dropped_only = bool(src_numbers) and not numbers_added
        return Divergence(
            dtype=DivergenceType.SURFACING,
            severity=Severity.CRITICAL,
            source_atom=source,
            output_atom=output,
            explanation=(
                f"Numeric values changed: source has {_fmt_numbers(src_numbers)}, "
                f"output has {_fmt_numbers(out_numbers)}"
            ),
            confidence=0.9,
            suggested_correction=source.text,
            verification_method=(
                f"Compare numbers directly: {_fmt_numbers(src_numbers)} vs {_fmt_numbers(out_numbers)}"
            ),
            code="numeric_dropped" if dropped_only else "numeric_change",
        )

    if not nli_equivalent:
        # Negation detection — "was not present" vs "was present".
        # Counted after contraction expansion, so "isn't" == "is not".
        src_negs = Counter(w for w in src_tokens if w in _NEGATIONS)
        out_negs = Counter(w for w in out_tokens if w in _NEGATIONS)
        n_src, n_out = sum(src_negs.values()), sum(out_negs.values())
        if n_src != n_out:
            removed = n_out < n_src
            changed = (src_negs - out_negs) if removed else (out_negs - src_negs)
            return Divergence(
                dtype=DivergenceType.STRENGTHENING if removed else DivergenceType.WEAKENING,
                severity=Severity.CRITICAL,
                source_atom=source,
                output_atom=output,
                explanation=(
                    f"Negation {'removed' if removed else 'added'}: "
                    f"{', '.join(sorted(changed)) or 'not'} — meaning may be inverted"
                ),
                confidence=0.85,
                suggested_correction=source.text,
                verification_method="Check whether the negation change inverts the factual claim",
                code="negation",
            )

        # Role swap — the same words in a different order, where what moved
        # is a number, a negation, or one half of an opposite pair:
        #   "5 mg dose, 10 mg limit"      → "10 mg dose, 5 mg limit"
        #   "safe for adults, unsafe for children" → the reverse
        # Bag-of-words checks see no change at all here.
        if Counter(src_tokens) == Counter(out_tokens) and src_tokens != out_tokens:
            moved = {a for a, b in zip(src_tokens, out_tokens) if a != b}
            pivots = sorted(
                w for w in moved
                if w[0].isdigit() or w in _NEGATIONS
                or any(
                    (w, o) in _ANTONYM_SET
                    or any(w == pfx + o or o == pfx + w for pfx in _NEG_PREFIXES)
                    for o in out_words
                )
            )
            if pivots:
                return Divergence(
                    dtype=DivergenceType.SURFACING,
                    severity=Severity.CRITICAL,
                    source_atom=source,
                    output_atom=output,
                    explanation=(
                        f"Role swap: same words, different order ({', '.join(pivots[:4])} moved) "
                        f"— who/what each value applies to may have changed"
                    ),
                    confidence=0.85,
                    suggested_correction=source.text,
                    verification_method="Check which subject each number, negation or predicate is attached to",
                    code="role_swap",
                )

        # Prefix-negation / antonym detection, on words that changed
        for src_w in sorted(removed_words):
            for out_w in sorted(added_words):
                is_reversal = any(
                    out_w == pfx + src_w or src_w == pfx + out_w
                    for pfx in _NEG_PREFIXES
                ) or (src_w, out_w) in _ANTONYM_SET
                if is_reversal:
                    return Divergence(
                        dtype=DivergenceType.SURFACING,
                        severity=Severity.CRITICAL,
                        source_atom=source,
                        output_atom=output,
                        explanation=f"Predicate reversal: '{src_w}' vs '{out_w}' — meaning inverted",
                        confidence=0.9,
                        suggested_correction=source.text,
                        verification_method=f"Compare: source says '{src_w}', output says '{out_w}'",
                        code="antonym",
                    )

    # --- End critical checks. Near-identical text can return early now. ---
    if similarity > 0.95:
        return None

    # Strengthening — added intensifiers OR removed hedges
    removed_hedges = removed_words & _HEDGE_WORDS
    added_intensifiers = added_words & _INTENSIFIERS
    if added_intensifiers or removed_hedges:
        return Divergence(
            dtype=DivergenceType.STRENGTHENING,
            severity=Severity.MAJOR,
            source_atom=source,
            output_atom=output,
            explanation=(
                (f"AI strengthened claim by adding: {', '.join(sorted(added_intensifiers))}" if added_intensifiers else "")
                + ("; " if added_intensifiers and removed_hedges else "")
                + (f"removed hedges: {', '.join(sorted(removed_hedges))}" if removed_hedges else "")
            ),
            confidence=min(0.6 + 0.1 * len(added_intensifiers | removed_hedges), 0.95),
            suggested_correction=source.text,
            verification_method=f"Compare source atom {source.index} with output atom {output.index}; hedges/qualifiers changed",
            code="strengthening",
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
            verification_method="Source states this without qualification; output adds hedging language",
            code="weakening",
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
            code="expanded",
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
            code="compressed",
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

        confirmed = semantic_sim is not None and semantic_sim < DIVERGENT_THRESHOLD
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
            verification_method=f"Side-by-side comparison; semantic similarity {'confirms divergence' if confirmed else 'uncertain'}",
            code="modified",
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
        code="omission",
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
        code="unaligned_addition",
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
# Grounding — NLI over atoms that alignment could not settle
# ---------------------------------------------------------------------------

# An atom counts as supported when some premise entails it at >= this.
GROUNDED_THRESHOLD = 0.5
# Contradiction must clear this to be reported (same bar as matched pairs).
CONTRADICTION_THRESHOLD = 0.7
# Below this support, an unsupported sentence is CRITICAL (blocks trust on its
# own) rather than MAJOR. Equal to GROUNDED_THRESHOLD by default, so every
# unsupported sentence blocks — the published single-threshold protocol.
# Lower it to let weakly supported sentences through as MAJOR findings.
FABRICATION_THRESHOLD = 0.5
# Both directions must entail at >= this for lexical heuristics to stand down.
EQUIVALENCE_THRESHOLD = 0.9
# A single other atom must entail at >= this to re-align a matched pair.
REALIGN_THRESHOLD = 0.8
# A deterministic check (changed number, flipped negation, role swap) or a
# pairwise contradiction can overrule the scorer's "supported" only when the
# output sentence is this close to a verbatim copy of its source sentence.
# Below it, the check is evidence of a loose pairing, not of an error.
VERBATIM_THRESHOLD = 0.9



@dataclass(frozen=True)
class Thresholds:
    """The decision thresholds, as one object a caller can override.

    grounded:       a sentence is supported when some source passage entails
                    it at >= this. Raise it to demand stronger support (more
                    flags); lower it to accept weaker support (fewer flags).
    contradiction:  NLI contradiction must reach this to be reported.
    fabrication:    below this support, an unsupported sentence is CRITICAL
                    (blocks trust on its own) rather than MAJOR. Defaults to
                    `grounded`, so every unsupported sentence blocks.
    verbatim:       alignment similarity above which a deterministic check
                    may overrule the scorer's "supported".
    """

    grounded: float = field(default_factory=lambda: GROUNDED_THRESHOLD)
    contradiction: float = field(default_factory=lambda: CONTRADICTION_THRESHOLD)
    fabrication: float = field(default_factory=lambda: FABRICATION_THRESHOLD)
    equivalence: float = field(default_factory=lambda: EQUIVALENCE_THRESHOLD)
    realign: float = field(default_factory=lambda: REALIGN_THRESHOLD)
    verbatim: float = field(default_factory=lambda: VERBATIM_THRESHOLD)

    def __post_init__(self) -> None:
        for name in ("grounded", "contradiction", "fabrication", "equivalence", "realign", "verbatim"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"threshold {name} must be between 0 and 1, got {value!r}")


_TOP_K_SINGLE = 3       # premises tried one at a time
_TOP_K_JOINED = 5       # premises concatenated for multi-sentence support
_PREMISE_WORD_BUDGET = 220  # keeps premise + hypothesis inside the 512-token window

_STOPWORDS = frozenset(
    "a an and are as at be been but by for from had has have he her his i if in "
    "is it its of on or she that the their them they this to was were which who "
    "will with would you your".split()
)


@dataclass(frozen=True)
class _Grounding:
    """Best NLI evidence for one hypothesis atom against a pool of premises."""

    entailment: float                 # best entailment over all premises
    contradiction: float              # best contradiction over single premises
    entail_atom: Atom | None          # single premise atom that best entails
    entail_atom_score: float
    contra_atom: Atom | None          # single premise atom that best contradicts


def _rank_pool(hyp_texts: list[str], pool_texts: list[str]) -> list[list[int]]:
    """For each hypothesis, pool indices ordered most-relevant first."""
    if _USE_SEMANTIC:
        try:
            embs = _encode_texts(hyp_texts + pool_texts)
            sims = embs[:len(hyp_texts)] @ embs[len(hyp_texts):].T
            return [list(np.argsort(-row)) for row in sims]
        except Exception:
            pass  # fall through to lexical ranking
    pool_sets = [set(_words(t)) - _STOPWORDS for t in pool_texts]
    ranked = []
    for text in hyp_texts:
        words = set(_words(text)) - _STOPWORDS
        scores = [
            len(words & ps) / (len(words | ps) or 1) for ps in pool_sets
        ]
        ranked.append(sorted(range(len(pool_texts)), key=lambda i: -scores[i]))
    return ranked


def _as_statement(question: str, answer: str) -> str:
    """Turn a question and a fragment answer into something NLI can judge."""
    return f'The answer to the question "{question.strip()}" is {answer.strip().rstrip(".")}.'


def _ground(
    hypotheses: Sequence[Atom],
    pool: Sequence[Atom],
    texts: dict[int, str] | None = None,
) -> dict[int, _Grounding]:
    """Ask NLI whether each hypothesis atom is supported by the pool.

    Each hypothesis is tested against its most relevant pool atoms one at
    a time (to localise support or contradiction) and against several of
    them concatenated in document order (so a sentence that fuses two
    source sentences still counts as supported). Raises if NLI fails.
    """
    if not hypotheses or not pool:
        return {}
    # `texts` overrides the hypothesis wording per atom index (used to phrase
    # a fragment answer as a statement when the question is known).
    texts = texts or {}
    hyp_texts = [_clean(texts.get(a.index, a.text)) for a in hypotheses]
    pool_texts = [_clean(a.text) for a in pool]
    ranked = _rank_pool(hyp_texts, pool_texts)

    premises: list[str] = []
    hyps: list[str] = []
    owner: list[tuple[int, int | None]] = []   # (hypothesis pos, pool pos or None=joined)
    for h, order in enumerate(ranked):
        for p in order[:_TOP_K_SINGLE]:
            premises.append(pool_texts[p]); hyps.append(hyp_texts[h]); owner.append((h, int(p)))
        chosen: list[int] = []
        budget = _PREMISE_WORD_BUDGET
        for p in order[:_TOP_K_JOINED]:
            cost = len(pool_texts[p].split())
            if chosen and cost > budget:
                break
            chosen.append(int(p)); budget -= cost
        if len(chosen) > 1:
            joined = " ".join(pool_texts[p] for p in sorted(chosen))
            premises.append(joined); hyps.append(hyp_texts[h]); owner.append((h, None))

    probs = _nli_probs(premises, hyps)

    best: dict[int, dict[str, Any]] = {
        h: {"ent": 0.0, "con": 0.0, "ent_atom": None, "ent_atom_score": 0.0, "con_atom": None}
        for h in range(len(hypotheses))
    }
    for (h, p), pr in zip(owner, probs):
        b = best[h]
        b["ent"] = max(b["ent"], pr.entailment)
        if p is not None:
            if pr.entailment > b["ent_atom_score"]:
                b["ent_atom_score"], b["ent_atom"] = pr.entailment, pool[p]
            if pr.contradiction > b["con"]:
                b["con"], b["con_atom"] = pr.contradiction, pool[p]
    return {
        hypotheses[h].index: _Grounding(
            entailment=b["ent"], contradiction=b["con"],
            entail_atom=b["ent_atom"], entail_atom_score=b["ent_atom_score"],
            contra_atom=b["con_atom"],
        )
        for h, b in best.items()
    }


_VERB_CUES = frozenset(
    "is are was were be been being am has have had do does did will would can could "
    "shall should may might must".split()
)


def _is_fragment(text: str) -> bool:
    """True for output that is not a proposition ("Delhi", "Todd Phillips", "no").

    NLI judges statements. Given a bare noun phrase as hypothesis it returns
    confident nonsense (contradiction ~1.0 for a name that IS in the source),
    so fragments are checked by containment instead and never sent to NLI.
    """
    tokens = _words(text)
    if len(tokens) < 4:
        return True
    if len(tokens) < 8 and not text.rstrip().endswith((".", "!", "?")):
        return not any(t in _VERB_CUES for t in tokens)
    return False


def _classify_fragment(atom: Atom, source_vocab: set[str]) -> Divergence | None:
    """A fragment answer can be located in the source, not verified against it."""
    content = [t for t in _words(atom.text) if t not in _STOPWORDS and t not in ("yes", "no")]
    if not content:
        return Divergence(
            dtype=DivergenceType.ADDITION,
            severity=Severity.MINOR,
            source_atom=None,
            output_atom=atom,
            explanation=f"Unverifiable fragment (no claim to check without the question): {atom.text[:60]}",
            confidence=0.3,
            suggested_correction="State the answer as a full sentence so it can be verified",
            verification_method="Read the answer together with the question it responds to",
            code="unverifiable_fragment",
        )
    missing = [t for t in content if t not in source_vocab]
    if not missing:
        return None  # every content word occurs in the source; nothing to disprove
    return Divergence(
        dtype=DivergenceType.ADDITION,
        severity=Severity.MAJOR,
        source_atom=None,
        output_atom=atom,
        explanation=f"Fragment not found in source (missing: {', '.join(missing[:5])}): {atom.text[:60]}",
        confidence=0.9,
        suggested_correction="Remove this content or cite a source for it",
        verification_method="Search the source document for these terms",
        code="fragment_not_in_source",
    )


def _contradiction(source: Atom | None, output: Atom, confidence: float, code: str) -> Divergence:
    return Divergence(
        dtype=DivergenceType.SURFACING,
        severity=Severity.CRITICAL,
        source_atom=source,
        output_atom=output,
        explanation=f"NLI contradiction detected ({confidence:.0%} confidence): meaning reversed or materially changed",
        confidence=confidence,
        suggested_correction=source.text if source else "Remove this content or cite a source for it",
        verification_method=f"NLI model classifies this pair as contradiction with {confidence:.0%} confidence",
        code=code,
    )


def _unsupported(
    source: Atom | None, output: Atom, entailment: float, fabrication: float,
) -> Divergence:
    """An output claim that NLI finds no support for anywhere relevant in the source."""
    fabricated = entailment < fabrication
    where = "within this sentence" if source is not None else "in this sentence"
    entailment = float(entailment)
    return Divergence(
        dtype=DivergenceType.ADDITION,
        severity=Severity.CRITICAL if fabricated else Severity.MAJOR,
        source_atom=source,
        output_atom=output,
        explanation=(
            f"Unsupported content {where} (best source support {entailment:.0%}): "
            f"{output.text[:80]}..."
        ),
        confidence=round(max(1.0 - entailment, 0.5), 4),
        suggested_correction=source.text if source else "Remove this content or cite a source for it",
        verification_method="NLI finds no source passage that entails this statement; search the source for a basis",
        code="unsupported_detail" if source is not None else "ungrounded",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Findings that claim "the source does not support this sentence". When the
# grounding scorer ran, it is the authority on that question.
_SUPPORT_CODES = frozenset({"ungrounded", "unsupported_detail", "contradicts_source"})
# Findings from comparing one output sentence with ONE aligned source
# sentence. Precise when the pair is near-verbatim; unreliable when the
# pairing is loose (a summary sentence is not wrong for differing from
# whichever source sentence it happens to resemble).
_PAIRWISE_CODES = frozenset({
    "numeric_change", "numeric_dropped", "negation", "antonym", "role_swap", "nli_contradiction",
})


def _arbitrate(
    divergences: list[Divergence],
    scores: dict[int, float],
    partners: dict[int, tuple[Atom, float]],
    outputs: dict[int, Atom],
    t: Thresholds,
) -> list[Divergence]:
    """Reconcile the pipeline's findings with the grounding scorer.

    For each output sentence the scorer judged:
      supported (score >= grounded)
          - "unsupported"/"contradicts" findings from the NLI pipeline are
            dropped: the scorer overrules them.
          - pairwise findings stand only if the pair is near-verbatim
            (alignment >= verbatim); otherwise they are kept as MINOR notes.
      unsupported (score < grounded)
          - existing findings stay and explain the failure; if there is
            none, an "ungrounded" finding is added.
          - severity is CRITICAL below `fabrication`, MAJOR otherwise.
    Measured on LLM-AggreFact: letting the pairwise checks block on their own
    lowered accuracy with every scorer tried, hence the verbatim gate.
    """
    result: list[Divergence] = []
    explained: set[int] = set()
    for d in divergences:
        out = d.output_atom
        arbitrated = d.code in _SUPPORT_CODES or d.code in _PAIRWISE_CODES
        if out is None or out.index not in scores or not arbitrated:
            result.append(d)
            continue
        score = scores[out.index]
        alignment = partners.get(out.index, (None, None))[1]
        if score >= t.grounded:
            if d.code in _SUPPORT_CODES:
                continue
            if alignment is not None and alignment >= t.verbatim:
                result.append(replace(d, alignment=alignment))
                explained.add(out.index)
            else:
                result.append(replace(
                    d, severity=Severity.MINOR, confidence=min(d.confidence, 0.3),
                    alignment=alignment,
                    explanation=d.explanation + f" — but the sentence is supported by the source overall (support {score:.0%})",
                ))
            continue
        severity = Severity.CRITICAL if score < t.fabrication else Severity.MAJOR
        if d.code in _SUPPORT_CODES and d.severity != Severity.CRITICAL:
            d = replace(d, severity=severity)
        if d.code in _SUPPORT_CODES:
            d = replace(d, confidence=round(max(1.0 - score, 0.5), 4))
        result.append(replace(d, alignment=alignment))
        explained.add(out.index)
    for index, score in scores.items():
        if score < t.grounded and index not in explained and index in outputs:
            partner = partners.get(index, (None, None))[0]
            result.append(_unsupported(partner, outputs[index], score, t.fabrication))
    return result


_MODES = (None, "strict", "rag", "summary")


def find_divergences(
    pair: CapturedPair,
    mode: str | None = None,
    *,
    thresholds: Thresholds | None = None,
    question: str | None = None,
) -> DivergenceReport:
    """Find all divergences between a captured source-output pair.

    This is the second step in Criterion. After capture, divergence.

    Compares molecules atom-by-atom, classifies each difference by
    type and severity, checks for broken bonds (evidence dropped
    while claims retained), and returns actionable corrections.

    Args:
        pair: A CapturedPair from the capture module.
        mode: "strict", "rag", "summary", or None.
            The caller's mode decides how lenient detection is. The
            OUTPUT's length never relaxes an explicitly chosen mode:
            "strict" stays strict however short the output is.
            None keeps the kernel's legacy behaviour, which infers
            summary leniency from the atom-count ratio.

        thresholds: Decision thresholds; defaults to Thresholds().
        question: The question the output answers, if any. With it, a
            fragment answer ("Delhi") is turned into a statement and
            judged by NLI instead of merely being located in the source.

    Returns:
        A DivergenceReport with every divergence classified, a
        `capabilities` dict naming the detectors that actually ran, and
        `support`: the best source support (0-1) NLI found for each
        output sentence it judged.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {_MODES}")
    t = thresholds or Thresholds()
    question = question.strip() if isinstance(question, str) and question.strip() else None

    source_atoms = pair.source.atoms
    output_atoms = pair.output.atoms
    caps: dict[str, Any] = {
        "scorer": False, "scorer_error": None if _USE_SCORER else "criterion.grounding not importable",
        "nli": False, "semantic": False,
        "nli_error": None if _USE_NLI else "criterion.nli not importable",
        "semantic_error": None if _USE_SEMANTIC else "criterion.semantic not importable",
    }

    matched, unmatched_src, unmatched_out = _align_atoms(source_atoms, output_atoms)
    if question:
        # A fragment answer must be judged against the question, not against
        # whichever source sentence happens to contain its words: "Martin
        # Scorsese" aligns with "Martin Scorsese produced it." however wrong
        # an answer it is to "Who directed the film?".
        kept = []
        for src, out, sim in matched:
            if _is_fragment(out.text):
                unmatched_src.append(src)
                unmatched_out.append(out)
            else:
                kept.append((src, out, sim))
        matched = kept
        unmatched_src.sort(key=lambda a: a.index)
        unmatched_out.sort(key=lambda a: a.index)

    # Leniency comes from the mode, not from how long the output happens to be.
    total_source = len(source_atoms)
    total_output = len(output_atoms)
    output_ratio = total_output / total_source if total_source > 0 else 1.0
    short_output = output_ratio < 0.5
    lenient_omissions = mode in ("rag", "summary") or (mode is None and short_output)
    compress_ok = mode in (None, "summary") and short_output
    legacy_addition_downgrade = mode in (None, "rag", "summary") and short_output

    # Semantic similarity for matched pairs (if available)
    sem_sims: list[float | None] = [None] * len(matched)
    if _USE_SEMANTIC and matched:
        try:
            sem_sims = batch_semantic_similarity(
                [_clean(src.text) for src, _, _ in matched],
                [_clean(out.text) for _, out, _ in matched],
            )
            caps["semantic"] = True
        except Exception as e:
            caps["semantic_error"] = f"{type(e).__name__}: {e}"

    # NLI on matched pairs, both directions (if available)
    fwd: list[Any] = [None] * len(matched)
    bwd: list[Any] = [None] * len(matched)
    nli_ok = False
    if _USE_NLI:
        try:
            src_texts = [_clean(src.text) for src, _, _ in matched]
            out_texts = [_clean(out.text) for _, out, _ in matched]
            both = _nli_probs(src_texts + out_texts, out_texts + src_texts)
            fwd, bwd = both[:len(matched)], both[len(matched):]
            nli_ok = True
        except Exception as e:
            caps["nli_error"] = f"{type(e).__name__}: {e}"

    # Ground every output atom that alignment + pairwise NLI left unsettled:
    # unmatched output atoms, and matched ones their partner does not entail.
    fragments = {a.index for a in output_atoms if _is_fragment(a.text)}
    # Fragments NLI cannot judge on their own. With the question, a fragment
    # becomes a statement ("The answer to ... is Delhi.") and is judged.
    unjudgeable = set() if question else fragments
    statements = (
        {a.index: _as_statement(question, a.text) for a in unmatched_out if a.index in fragments}
        if question else {}
    )
    # The grounding scorer: P(supported) for every output sentence that makes
    # a statement, taken as the maximum over chunks of the source.
    scores: dict[int, float] = {}
    if _USE_SCORER:
        # Fragments are not scored here. With a question they become
        # statements ("The answer to ... is Delhi."), and those are judged by
        # the NLI model below: measured on answer fragments, the grounding
        # scorer rejected 82% of CORRECT answers phrased that way.
        judged = [a for a in output_atoms if a.index not in fragments]
        try:
            results = _scorer_support(
                _clean(pair.source.raw_text), [_clean(a.text) for a in judged],
            )
            scores = {a.index: float(sc) for a, (sc, _) in zip(judged, results)}
            caps["scorer"] = True
        except Exception as e:
            caps["scorer_error"] = f"{type(e).__name__}: {e}"
    scorer_ok = caps["scorer"]

    grounding: dict[int, _Grounding] = {}
    if nli_ok:
        need = [a for a in unmatched_out if a.index not in unjudgeable] + [
            out for (_, out, _), f in zip(matched, fwd)
            if f.entailment < t.grounded and out.index not in fragments
        ]
        if scorer_ok and mode != "strict":
            # The scorer has already decided what is supported. NLI is only
            # needed to say WHY a failed sentence fails (contradiction or no
            # support). Strict mode still grounds everything, because it also
            # uses the result to decide which source sentences were omitted.
            need = [
                a for a in need
                if a.index in statements or scores.get(a.index, 0.0) < t.grounded
            ]
        try:
            grounding = _ground(need, source_atoms, statements)
        except Exception as e:
            nli_ok = False
            caps["nli_error"] = f"{type(e).__name__}: {e}"
    caps["nli"] = nli_ok
    if _USE_SEMANTIC and not caps["semantic"] and not caps["semantic_error"]:
        caps["semantic"] = True  # available; nothing needed encoding

    # Best source support per judged output sentence (the groundedness score).
    support: dict[int, float] = {}
    if nli_ok:
        for (_, out, _), f in zip(matched, fwd):
            if f is not None and out.index not in fragments:
                support[out.index] = f.entailment
        for idx, g in grounding.items():
            support[idx] = max(support.get(idx, 0.0), g.entailment)

    divergences: list[Divergence] = []
    covered_src: set[int] = {src.index for src, _, _ in matched}
    released_src: list[Atom] = []   # matched sources whose partner was re-aligned away

    def _downgrade_if_compressing(div: Divergence) -> Divergence:
        # Summaries reword and compress; that's their job. NEVER downgrade
        # CRITICAL findings — numeric changes and inversions are factual errors.
        if compress_ok and div.severity != Severity.CRITICAL and div.dtype in (
            DivergenceType.STRENGTHENING, DivergenceType.WEAKENING,
            DivergenceType.CONCEALING, DivergenceType.SURFACING,
        ):
            return replace(div, severity=Severity.MINOR, confidence=min(div.confidence, 0.4))
        return div

    for i, (src, out, sim) in enumerate(matched):
        f, b = fwd[i], bwd[i]
        # A fragment that merely truncates its source sentence ("Todd Phillips"
        # from "Todd Phillips directed the film.") asserts nothing new; NLI on
        # it is noise. A fragment with new words ("Revenue decreased") is judged.
        truncation = out.index in fragments and set(_words(out.text)) <= set(_words(src.text))
        if not nli_ok or f is None or truncation:
            div = _classify_matched_pair(src, out, sim, semantic_sim=sem_sims[i])
            if div is not None:
                divergences.append(_downgrade_if_compressing(div))
            continue

        if f.entailment >= t.grounded:
            equivalent = (
                f.entailment >= t.equivalence
                and b.entailment >= t.equivalence
            )
            div = _classify_matched_pair(
                src, out, sim, semantic_sim=sem_sims[i],
                nli_label="entailment" if f.label == "entailment" else None,
                nli_equivalent=equivalent,
            )
            if (
                div is None and mode == "strict"
                and b.entailment < t.grounded
                and out.index not in fragments
            ):
                # The source implies the output but not the reverse: the
                # output is true and vaguer ("March 15, 2027" → "early 2027").
                # Only strict mode treats lost precision as a finding.
                div = Divergence(
                    dtype=DivergenceType.CONCEALING,
                    severity=Severity.MAJOR,
                    source_atom=src,
                    output_atom=out,
                    explanation="Detail lost: the output is implied by the source but no longer implies it (less specific)",
                    confidence=0.7,
                    suggested_correction=src.text,
                    verification_method="Compare the two sentences for dropped specifics (dates, amounts, names, conditions)",
                    code="detail_lost",
                )
            if div is not None:
                divergences.append(_downgrade_if_compressing(div))
            continue

        # The aligned source atom does not entail this output atom.
        g = grounding.get(out.index)
        other = g.entail_atom if g else None
        if (
            g is not None and other is not None and other.index != src.index
            and g.entail_atom_score >= t.realign
        ):
            # Alignment picked the wrong partner: another source atom says
            # exactly this. Judge against that one; release the original.
            covered_src.add(other.index)
            covered_src.discard(src.index)
            released_src.append(src)
            div = _classify_matched_pair(
                other, out, _similarity(other.text, out.text), nli_label="entailment",
            )
            if div is not None:
                divergences.append(_downgrade_if_compressing(div))
            continue

        if f.contradiction >= t.contradiction and f.label == "contradiction":
            divergences.append(_contradiction(src, out, f.contradiction, "nli_contradiction"))
            continue
        if g is not None and g.contradiction >= t.contradiction and g.contradiction > g.entailment:
            divergences.append(_contradiction(g.contra_atom, out, g.contradiction, "contradicts_source"))
            continue

        supported_elsewhere = g is not None and g.entailment >= t.grounded
        div = _classify_matched_pair(src, out, sim, semantic_sim=sem_sims[i])
        # A specific deterministic finding (changed number, flipped negation,
        # shifted hedge) explains the mismatch better than "unsupported".
        # Vague ones ("expanded", "modified") do not: if nothing in the
        # source entails this sentence, say so.
        specific = div is not None and (
            div.severity == Severity.CRITICAL
            or div.code in ("strengthening", "weakening")
        )
        if not specific and not supported_elsewhere:
            div = _unsupported(src, out, g.entailment if g else f.entailment, t.fabrication)
        if div is not None:
            divergences.append(_downgrade_if_compressing(div))

    # Unmatched output atoms: supported, contradicted, or ungrounded?
    pending_additions: list[Atom] = []
    source_vocab: set[str] | None = None
    for atom in unmatched_out:
        if nli_ok and atom.index in unjudgeable:
            if source_vocab is None:
                source_vocab = set(_words(pair.source.raw_text))
            div = _classify_fragment(atom, source_vocab)
            if div is not None:
                divergences.append(div)
            continue
        g = grounding.get(atom.index) if nli_ok else None
        if g is None and scorer_ok and atom.index in scores:
            if scores[atom.index] < t.grounded:
                divergences.append(_unsupported(None, atom, scores[atom.index], t.fabrication))
            continue                         # the scorer decides; see _arbitrate
        if g is None:
            pending_additions.append(atom)   # no model at all: legacy handling below
            continue
        if g.entailment >= t.grounded:
            # Some source passage supports this — a paraphrase that alignment
            # missed, not an addition. (Retrieved documents may disagree with
            # each other; support from any of them grounds the statement.)
            # The source atom it restates is therefore not an omission either.
            if g.entail_atom is not None and g.entail_atom_score >= t.grounded:
                covered_src.add(g.entail_atom.index)
            continue
        if g.contradiction >= t.contradiction:
            divergences.append(_contradiction(g.contra_atom, atom, g.contradiction, "contradicts_source"))
            if g.contra_atom is not None:
                covered_src.add(g.contra_atom.index)  # addressed (wrongly), not omitted
            continue
        # A fragment judged through its question IS the whole answer, so an
        # unsupported one blocks; an unsupported sentence among others only
        # blocks when its support is below the fabrication threshold.
        bar = t.grounded if atom.index in statements else t.fabrication
        divergences.append(_unsupported(None, atom, g.entailment, bar))

    if scorer_ok:
        divergences = _arbitrate(
            divergences, scores, {out.index: (src, sim) for src, out, sim in matched},
            {a.index: a for a in output_atoms}, t,
        )
        # Scorer scores for sentences; NLI support is kept only for fragment
        # answers judged through their question.
        support = {**{k: v for k, v in support.items() if k in statements}, **scores}

    # Source atoms still unaccounted for. With NLI, check whether the output
    # states them somewhere alignment did not look (merged/reworded sentences).
    omitted = [a for a in list(unmatched_src) + released_src if a.index not in covered_src]
    if nli_ok and omitted and output_atoms and not lenient_omissions:
        try:
            reverse = _ground(omitted, output_atoms)
            omitted = [
                a for a in omitted
                if reverse[a.index].entailment < t.grounded
            ]
        except Exception as e:
            caps["nli_error"] = f"{type(e).__name__}: {e}"
    omitted.sort(key=lambda a: a.index)

    # Bond analysis (computed before omissions so we can check bond context)
    omitted_indices = {a.index for a in omitted}
    kept_indices = {a.index for a in source_atoms} - omitted_indices
    broken_bonds = _find_broken_bonds(pair.source, kept_indices, omitted_indices)
    broken_source_indices = {bd.source_atom.index for bd in broken_bonds}

    for atom in omitted:
        div = _classify_omission(atom, pair.source)
        # Where omission is expected (summary / RAG answer), plain omissions
        # are MINOR. Omissions that break an evidential bond keep their
        # severity — those are real integrity failures.
        if lenient_omissions and atom.index not in broken_source_indices:
            div = replace(div, severity=Severity.MINOR, confidence=0.3)
        divergences.append(div)

    # Additions without NLI: the legacy path. Embedding similarity cannot
    # tell a paraphrase from a contradiction, so this is strictly weaker
    # than grounding; capabilities["nli"] is False whenever it runs.
    add_max_sims: dict[int, float] = {}
    if pending_additions and legacy_addition_downgrade and _USE_SEMANTIC and unmatched_src:
        try:
            embs = _encode_texts(
                [_clean(a.text) for a in pending_additions]
                + [_clean(a.text) for a in unmatched_src]
            )
            sim_matrix = embs[:len(pending_additions)] @ embs[len(pending_additions):].T
            for oi, atom in enumerate(pending_additions):
                add_max_sims[atom.index] = float(sim_matrix[oi].max())
            caps["semantic"] = True
        except Exception as e:
            caps["semantic_error"] = f"{type(e).__name__}: {e}"

    for atom in pending_additions:
        div = _classify_addition(atom)
        if legacy_addition_downgrade:
            max_sim = add_max_sims.get(atom.index, 0.0)
            if max_sim >= 0.75:
                div = replace(
                    div, severity=Severity.COSMETIC, confidence=0.2,
                    explanation=f"Likely paraphrase of source content (sem={max_sim:.2f}): {atom.text[:60]}...",
                )
            elif max_sim >= 0.4:
                div = replace(
                    div, severity=Severity.MINOR, confidence=0.4,
                    explanation=f"Possible paraphrase in summary (sem={max_sim:.2f}): {atom.text[:60]}...",
                )
            else:
                div = replace(div, severity=Severity.MINOR, confidence=0.5)
        divergences.append(div)

    # Invisible / look-alike characters the output introduced
    for atom in output_atoms:
        hidden = _hidden_content(atom.text, pair.source.raw_text)
        if hidden:
            divergences.append(Divergence(
                dtype=DivergenceType.CONCEALING,
                severity=Severity.CRITICAL,
                source_atom=None,
                output_atom=atom,
                explanation=f"Output contains {hidden} not present in the source",
                confidence=0.9,
                suggested_correction="Remove the invisible or look-alike characters; what a reader sees must be what a machine reads",
                verification_method="Inspect the output's code points (e.g. unicodedata.name) rather than its rendering",
                code="hidden_chars",
            ))

    return DivergenceReport(
        pair=pair,
        divergences=tuple(divergences),
        bond_divergences=tuple(broken_bonds),
        source_atoms_matched=len(matched) - len(released_src),
        source_atoms_unmatched=len(omitted),
        output_atoms_unmatched=len(unmatched_out),
        capabilities=caps,
        support={k: round(v, 4) for k, v in sorted(support.items())},
    )
