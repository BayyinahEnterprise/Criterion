"""
Module 1: Capture (the Witness)

Furqan reading order: 1 of 4 (kernel)
Dependency depth: 0 — depends on nothing else in Criterion

Takes in a source document and an AI-generated output.
Decomposes each into Atoms (typed semantic units), connects
them with Bonds (structural relationships), and composes
them into Molecules (complete documents with internal structure).

Produces a CapturedPair of two Molecules, ready for divergence
analysis.

Architecture inspired by FATIMA's holographic encoding:
    Atom     — the smallest semantic unit that can be independently
               correct or incorrect (a claim, a piece of evidence,
               a definition, a reference, a conclusion)
    Bond     — a typed relationship between two atoms
               (supports, contradicts, defines, cites, follows)
    Molecule — a complete document: atoms + bonds + content hash

This is the entry point. Without capture, nothing else runs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

from criterion._version import __version__


# ---------------------------------------------------------------------------
# Atom — the semantic unit
# ---------------------------------------------------------------------------

class AtomType(Enum):
    """What kind of semantic unit this atom carries.

    The type determines how severe its omission or modification is.
    Dropping evidence that supports a claim is worse than dropping
    a stylistic observation.
    """

    CLAIM = "claim"              # An assertion: "X is true"
    EVIDENCE = "evidence"        # Supports a claim: "because Y"
    DEFINITION = "definition"    # Defines a term or concept
    REFERENCE = "reference"      # Points to an external source
    CONCLUSION = "conclusion"    # A derived judgment or summary
    METADATA = "metadata"        # Structural/bibliographic info
    OBSERVATION = "observation"  # Descriptive, non-assertive content

    @property
    def omission_severity(self) -> str:
        """How bad is it if this atom is omitted from the output?"""
        critical = {AtomType.CLAIM, AtomType.EVIDENCE, AtomType.CONCLUSION}
        major = {AtomType.DEFINITION, AtomType.REFERENCE}
        return (
            "critical" if self in critical
            else "major" if self in major
            else "minor"
        )


@dataclass(frozen=True)
class Atom:
    """One typed semantic unit in a document.

    An Atom is the resolution at which divergences are detected.
    It knows what kind of content it carries (claim, evidence, etc.)
    and can compute a stable fingerprint for alignment.
    """

    index: int
    text: str
    atom_type: AtomType
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            # Frozen dataclass: use object.__setattr__
            h = hashlib.sha256(
                re.sub(r"\s+", " ", self.text.strip().lower()).encode("utf-8")
            ).hexdigest()[:16]
            object.__setattr__(self, "content_hash", h)

    @property
    def fingerprint(self) -> str:
        """Stable content hash for alignment."""
        return self.content_hash


# ---------------------------------------------------------------------------
# Bond — the relationship between atoms
# ---------------------------------------------------------------------------

class BondType(Enum):
    """How two atoms relate to each other.

    Bonds give the molecule its structure. Without bonds, a document
    is a list of sentences. With bonds, it is a connected argument.
    """

    SUPPORTS = "supports"        # Atom A is evidence for atom B
    CONTRADICTS = "contradicts"  # Atom A contradicts atom B
    DEFINES = "defines"          # Atom A defines a term used in atom B
    CITES = "cites"              # Atom A references atom B (or external)
    FOLLOWS = "follows"          # Atom A comes after atom B (sequence)
    ELABORATES = "elaborates"    # Atom A expands on atom B
    QUALIFIES = "qualifies"      # Atom A limits or constrains atom B


@dataclass(frozen=True)
class Bond:
    """A typed relationship between two atoms.

    source_index and target_index refer to Atom.index values
    within the same Molecule.
    """

    source_index: int
    target_index: int
    bond_type: BondType

    @property
    def is_evidential(self) -> bool:
        """Does this bond carry evidential weight?"""
        return self.bond_type in (BondType.SUPPORTS, BondType.CONTRADICTS)


# ---------------------------------------------------------------------------
# Molecule — the composed document
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Molecule:
    """A complete document decomposed into atoms and bonds.

    The molecule is the unit that the divergence engine compares.
    Two molecules (source and output) are aligned atom-by-atom,
    and their bond structures are compared for missing or altered
    relationships.
    """

    atoms: tuple[Atom, ...]
    bonds: tuple[Bond, ...]
    raw_text: str
    sha256: str
    role: str  # "source" | "output"

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

    @property
    def bond_count(self) -> int:
        return len(self.bonds)

    def atoms_of_type(self, atom_type: AtomType) -> tuple[Atom, ...]:
        """Return all atoms of a given type."""
        return tuple(a for a in self.atoms if a.atom_type == atom_type)

    def bonds_for_atom(self, index: int) -> tuple[Bond, ...]:
        """Return all bonds involving an atom (as source or target)."""
        return tuple(
            b for b in self.bonds
            if b.source_index == index or b.target_index == index
        )

    def supporters_of(self, index: int) -> tuple[Atom, ...]:
        """Return atoms that support the given atom."""
        support_bonds = [
            b for b in self.bonds
            if b.target_index == index and b.bond_type == BondType.SUPPORTS
        ]
        atom_map = {a.index: a for a in self.atoms}
        return tuple(atom_map[b.source_index] for b in support_bonds if b.source_index in atom_map)


# ---------------------------------------------------------------------------
# CapturedPair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapturedPair:
    """The aligned pair ready for divergence analysis.

    This is what the Capture module produces and the Divergence
    module consumes. It is the contract between modules 1 and 2.
    """

    source: Molecule
    output: Molecule
    captured_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    capture_version: str = __version__

    @property
    def source_hash(self) -> str:
        return self.source.sha256

    @property
    def output_hash(self) -> str:
        return self.output.sha256


# ---------------------------------------------------------------------------
# Decomposition engine
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"“(])"
    r"|(?<=\n)\s*\n"
)

_HEADING_PATTERN = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
_LIST_ITEM_PATTERN = re.compile(r"^[\s]*[-*•]\s+.+$", re.MULTILINE)

# Heuristics for atom type classification
_CLAIM_SIGNALS = re.compile(
    r"\b(is|are|was|were|will|must|should|shall|can|cannot|proves?|demonstrates?|shows?|confirms?|establishes?)\b",
    re.IGNORECASE,
)
_EVIDENCE_SIGNALS = re.compile(
    r"\b(because|since|evidence|data|study|found|showed|measured|observed|according to|reported)\b",
    re.IGNORECASE,
)
_DEFINITION_SIGNALS = re.compile(
    r"\b(means?|defined as|refers to|is called|known as|i\.e\.|that is)\b",
    re.IGNORECASE,
)
_REFERENCE_SIGNALS = re.compile(
    r"\b(see|cf\.|ibid|op\.?\s?cit|cited in|\[\d+\]|\(\d{4}\))\b",
    re.IGNORECASE,
)
_CONCLUSION_SIGNALS = re.compile(
    r"\b(therefore|thus|consequently|in conclusion|overall|in summary|hence|accordingly)\b",
    re.IGNORECASE,
)


def _classify_atom_type(text: str, kind: str) -> AtomType:
    """Classify the semantic type of a text segment.

    Uses structural signals (headings, list markers) and lexical
    heuristics. The classification is best-effort — the user can
    always override.
    """
    if kind == "heading":
        return AtomType.METADATA

    # Score each type by signal presence
    scores: dict[AtomType, int] = {t: 0 for t in AtomType}

    if _CONCLUSION_SIGNALS.search(text):
        scores[AtomType.CONCLUSION] += 3
    if _EVIDENCE_SIGNALS.search(text):
        scores[AtomType.EVIDENCE] += 2
    if _DEFINITION_SIGNALS.search(text):
        scores[AtomType.DEFINITION] += 2
    if _REFERENCE_SIGNALS.search(text):
        scores[AtomType.REFERENCE] += 2
    if _CLAIM_SIGNALS.search(text):
        scores[AtomType.CLAIM] += 1  # Claims are common, low signal

    best = max(scores, key=lambda t: scores[t])
    if scores[best] == 0:
        return AtomType.OBSERVATION  # No signal → descriptive
    return best


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences."""
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p for p in parts if p and p.strip()]


def _decompose_text(text: str) -> tuple[list[Atom], list[Bond]]:
    """Decompose text into Atoms and infer Bonds between them.

    Pass 1: segment into atoms with type classification.
    Pass 2: infer bonds from adjacency and type relationships.
    """
    atoms: list[Atom] = []
    idx = 0

    # --- Pass 1: Segment and classify ---

    special_spans: list[tuple[int, int, str]] = []
    for m in _HEADING_PATTERN.finditer(text):
        special_spans.append((m.start(), m.end(), "heading"))
    for m in _LIST_ITEM_PATTERN.finditer(text):
        if not any(s <= m.start() < e for s, e, _ in special_spans):
            special_spans.append((m.start(), m.end(), "list_item"))
    special_spans.sort(key=lambda s: s[0])

    pos = 0
    for span_start, span_end, kind in special_spans:
        if pos < span_start:
            prose = text[pos:span_start].strip()
            if prose:
                for sentence in _split_sentences(prose):
                    s = sentence.strip()
                    if s:
                        atype = _classify_atom_type(s, "sentence")
                        atoms.append(Atom(index=idx, text=s, atom_type=atype))
                        idx += 1
        content = text[span_start:span_end].strip()
        if content:
            atype = _classify_atom_type(content, kind)
            atoms.append(Atom(index=idx, text=content, atom_type=atype))
            idx += 1
        pos = span_end

    if pos < len(text):
        prose = text[pos:].strip()
        if prose:
            for sentence in _split_sentences(prose):
                s = sentence.strip()
                if s:
                    atype = _classify_atom_type(s, "sentence")
                    atoms.append(Atom(index=idx, text=s, atom_type=atype))
                    idx += 1

    # --- Pass 2: Infer bonds ---
    bonds: list[Bond] = []

    for i in range(1, len(atoms)):
        prev = atoms[i - 1]
        curr = atoms[i]

        # Sequential bond (always)
        bonds.append(Bond(
            source_index=prev.index,
            target_index=curr.index,
            bond_type=BondType.FOLLOWS,
        ))

        # Evidence → preceding claim
        if curr.atom_type == AtomType.EVIDENCE and prev.atom_type == AtomType.CLAIM:
            bonds.append(Bond(
                source_index=curr.index,
                target_index=prev.index,
                bond_type=BondType.SUPPORTS,
            ))

        # Definition → preceding or following atom that uses defined term
        if curr.atom_type == AtomType.DEFINITION:
            bonds.append(Bond(
                source_index=curr.index,
                target_index=prev.index,
                bond_type=BondType.DEFINES,
            ))

        # Conclusion → all preceding claims
        if curr.atom_type == AtomType.CONCLUSION:
            for j in range(i):
                if atoms[j].atom_type == AtomType.CLAIM:
                    bonds.append(Bond(
                        source_index=atoms[j].index,
                        target_index=curr.index,
                        bond_type=BondType.SUPPORTS,
                    ))

        # Elaboration: observation after a claim
        if curr.atom_type == AtomType.OBSERVATION and prev.atom_type == AtomType.CLAIM:
            bonds.append(Bond(
                source_index=curr.index,
                target_index=prev.index,
                bond_type=BondType.ELABORATES,
            ))

    return atoms, bonds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def capture(source_text: str, output_text: str) -> CapturedPair:
    """Capture a source document and an AI-generated output.

    This is the entry point to Criterion. Everything starts here.

    Decomposes each text into a Molecule (atoms + bonds + hash),
    producing a CapturedPair ready for divergence analysis.

    Args:
        source_text: The original source material the AI was given.
        output_text: The AI-generated output to verify against the source.

    Returns:
        A CapturedPair of two Molecules.
    """
    source_mol = _build_molecule(source_text, role="source")
    output_mol = _build_molecule(output_text, role="output")
    return CapturedPair(source=source_mol, output=output_mol)


def _build_molecule(text: str, role: str) -> Molecule:
    """Build a Molecule from raw text."""
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    atoms, bonds = _decompose_text(text)
    return Molecule(
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        raw_text=text,
        sha256=sha,
        role=role,
    )
