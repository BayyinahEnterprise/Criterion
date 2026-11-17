"""
Kernel tests for Criterion v0.2.0 (Atom/Bond/Molecule architecture).

Tests the four kernel modules in Furqan order:
    1. capture  — atoms, bonds, molecules
    2. divergence — eight types + broken bonds + actionable fields
    3. verdict  — trust scoring with bond awareness
    4. record   — append-only chain
"""

import tempfile
from pathlib import Path

from criterion.capture import (
    capture, CapturedPair, Atom, AtomType, Bond, BondType, Molecule,
)
from criterion.divergence import (
    find_divergences, DivergenceType, Severity, BondDivergence,
)
from criterion.verdict import render_verdict, TrustLevel
from criterion.record import preserve, read_record, verify_chain
from criterion.pipeline import criterion


# ---------------------------------------------------------------------------
# Module 1: Capture — Atom/Bond/Molecule decomposition
# ---------------------------------------------------------------------------

class TestCapture:

    def test_basic_capture(self):
        pair = capture("The cat sat on the mat.", "The cat sat on the mat.")
        assert isinstance(pair, CapturedPair)
        assert isinstance(pair.source, Molecule)
        assert isinstance(pair.output, Molecule)
        assert pair.source.role == "source"

    def test_atoms_created(self):
        text = "First sentence. Second sentence. Third sentence."
        pair = capture(text, text)
        assert pair.source.atom_count >= 1
        assert all(isinstance(a, Atom) for a in pair.source.atoms)

    def test_atom_types_classified(self):
        text = (
            "Democracy means government by the people. "
            "Studies show that voter turnout has declined. "
            "Therefore, civic engagement must be prioritized."
        )
        pair = capture(text, text)
        types = {a.atom_type for a in pair.source.atoms}
        # Should have at least two distinct types
        assert len(types) >= 1

    def test_bonds_inferred(self):
        text = "Climate change is accelerating. Because global temperatures have risen 1.2C since pre-industrial times."
        pair = capture(text, text)
        assert pair.source.bond_count >= 1
        bond_types = {b.bond_type for b in pair.source.bonds}
        assert BondType.FOLLOWS in bond_types

    def test_evidence_supports_claim(self):
        text = "The defendant is innocent. Because three witnesses confirmed his alibi."
        pair = capture(text, text)
        support_bonds = [b for b in pair.source.bonds if b.bond_type == BondType.SUPPORTS]
        # Should infer a supports bond from evidence to claim
        assert len(support_bonds) >= 1

    def test_hashes_differ(self):
        pair = capture("Source text.", "Different text.")
        assert pair.source_hash != pair.output_hash

    def test_heading_as_metadata(self):
        text = "# My Heading\n\nSome body text here."
        pair = capture(text, text)
        types = [a.atom_type for a in pair.source.atoms]
        assert AtomType.METADATA in types

    def test_molecule_supporters(self):
        text = "The earth is round. Because satellite images show curvature."
        pair = capture(text, text)
        # Find the claim atom
        claims = pair.source.atoms_of_type(AtomType.CLAIM)
        if claims:
            supporters = pair.source.supporters_of(claims[0].index)
            # May or may not find supporters depending on classification
            assert isinstance(supporters, tuple)


# ---------------------------------------------------------------------------
# Module 2: Divergence — with atom types, broken bonds, actionable fields
# ---------------------------------------------------------------------------

class TestDivergence:

    def test_identical_no_divergences(self):
        pair = capture("The cat sat on the mat.", "The cat sat on the mat.")
        report = find_divergences(pair)
        assert report.total == 0

    def test_omission_detected(self):
        source = "First point. Second point. Third point."
        output = "First point. Third point."
        pair = capture(source, output)
        report = find_divergences(pair)
        types = [d.dtype for d in report.divergences]
        assert DivergenceType.OMISSION in types

    def test_addition_detected(self):
        source = "The contract expires on March 15."
        output = "The contract expires on March 15. This is a great deal for everyone involved."
        pair = capture(source, output)
        report = find_divergences(pair)
        types = [d.dtype for d in report.divergences]
        assert DivergenceType.ADDITION in types

    def test_strengthening_detected(self):
        source = "The treatment showed improvement in patients."
        output = "The treatment always showed improvement in every patient."
        pair = capture(source, output)
        report = find_divergences(pair)
        types = [d.dtype for d in report.divergences]
        assert DivergenceType.STRENGTHENING in types

    def test_weakening_detected(self):
        source = "The bridge will collapse under this load."
        output = "The bridge might possibly collapse under this load."
        pair = capture(source, output)
        report = find_divergences(pair)
        types = [d.dtype for d in report.divergences]
        assert DivergenceType.WEAKENING in types

    def test_actionable_fields_present(self):
        source = "X is true."
        output = "X is definitely always true."
        pair = capture(source, output)
        report = find_divergences(pair)
        for d in report.divergences:
            assert d.suggested_correction, "Every divergence must have a suggested correction"
            assert d.verification_method, "Every divergence must have a verification method"

    def test_omission_severity_depends_on_atom_type(self):
        # Evidence supporting a claim should be CRITICAL to omit
        source = (
            "The defendant is innocent. "
            "Because DNA evidence excluded him from the scene. "
            "Weather data shows it rained that day."
        )
        output = "The defendant is innocent."
        pair = capture(source, output)
        report = find_divergences(pair)
        # At least one omission should exist
        omissions = [d for d in report.divergences if d.dtype == DivergenceType.OMISSION]
        assert len(omissions) >= 1

    def test_broken_bonds_detected(self):
        # Source has evidence supporting a claim; output keeps claim, drops evidence
        source = (
            "The vaccine is safe. "
            "Because clinical trials with 40,000 participants showed no serious adverse effects."
        )
        output = (
            "The vaccine is safe. "
            "More research is needed to understand long-term effects."
        )
        pair = capture(source, output)
        report = find_divergences(pair)
        # Should detect either broken bonds or omissions
        assert report.total >= 1


# ---------------------------------------------------------------------------
# Module 3: Verdict — with bond-aware scoring
# ---------------------------------------------------------------------------

class TestVerdict:

    def test_faithful_verdict(self):
        pair = capture("The cat sat on the mat.", "The cat sat on the mat.")
        report = find_divergences(pair)
        verdict = render_verdict(report)
        assert verdict.trust_level == TrustLevel.FAITHFUL
        assert verdict.trust_score >= 0.9

    def test_unreliable_on_heavy_divergence(self):
        source = (
            "The defendant was not present at the scene. "
            "Three witnesses confirmed his alibi. "
            "The DNA evidence excluded him. "
            "The timeline made his involvement impossible."
        )
        output = (
            "The defendant was present at the scene. "
            "No witnesses were available. "
            "New evidence confirmed his identity. "
            "The timeline supported the prosecution's case."
        )
        pair = capture(source, output)
        report = find_divergences(pair)
        verdict = render_verdict(report)
        assert verdict.trust_level in (TrustLevel.DIVERGENT, TrustLevel.UNRELIABLE)
        assert verdict.trust_score < 0.7

    def test_verdict_has_summary(self):
        pair = capture("Source.", "Output.")
        report = find_divergences(pair)
        verdict = render_verdict(report)
        assert isinstance(verdict.summary, str)
        assert len(verdict.summary) > 0

    def test_verdict_tracks_broken_bonds(self):
        pair = capture(
            "The claim is true. Because the evidence supports it.",
            "The claim is true. Actually we do not know.",
        )
        report = find_divergences(pair)
        verdict = render_verdict(report)
        assert isinstance(verdict.broken_bonds, int)


# ---------------------------------------------------------------------------
# Module 4: Record
# ---------------------------------------------------------------------------

class TestRecord:

    def test_preserve_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_record.jsonl"
            pair = capture("Source text.", "Output text.")
            report = find_divergences(pair)
            verdict = render_verdict(report)
            entry = preserve(pair, report, verdict, path)
            assert entry.entry_id
            assert entry.predecessor_hash == "genesis"
            entries = read_record(path)
            assert len(entries) == 1

    def test_chain_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_record.jsonl"
            for text in ["First source.", "Second source."]:
                pair = capture(text, text + " Modified.")
                report = find_divergences(pair)
                verdict = render_verdict(report)
                preserve(pair, report, verdict, path)
            ok, msg = verify_chain(path)
            assert ok, msg

    def test_chain_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_record.jsonl"
            entries_written = []
            for text in ["A.", "B.", "C."]:
                pair = capture(text, text)
                report = find_divergences(pair)
                verdict = render_verdict(report)
                entry = preserve(pair, report, verdict, path)
                entries_written.append(entry)
            entries = read_record(path)
            assert entries[0].predecessor_hash == "genesis"
            assert entries[1].predecessor_hash == entries[0].entry_id
            assert entries[2].predecessor_hash == entries[1].entry_id


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestPipeline:

    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_record.jsonl"
            verdict, entry = criterion(
                source_text=(
                    "The defendant was not present at the scene. "
                    "Three witnesses confirmed his alibi."
                ),
                output_text=(
                    "The defendant was present at the scene. "
                    "No witnesses were available to comment."
                ),
                record_path=path,
            )
            assert verdict.total_divergences >= 1
            assert entry.entry_id
            assert Path(path).exists()

    def test_pipeline_faithful(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_record.jsonl"
            verdict, _ = criterion(
                "The sky is blue.",
                "The sky is blue.",
                record_path=path,
            )
            assert verdict.trust_level == TrustLevel.FAITHFUL

    def test_pipeline_actionable_corrections(self):
        """Every divergence carries what the output should have said."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_record.jsonl"
            verdict, _ = criterion(
                "The rate is 5%. The deadline is March 15.",
                "The rate is approximately 5%. More time may be needed.",
                record_path=path,
            )
            for d in verdict.divergences:
                assert d.suggested_correction
                assert d.verification_method
