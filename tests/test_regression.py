"""
Regression tests for every defect found in the v0.2.0 independent audits.

Each test corresponds to a specific audit finding from GPT-6, Grok, Kimi,
and the fourth auditor. Tests are named after the finding ID.

Auditors: GPT-6 (C-01 through H-03), Grok, Kimi, fourth auditor.
"""

import json
import tempfile
from pathlib import Path

import criterion
from criterion.api import verify, Mode
from criterion.capture import capture
from criterion.divergence import find_divergences, DivergenceType, Severity
from criterion.verdict import render_verdict, TrustLevel
from criterion.record import preserve, verify_chain, read_record


# ---------------------------------------------------------------------------
# C-01: Audit chain must detect content tampering
# ---------------------------------------------------------------------------

class TestAuditChainIntegrity:

    def test_unmodified_chain_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.jsonl"
            pair = capture("Source.", "Source.")
            report = find_divergences(pair)
            verdict = render_verdict(report)
            preserve(pair, report, verdict, path)
            ok, msg = verify_chain(path)
            assert ok, msg

    def test_tampered_trust_score_detected(self):
        """C-01: Modifying trust_score must break the chain."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.jsonl"
            pair = capture("Source text.", "Source text.")
            report = find_divergences(pair)
            verdict = render_verdict(report)
            preserve(pair, report, verdict, path)

            # Tamper with the trust_score
            with open(path) as f:
                data = json.loads(f.read().strip())
            data["trust_score"] = 0.1  # Changed from original
            with open(path, "w") as f:
                f.write(json.dumps(data) + "\n")

            ok, msg = verify_chain(path)
            assert not ok, "Tampered trust_score should break the chain"
            assert "content hash mismatch" in msg or "modified" in msg

    def test_tampered_source_text_detected(self):
        """C-01: Modifying source_text must break the chain."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.jsonl"
            pair = capture("Original source.", "Original source.")
            report = find_divergences(pair)
            verdict = render_verdict(report)
            preserve(pair, report, verdict, path)

            with open(path) as f:
                data = json.loads(f.read().strip())
            data["source_text"] = "ATTACKER CONTROLLED"
            with open(path, "w") as f:
                f.write(json.dumps(data) + "\n")

            ok, msg = verify_chain(path)
            assert not ok, "Tampered source text should break the chain"

    def test_tampered_summary_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.jsonl"
            pair = capture("Source.", "Modified output.")
            report = find_divergences(pair)
            verdict = render_verdict(report)
            preserve(pair, report, verdict, path)

            with open(path) as f:
                data = json.loads(f.read().strip())
            data["summary"] = "Everything is fine, nothing to see here"
            with open(path, "w") as f:
                f.write(json.dumps(data) + "\n")

            ok, msg = verify_chain(path)
            assert not ok


# ---------------------------------------------------------------------------
# C-02: RAG mode must not crash on Severity comparison
# ---------------------------------------------------------------------------

class TestRAGCrash:

    def test_rag_mode_surfacing_no_crash(self):
        """C-02: RAG mode must not TypeError on Severity comparison."""
        # This would previously crash with:
        # TypeError: '<=' not supported between instances of 'Severity' and 'Severity'
        result = verify(
            source="The sky is blue.",
            output="The sky is blue and vast.",
            mode="rag",
        )
        assert isinstance(result.trust_level, str)

    def test_rag_mode_with_paraphrase(self):
        result = verify(
            source="The earth orbits the sun.",
            output="Our planet revolves around the sun.",
            mode="rag",
        )
        assert isinstance(result.trust_score, float)

    def test_all_modes_all_types_no_crash(self):
        """Every mode × divergence type combination must not crash."""
        pairs = [
            ("First. Second. Third.", "First. Third."),  # omission
            ("The sky is blue.", "The sky is blue. It is also beautiful."),  # addition
            ("It may work.", "It always works."),  # strengthening
            ("It works.", "It might work."),  # weakening
        ]
        for mode in ["rag", "summary", "strict", "auto"]:
            for src, out in pairs:
                result = verify(source=src, output=out, mode=mode)
                assert isinstance(result.trust_level, str), f"Crashed on mode={mode}"


# ---------------------------------------------------------------------------
# C-03: Batch CLI must return nonzero on failures
# ---------------------------------------------------------------------------

class TestBatchExitCodes:

    def test_batch_with_untrustworthy_returns_nonzero(self):
        """C-03: Batch must not return 0 when results are untrustworthy."""
        from criterion.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            pairs_path = Path(tmp) / "pairs.jsonl"
            out_path = Path(tmp) / "results.jsonl"
            pairs_path.write_text(
                json.dumps({"source": "The rate is 5%.", "output": "The rate is 50%."}) + "\n"
            )
            exit_code = main(["batch", str(pairs_path), "-o", str(out_path)])
            assert exit_code != 0, "Batch with untrustworthy content must return nonzero"


# ---------------------------------------------------------------------------
# C-04: Critical numeric/negation must not be downgraded by summary heuristics
# ---------------------------------------------------------------------------

class TestCriticalNotDowngraded:

    def test_numeric_change_not_downgraded_in_summary(self):
        """C-04: Numeric changes must stay CRITICAL even in summary-length output."""
        result = verify(
            source="The trial enrolled 400 patients. The drug showed 60% efficacy.",
            output="The trial enrolled 4,000 patients.",
            mode="strict",
        )
        critical = [d for d in result.divergences if d.severity == Severity.CRITICAL]
        assert len(critical) >= 1, "Numeric change must remain CRITICAL"

    def test_negation_inversion_not_downgraded(self):
        result = verify(
            source="The patient has no allergy. The doctor confirmed this.",
            output="The patient has an allergy.",
            mode="strict",
        )
        assert not result.is_trustworthy, "Negation inversion must be untrustworthy"

    def test_predicate_reversal_detected(self):
        """'safe' → 'unsafe' must be caught."""
        result = verify(
            source="The drug is safe.",
            output="The drug is unsafe.",
            mode="strict",
        )
        assert not result.is_trustworthy, "Predicate reversal must be untrustworthy"
        critical = [d for d in result.divergences if d.severity == Severity.CRITICAL]
        assert len(critical) >= 1


# ---------------------------------------------------------------------------
# Empty input handling
# ---------------------------------------------------------------------------

class TestEmptyInputs:

    def test_empty_output_is_unreliable(self):
        result = verify(source="Some source text.", output="", mode="rag")
        assert result.trust_level == "unreliable"
        assert not result.is_trustworthy

    def test_empty_output_is_unreliable_summary(self):
        result = verify(source="Some source text.", output="", mode="summary")
        assert not result.is_trustworthy

    def test_empty_source_nonempty_output_is_unreliable(self):
        result = verify(source="", output="Some AI output.", mode="strict")
        assert not result.is_trustworthy

    def test_whitespace_only_output_is_unreliable(self):
        result = verify(source="Real source.", output="   \n\t  ", mode="rag")
        assert not result.is_trustworthy


# ---------------------------------------------------------------------------
# Numeric detection: added numbers must be caught
# ---------------------------------------------------------------------------

class TestNumericDetection:

    def test_added_number_detected(self):
        """Numbers added by the output that aren't in the source must be flagged."""
        result = verify(
            source="The treatment improved patients.",
            output="The treatment improved 90% of patients.",
            mode="strict",
        )
        assert not result.is_trustworthy or any(
            d.severity == Severity.CRITICAL for d in result.divergences
        ), "Added number '90%' must be flagged"

    def test_changed_number_detected(self):
        result = verify(
            source="The study found that 60% of participants improved.",
            output="The study found that 90% of participants improved.",
            mode="strict",
        )
        assert not result.is_trustworthy

    def test_same_numbers_not_flagged(self):
        result = verify(
            source="The rate is 5%.",
            output="The rate is 5%.",
            mode="strict",
        )
        assert result.is_trustworthy


# ---------------------------------------------------------------------------
# Strengthening in summary mode
# ---------------------------------------------------------------------------

class TestSummaryModeStrengthening:

    def test_always_strengthening_in_summary(self):
        """'showed improvement' → 'always showed improvement in every patient' must be caught."""
        result = verify(
            source="The treatment showed improvement in patients.",
            output="The treatment always showed improvement in every patient.",
            mode="summary",
        )
        # Should have at least one divergence (strengthening with intensifiers)
        strengthening = [d for d in result.divergences if d.dtype == DivergenceType.STRENGTHENING]
        assert len(strengthening) >= 1 or not result.is_trustworthy, \
            "Strengthening with intensifiers must be detected even in summary mode"


# ---------------------------------------------------------------------------
# Fabricated claim in summary mode
# ---------------------------------------------------------------------------

class TestFabricatedClaim:

    def test_fabricated_claim_in_summary(self):
        """Adding 'The moon is made of cheese' to a summary must be caught."""
        result = verify(
            source="The earth is round. It orbits the sun in 365 days.",
            output="The earth is round. The moon is made of cheese.",
            mode="summary",
        )
        additions = [d for d in result.divergences if d.dtype == DivergenceType.ADDITION]
        assert len(additions) >= 1 or not result.is_trustworthy


# ---------------------------------------------------------------------------
# Recommended test matrix from GPT-6 audit
# ---------------------------------------------------------------------------

class TestAuditMatrix:

    def test_faithful_paraphrase_accepted(self):
        result = verify(
            source="The cat sat on the mat.",
            output="The cat was sitting on the mat.",
            mode="strict",
        )
        # Should not have critical findings for a faithful paraphrase
        critical = [d for d in result.divergences if d.severity == Severity.CRITICAL]
        assert len(critical) == 0, "Faithful paraphrase should not have critical findings"

    def test_may_to_will_strengthening(self):
        """'may improve' → 'will improve' should flag unjustified strengthening."""
        result = verify(
            source="The drug may improve outcomes.",
            output="The drug will improve outcomes.",
            mode="strict",
        )
        # 'may' removed = negation/hedge removed
        assert result.total_divergences >= 1

    def test_summary_valid_compression_accepted(self):
        """Summary that drops background but preserves central meaning should pass."""
        result = verify(
            source="Apple Inc., headquartered in Cupertino, California, reported "
                   "quarterly revenue of $94.8 billion. The company was founded by "
                   "Steve Jobs in 1976. iPhone revenue was $51.3 billion.",
            output="Apple reported $94.8 billion in revenue with iPhone at $51.3 billion.",
            mode="summary",
        )
        assert result.is_trustworthy, "Valid summary compression should be trustworthy"
