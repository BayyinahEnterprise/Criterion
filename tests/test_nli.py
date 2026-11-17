"""
NLI integration tests — verifies that the Natural Language Inference
model catches the entire class of "same words, different meaning" attacks
that heuristics cannot handle.

These are the exact adversarial cases from the GPT-6 red-team audit,
the fourth auditor's retest, and the Kimi red-team report.
"""

import pytest

from criterion.api import verify
from criterion.divergence import DivergenceType, Severity

# Skip all NLI tests if torch/transformers not installed
try:
    import torch
    import transformers
    _HAS_NLI = True
except ImportError:
    _HAS_NLI = False

pytestmark = pytest.mark.skipif(not _HAS_NLI, reason="NLI requires torch and transformers")


class TestNLIContradictions:
    """Every case here was a known false-pass in v0.2.3 without NLI."""

    def test_numeric_role_swap(self):
        """RT-01: Dosage 5mg limit 10mg → dosage 10mg limit 5mg."""
        result = verify(
            source="The dosage is 5 mg and the limit is 10 mg.",
            output="The dosage is 10 mg and the limit is 5 mg.",
            mode="strict",
        )
        assert not result.is_trustworthy, f"Numeric role swap must be untrustworthy: {result.to_dict()}"

    def test_predicate_swap(self):
        """RT-02: Safe for adults/unsafe for children → reversed."""
        result = verify(
            source="The treatment is safe for adults and unsafe for children.",
            output="The treatment is unsafe for adults and safe for children.",
            mode="strict",
        )
        assert not result.is_trustworthy

    def test_negation_scope_swap(self):
        """RT-03: Alice not approved / Bob approved → reversed."""
        result = verify(
            source="Alice is not approved and Bob is approved.",
            output="Alice is approved and Bob is not approved.",
            mode="strict",
        )
        assert not result.is_trustworthy

    def test_entity_substitution(self):
        """Alice → Bob should be caught."""
        result = verify(
            source="Alice signed the contract.",
            output="Bob signed the contract.",
            mode="strict",
        )
        assert not result.is_trustworthy

    def test_relation_reversal(self):
        """Alice sold to Bob → Bob sold to Alice."""
        result = verify(
            source="Alice sold the car to Bob.",
            output="Bob sold the car to Alice.",
            mode="strict",
        )
        assert not result.is_trustworthy

    def test_currency_change(self):
        """$1,000 → $2,000."""
        result = verify(
            source="The fee is $1,000.",
            output="The fee is $2,000.",
            mode="strict",
        )
        assert not result.is_trustworthy

    def test_currency_unit_change(self):
        """$100 → €100."""
        result = verify(
            source="The fee is $100.",
            output="The fee is €100.",
            mode="strict",
        )
        assert not result.is_trustworthy

    def test_inflected_antonym(self):
        """Revenue increased → Revenue decreased."""
        result = verify(
            source="Revenue increased.",
            output="Revenue decreased.",
            mode="strict",
        )
        assert not result.is_trustworthy

    def test_location_substitution(self):
        """Paris → London."""
        result = verify(
            source="The meeting is in Paris.",
            output="The meeting is in London.",
            mode="strict",
        )
        assert not result.is_trustworthy

    def test_unit_substitution(self):
        """5 mg → 5 g."""
        result = verify(
            source="The dosage is 5 mg.",
            output="The dosage is 5 g.",
            mode="strict",
        )
        assert not result.is_trustworthy


class TestNLIEntailments:
    """These should still be accepted as faithful paraphrases."""

    def test_faithful_paraphrase(self):
        result = verify(
            source="The cat sat on the mat.",
            output="The cat was sitting on the mat.",
            mode="strict",
        )
        assert result.is_trustworthy

    @pytest.mark.xfail(
        strict=False,
        reason="Known limitation of the default NLI model (tasksource/deberta-base-long-nli): "
               "it has no unit knowledge and scores '365 days' vs 'about a year' as a "
               "contradiction (0.89). Passes with CRITERION_NLI_MODEL=cross-encoder/"
               "nli-deberta-v3-small. Documented in README > Limitations.",
    )
    def test_faithful_numeric_paraphrase(self):
        result = verify(
            source="The earth orbits the sun in 365 days.",
            output="The earth orbits the sun in about a year.",
            mode="strict",
        )
        assert result.is_trustworthy

    def test_identical_text(self):
        result = verify(
            source="The sky is blue.",
            output="The sky is blue.",
            mode="strict",
        )
        assert result.is_trustworthy
