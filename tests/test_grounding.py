"""Tests for the grounding scorer integration (v0.5.0).

The arbitration tests stub the scorer, so they run without any model and
pin the decision logic exactly. Tests marked needs_nli use the real models.
"""

from __future__ import annotations

import pytest

import criterion.divergence as dv
from criterion import CapabilityError, grounding, verify
from criterion.capture import _split_sentences
from tests.conftest import needs_nli


def stub_scorer(monkeypatch, score_for):
    """Replace the grounding scorer with `score_for(claim) -> float`."""
    monkeypatch.setattr(dv, "_USE_SCORER", True)
    monkeypatch.setattr(dv, "_USE_NLI", False)
    monkeypatch.setattr(
        dv, "_scorer_support",
        lambda source, claims: [(score_for(c), "") for c in claims],
    )


def check(source, output, mode="rag", **kw):
    return verify(source, output, mode=mode, record=False, **kw)


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("The drugs were administered by Dr. Conrad Murray. He denies it.",
     ["The drugs were administered by Dr. Conrad Murray.", "He denies it."]),
    ("J. K. Rowling visited the U.S. Army base. It went well.",
     ["J. K. Rowling visited the U.S. Army base.", "It went well."]),
    ("Revenue grew 3.5% vs. 2.1% last year. See Fig. 3 for detail.",
     ["Revenue grew 3.5% vs. 2.1% last year.", "See Fig. 3 for detail."]),
    ("the trial ended . \nthe jury left .", ["the trial ended .", "the jury left ."]),
    ("1. First item. 2. Second item.", ["1. First item.", "2. Second item."]),
    ("It works! Does it? Yes.", ["It works!", "Does it?", "Yes."]),
])
def test_sentence_splitting(text, expected):
    assert [s.strip() for s in _split_sentences(text)] == expected


# --------------------------------------------------------------------------
# Arbitration: the scorer decides support; pairwise checks explain
# --------------------------------------------------------------------------

DOCS = (
    "The company reported revenue of 60 million dollars in 2023. "
    "Its workforce grew to 900 people. The chief executive did not resign."
)


def test_unsupported_sentence_blocks_and_is_explained(monkeypatch):
    stub_scorer(monkeypatch, lambda c: 0.03)
    r = check(DOCS, "The regulator fined the company last spring.")
    assert not r.is_trustworthy
    assert [d.code for d in r.divergences if d.severity.value == "critical"] == ["ungrounded"]
    assert r.groundedness == pytest.approx(0.03)
    assert r.capabilities["scorer"] is True and r.capabilities["model_checked"] is True


def test_supported_sentence_passes(monkeypatch):
    stub_scorer(monkeypatch, lambda c: 0.97)
    r = check(DOCS, "Headcount rose to nine hundred employees.")
    assert r.is_trustworthy and r.critical_count == 0
    assert r.groundedness == pytest.approx(0.97)


def test_loose_pairwise_finding_does_not_overrule_the_scorer(monkeypatch):
    # The output resembles the revenue sentence but is really about the
    # workforce figure. v0.4.0 blamed it for "changing" 60 to 900.
    stub_scorer(monkeypatch, lambda c: 0.95)
    r = check(DOCS, "The company reported a workforce of 900 people in 2023.")
    assert r.is_trustworthy, r.to_dict()
    notes = [d for d in r.divergences if d.code.startswith("numeric")]
    assert all(d.severity.value == "minor" for d in notes)


def test_near_verbatim_change_blocks_even_if_the_scorer_is_fooled(monkeypatch):
    stub_scorer(monkeypatch, lambda c: 0.99)
    r = check(DOCS, "The company reported revenue of 90 million dollars in 2023.")
    assert not r.is_trustworthy, r.to_dict()
    hit = [d for d in r.divergences if d.code == "numeric_change"]
    assert hit and hit[0].severity.value == "critical" and hit[0].alignment >= 0.9
    r = check(DOCS, "The chief executive did resign.")
    assert not r.is_trustworthy, r.to_dict()


def test_fabrication_threshold_can_let_weak_support_through(monkeypatch):
    stub_scorer(monkeypatch, lambda c: 0.35)
    out = "The regulator fined the company last spring."
    assert not check(DOCS, out).is_trustworthy                       # default: blocks
    lenient = check(DOCS, out, thresholds={"fabrication": 0.2})
    assert lenient.critical_count == 0 and lenient.major_count == 1  # reported, not blocking


def test_every_output_sentence_is_judged(monkeypatch):
    stub_scorer(monkeypatch, lambda c: 0.02 if "fined" in c else 0.98)
    r = check(DOCS, "Its workforce grew to 900 people. The regulator fined the company.")
    assert not r.is_trustworthy
    assert len(r.sentence_support) == 2
    assert r.groundedness == pytest.approx(0.02)


def test_scorer_failure_fails_closed(monkeypatch):
    def boom(source, claims):
        raise RuntimeError("weights missing")
    monkeypatch.setattr(dv, "_USE_SCORER", True)
    monkeypatch.setattr(dv, "_scorer_support", boom)
    monkeypatch.delenv("CRITERION_REQUIRE_NLI", raising=False)
    with pytest.raises(CapabilityError, match="weights missing"):
        check(DOCS, "Its workforce grew to 900 people.")
    r = check(DOCS, "Its workforce grew to 900 people.", require_nli=False)
    assert r.capabilities["scorer"] is False and not r.is_trustworthy
    assert "weights missing" in r.capabilities["scorer_error"]


def test_an_nli_model_alone_is_not_a_model_checked_verdict(monkeypatch):
    """Falling back from the grounding scorer to plain NLI would be a silent
    downgrade of several accuracy points; it must not count as checked."""
    monkeypatch.setattr(dv, "_USE_SCORER", False)
    monkeypatch.delenv("CRITERION_REQUIRE_NLI", raising=False)
    with pytest.raises(CapabilityError):
        check(DOCS, "Its workforce grew to 900 people.")


# --------------------------------------------------------------------------
# Model provenance
# --------------------------------------------------------------------------

def test_scorers_are_pinned_to_full_commit_hashes():
    for name in ("minicheck", "factcg"):
        spec = grounding._SPECS[name]
        assert len(spec["revision"]) == 40 and int(spec["revision"], 16) >= 0
    assert grounding.scorer_info()["scorer"] == grounding.SCORER_NAME


@needs_nli
def test_long_unpunctuated_source_is_chunked_not_truncated():
    words = " ".join(f"w{i}" for i in range(3000))
    chunks = grounding.build_chunks(words)
    assert len(chunks) > 3
    assert " ".join(chunks).split() == words.split()          # nothing lost
    assert all(grounding._count(c) <= 400 for c in chunks)


@needs_nli
def test_real_scorer_separates_supported_from_fabricated():
    src = ("The trial enrolled 400 adults across six hospitals. "
           "Blood pressure fell by an average of 9 points in the treatment arm.")
    (good, _), (bad, _) = grounding.support(src, [
        "Blood pressure dropped by about 9 points among treated patients.",
        "The regulator approved the drug for children last year.",
    ])
    assert good > 0.5 > bad
