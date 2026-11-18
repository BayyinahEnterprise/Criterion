"""Tests for the grounding scorer integration (v0.5.0, two-model verdict v0.6.0).

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


# --------------------------------------------------------------------------
# Two models: averaged score, and a third verdict when they disagree (v0.6.0)
# --------------------------------------------------------------------------

def stub_two(monkeypatch, votes_for):
    """Stub a two-model scorer: `votes_for(claim) -> (minicheck, factcg)`."""
    monkeypatch.setattr(dv, "_USE_SCORER", True)
    monkeypatch.setattr(dv, "_USE_NLI", False)

    def fake(source, claims):
        out = []
        for c in claims:
            a, b = votes_for(c)
            out.append(grounding.Support((a + b) / 2, "", {"minicheck": a, "factcg": b}))
        return out
    monkeypatch.setattr(dv, "_scorer_support", fake)


@pytest.mark.parametrize("votes,expected", [
    ({"a": 0.95, "b": 0.90}, "confident"),
    ({"a": 0.05, "b": 0.10}, "confident"),
    ({"a": 0.95, "b": 0.60}, "probable"),
    ({"a": 0.45, "b": 0.05}, "probable"),
    ({"a": 0.95, "b": 0.40}, "unsure"),
    ({"a": 0.49, "b": 0.50}, "unsure"),
    ({"a": 0.95}, None),
    ({}, None),
])
def test_agreement_labels(votes, expected):
    assert dv.agreement(votes, 0.5, 0.3) == expected


def test_support_is_still_a_pair():
    s = grounding.Support(0.7, "chunk", {"minicheck": 0.9, "factcg": 0.5})
    score, chunk = s
    assert (score, chunk) == (0.7, "chunk") and s.members["factcg"] == 0.5


def test_models_agreeing_clearly_is_confident(monkeypatch):
    stub_two(monkeypatch, lambda c: (0.97, 0.93))
    r = check(DOCS, "Its workforce grew to 900 people.")
    assert r.is_trustworthy and r.confidence == "confident" and r.unsure_count == 0
    row = r.sentence_support[0]
    assert row["models"] == {"minicheck": 0.97, "factcg": 0.93}
    assert row["support"] == pytest.approx(0.95) and row["confidence"] == "confident"


def test_models_disagreeing_is_unsure_and_says_so(monkeypatch):
    # Average 0.65 passes; one model alone would have rejected it.
    stub_two(monkeypatch, lambda c: (0.95, 0.35) if "fined" in c else (0.97, 0.93))
    r = check(DOCS, "Its workforce grew to 900 people. The regulator fined the company.")
    assert r.is_trustworthy                      # the average decides the verdict
    assert r.confidence == "unsure" and r.unsure_count == 1
    assert "Unsure" in r.summary
    assert r.to_dict()["confidence"] == "unsure"
    assert [x["confidence"] for x in r.sentence_support] == ["confident", "unsure"]


def test_unsure_never_rescues_a_rejected_sentence(monkeypatch):
    stub_two(monkeypatch, lambda c: (0.60, 0.10))     # average 0.35: unsupported
    r = check(DOCS, "The regulator fined the company.")
    assert not r.is_trustworthy and r.confidence == "unsure"


def test_weakest_label_wins_across_sentences(monkeypatch):
    stub_two(monkeypatch, lambda c: (0.97, 0.60) if "fined" in c else (0.97, 0.93))
    r = check(DOCS, "Its workforce grew to 900 people. The regulator fined the company.")
    assert r.confidence == "probable" and r.unsure_count == 0


def test_one_model_gives_no_confidence_label(monkeypatch):
    stub_scorer(monkeypatch, lambda c: 0.98)
    r = check(DOCS, "Its workforce grew to 900 people.")
    assert r.is_trustworthy and r.confidence is None
    assert "models" not in r.sentence_support[0]


def test_confident_margin_is_a_threshold(monkeypatch):
    stub_two(monkeypatch, lambda c: (0.97, 0.70))
    assert check(DOCS, "Its workforce grew to 900 people.").confidence == "probable"
    r = check(DOCS, "Its workforce grew to 900 people.", thresholds={"confident": 0.1})
    assert r.confidence == "confident" and r.thresholds["confident"] == 0.1


@pytest.mark.parametrize("name,expected", [
    ("ensemble", ("minicheck", "factcg")),
    ("minicheck", ("minicheck",)),
    ("factcg+minicheck", ("factcg", "minicheck")),
])
def test_member_names(monkeypatch, name, expected):
    monkeypatch.setattr(grounding, "SCORER_NAME", name)
    assert grounding.member_names() == expected


@pytest.mark.parametrize("name", ["", "bogus", "minicheck+bogus", "minicheck+minicheck"])
def test_bad_scorer_names_are_rejected(monkeypatch, name):
    monkeypatch.setattr(grounding, "SCORER_NAME", name)
    with pytest.raises(ValueError):
        grounding.member_names()


def test_ensemble_loads_all_or_nothing(monkeypatch):
    """One member failing must fail the scorer, not shrink it silently."""
    class Fake:
        def __init__(self, name):
            if name == "factcg":
                raise OSError("no weights for factcg")
            self.name = name
    monkeypatch.setattr(grounding, "SCORER_NAME", "ensemble")
    monkeypatch.setattr(grounding, "_Scorer", Fake)
    monkeypatch.setattr(grounding, "_scorers", None)
    monkeypatch.setattr(grounding, "_load_failed", False)
    monkeypatch.setattr(grounding, "_load_error", None)
    with pytest.raises(OSError):
        grounding._load()
    assert grounding._scorers is None
    with pytest.raises(RuntimeError, match="factcg"):
        grounding._load()


def test_cli_exit_code_for_unsure(monkeypatch, tmp_path):
    from criterion.__main__ import main
    stub_two(monkeypatch, lambda c: (0.95, 0.35))
    src = tmp_path / "s.txt"; src.write_text(DOCS)
    out = tmp_path / "o.txt"; out.write_text("Its workforce grew to 900 people.")
    base = ["verify", str(src), str(out), "--mode", "rag", "--no-record", "--quiet"]
    assert main(base) == 0
    assert main(base + ["--fail-on-unsure"]) == 3


@needs_nli
def test_default_scorer_is_two_models_and_reports_both():
    if grounding.SCORER_NAME != "ensemble":
        pytest.skip("a single scorer was selected with CRITERION_SCORER")
    src = ("The trial enrolled 400 adults across six hospitals. "
           "Blood pressure fell by an average of 9 points in the treatment arm.")
    good, bad = grounding.support(src, [
        "Blood pressure dropped by about 9 points among treated patients.",
        "The regulator approved the drug for children last year.",
    ])
    assert set(good.members) == {"minicheck", "factcg"}
    assert good[0] == pytest.approx(sum(good.members.values()) / 2)
    assert good[0] > 0.5 > bad[0]
    assert grounding.scorer_info()["combine"] == "mean"


def test_long_inputs_are_batched_small():
    """Eight 2,048-token inputs in one batch exhausted memory; cap by size."""
    sc = object.__new__(grounding._Scorer)
    sc.name, sc.spec = "factcg", grounding._SPECS["factcg"]
    long_, short = "w " * 4000, "w " * 20
    chunks = [short] * 10 + [long_] * 5
    claims = ["c"] * 15
    order = sorted(range(15), key=lambda i: len(chunks[i]))
    batches = list(sc._batches(order, chunks, claims))
    assert sorted(i for b in batches for i in b) == list(range(15))     # nothing dropped
    assert all(len(b) <= grounding.BATCH_SIZE for b in batches)
    assert all(len(b) == 1 for b in batches if any(chunks[i] is long_ for i in b))
    assert len(batches[0]) == grounding.BATCH_SIZE
