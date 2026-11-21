"""Tests for the reviewer workflow and the record changes behind it (v0.7.0).

The scorer is stubbed, so these run without any model.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import criterion.divergence as dv
from criterion import grounding, review, verify
from criterion.__main__ import main
from criterion.record import read_record, verify_chain

DATA = Path(__file__).parent / "data"
SOURCE = ("The company reported revenue of 12 million dollars. Its workforce grew to 900 people. "
          "The chief executive praised the sales team.")
# One sentence per label: the stub votes by keyword.
OUTPUT = ("Revenue was 12 million dollars. The workforce reached 900 people. "
          "The regulator fined the company. The chief executive resigned.")
VOTES = {
    "Revenue": (0.97, 0.93),       # confident, supported
    "workforce": (0.95, 0.62),     # probable, supported
    "regulator": (0.90, 0.30),     # unsure (average 0.60: supported)
    "resigned": (0.04, 0.06),      # confident, unsupported
}


@pytest.fixture
def two_models(monkeypatch):
    monkeypatch.setattr(dv, "_USE_SCORER", True)
    monkeypatch.setattr(dv, "_USE_NLI", False)

    def fake(source, claims):
        out = []
        for c in claims:
            a, b = next(v for k, v in VOTES.items() if k in c)
            out.append(grounding.Support((a + b) / 2, f"passage for: {c}", {"minicheck": a, "factcg": b}))
        return out
    monkeypatch.setattr(dv, "_scorer_support", fake)


@pytest.fixture
def ledger(tmp_path, two_models):
    p = tmp_path / "record.jsonl"
    verify(SOURCE, OUTPUT, mode="rag", record_path=p)
    return p


def lines(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def rewrite(p, rows):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


# --------------------------------------------------------------------------
# Record: old ledgers keep verifying; new fields are protected
# --------------------------------------------------------------------------

def test_ledger_written_by_060_still_verifies_and_extends(tmp_path, two_models):
    p = tmp_path / "old.jsonl"
    shutil.copy(DATA / "ledger_scheme2.jsonl", p)
    before = p.read_bytes()
    ok, msg = verify_chain(p)
    assert ok, msg
    verify(SOURCE, OUTPUT, mode="rag", record_path=p)
    assert p.read_bytes().startswith(before)                 # old entries untouched
    ok, msg = verify_chain(p)
    assert ok, msg
    assert [e.hash_scheme for e in read_record(p)] == [2, 2, 3]


def test_unprotected_fields_cannot_ride_on_an_old_entry(tmp_path):
    p = tmp_path / "old.jsonl"
    rows = lines(DATA / "ledger_scheme2.jsonl")
    rows[0]["sentences"] = [{"index": 0, "text_hash": "0" * 64, "support": 0.99}]
    rewrite(p, rows)
    ok, msg = verify_chain(p)
    assert not ok and "does not protect" in msg


def test_verdict_records_each_sentence_with_its_evidence(ledger):
    (e,) = read_record(ledger)
    assert e.kind == "verdict" and e.hash_scheme == 3 and len(e.sentences) == 4
    row, text = e.sentences[2], e.sentence_texts[2]
    assert row["models"] == {"minicheck": 0.9, "factcg": 0.3} and row["confidence"] == "unsure"
    assert text["text"] == "The regulator fined the company."
    assert text["evidence"].startswith("passage for:")
    assert verify_chain(ledger)[0]


@pytest.mark.parametrize("edit", [
    lambda r: r["sentences"][2].__setitem__("support", 0.99),
    lambda r: r["sentences"][2].__setitem__("confidence", "confident"),
    lambda r: r["sentences"][2]["models"].__setitem__("factcg", 0.9),
    lambda r: r["sentence_texts"][2].__setitem__("text", "The regulator praised the company."),
    lambda r: r["sentence_texts"][2].__setitem__("evidence", "The regulator fined the company."),
    lambda r: r["sentence_texts"].pop(),
    lambda r: r.__setitem__("kind", "review"),
])
def test_tampering_with_sentence_rows_breaks_the_chain(ledger, edit):
    rows = lines(ledger)
    edit(rows[0])
    rewrite(ledger, rows)
    assert not verify_chain(ledger)[0]


def test_text_can_be_withheld_or_redacted_later(tmp_path, two_models):
    p = tmp_path / "r.jsonl"
    verify(SOURCE, OUTPUT, mode="rag", record_path=p, store_text=False)
    (e,) = read_record(p)
    assert e.sentence_texts is None and len(e.sentences[0]["text_hash"]) == 64
    assert verify_chain(p)[0]
    (item,) = [i for i in review.queue(p) if i.confidence == "unsure"]
    assert item.text is None and item.support == pytest.approx(0.6)

    q = tmp_path / "full.jsonl"
    verify(SOURCE, OUTPUT, mode="rag", record_path=q)
    rows = lines(q)
    rows[0]["sentence_texts"] = None                          # redact after the fact
    rows[0]["source_text"] = rows[0]["output_text"] = None
    rewrite(q, rows)
    assert verify_chain(q)[0]


# --------------------------------------------------------------------------
# Queue
# --------------------------------------------------------------------------

def test_queue_runs_from_least_to_most_reliable(ledger):
    items = review.queue(ledger, spot_check=0)
    assert [(i.confidence, i.machine) for i in items] == [
        ("unsure", "supported"), ("probable", "supported"), ("confident", "unsupported"),
    ]
    assert items[0].reason == "the two models disagree"
    assert items[0].evidence == "passage for: The regulator fined the company."
    assert items[2].reason == "the machine found it unsupported"


def test_confident_supported_is_left_out_unless_sampled(ledger):
    assert all(i.sentence != 0 for i in review.queue(ledger, spot_check=0))
    (extra,) = [i for i in review.queue(ledger, spot_check=1) if i.sentence == 0]
    assert extra.reason == "spot check of a confident verdict"
    assert len(review.queue(ledger, everything=True)) == 4


def test_spot_check_is_deterministic_and_roughly_the_share_asked(tmp_path, two_models):
    p = tmp_path / "r.jsonl"
    for k in range(200):
        verify(SOURCE + f" Note {k}.", "Revenue was 12 million dollars.", mode="rag", record_path=p)
    first = [i.entry_id for i in review.queue(p, spot_check=0.25)]
    assert first == [i.entry_id for i in review.queue(p, spot_check=0.25)]
    assert 30 <= len(first) <= 70
    with pytest.raises(ValueError):
        review.queue(p, spot_check=1.5)


def test_unverified_output_is_queued_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(dv, "_USE_SCORER", False)
    monkeypatch.setattr(dv, "_USE_NLI", False)
    p = tmp_path / "r.jsonl"
    verify(SOURCE, "Revenue was 12 million dollars.", mode="rag", record_path=p, require_nli=False)
    (item,) = review.queue(p)
    assert item.sentence is None and item.machine == "unsupported" and item.confidence is None
    review.decide(item.entry_id, "supported", "Aisha", record_path=p)
    assert review.queue(p) == [] and verify_chain(p)[0]


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------

def test_a_decision_is_appended_and_the_verdict_is_untouched(ledger):
    before = ledger.read_bytes()
    (verdict,) = read_record(ledger)
    entry = review.decide(verdict.entry_id[:10], "unsupported", "Aisha", sentence=2,
                          note="No fine is mentioned.", record_path=ledger)
    assert ledger.read_bytes().startswith(before)
    assert entry.kind == "review" and entry.predecessor_hash == verdict.entry_id
    assert entry.review == {"of": verdict.entry_id, "sentence": 2, "decision": "unsupported",
                            "reviewer": "Aisha", "note": "No fine is mentioned."}
    assert (entry.source_hash, entry.output_hash) == (verdict.source_hash, verdict.output_hash)
    ok, msg = verify_chain(ledger)
    assert ok and "2 entries" in msg
    assert all(i.sentence != 2 for i in review.queue(ledger))
    assert any(i.sentence == 2 for i in review.queue(ledger, include_reviewed=True))


@pytest.mark.parametrize("kwargs,match", [
    (dict(decision="maybe", reviewer="A", sentence=2), "decision must be"),
    (dict(decision="supported", reviewer="  ", sentence=2), "reviewer name"),
    (dict(decision="supported", reviewer="A", sentence=9), "not one of the 4"),
    (dict(decision="supported", reviewer="A", sentence=-1), "not one of the 4"),
    (dict(decision="supported", reviewer="A", sentence=True), "not one of the 4"),
    (dict(decision="supported", reviewer="A"), "say which one"),
])
def test_bad_decisions_are_refused_and_nothing_is_written(ledger, kwargs, match):
    before = ledger.read_bytes()
    (verdict,) = read_record(ledger)
    decision = kwargs.pop("decision"); reviewer = kwargs.pop("reviewer")
    with pytest.raises(ValueError, match=match):
        review.decide(verdict.entry_id, decision, reviewer, record_path=ledger, **kwargs)
    assert ledger.read_bytes() == before


def test_entry_lookup_needs_an_unambiguous_prefix(ledger):
    (verdict,) = read_record(ledger)
    with pytest.raises(ValueError, match="at least the first 8"):
        review.decide(verdict.entry_id[:4], "supported", "A", sentence=0, record_path=ledger)
    with pytest.raises(ValueError, match="no entry"):
        review.decide("f" * 12 if not verdict.entry_id.startswith("f" * 12) else "0" * 12,
                      "supported", "A", sentence=0, record_path=ledger)


def test_a_review_cannot_be_reviewed(ledger):
    (verdict,) = read_record(ledger)
    r = review.decide(verdict.entry_id, "supported", "A", sentence=0, record_path=ledger)
    with pytest.raises(ValueError, match="only a verdict"):
        review.decide(r.entry_id, "supported", "B", record_path=ledger)


@pytest.mark.parametrize("edit,match", [
    (lambda r: r["review"].__setitem__("decision", "supported"), "content hash mismatch"),
    (lambda r: r["review"].__setitem__("reviewer", "Mallory"), "content hash mismatch"),
    (lambda r: r["review"].__setitem__("sentence", 1), "content hash mismatch"),
])
def test_editing_a_decision_breaks_the_chain(ledger, edit, match):
    (verdict,) = read_record(ledger)
    review.decide(verdict.entry_id, "unsupported", "Aisha", sentence=2, record_path=ledger)
    rows = lines(ledger)
    edit(rows[1])
    rewrite(ledger, rows)
    ok, msg = verify_chain(ledger)
    assert not ok and match in msg


def test_a_review_must_point_at_an_earlier_verdict_of_the_same_text(ledger, tmp_path):
    """A correctly hashed review is still rejected if what it points at is wrong."""
    from criterion import record as rec
    (verdict,) = read_record(ledger)
    review.decide(verdict.entry_id, "unsupported", "Aisha", sentence=2, record_path=ledger)

    def forged(change):
        rows = lines(ledger)
        change(rows[1])
        rows[1]["entry_id"] = rec._entry_id_v3(rows[1])
        p = tmp_path / "forged.jsonl"
        rewrite(p, rows)
        return verify_chain(p)

    assert "earlier verdict" in forged(lambda r: r["review"].__setitem__("of", "0" * 64))[1]
    assert "different text" in forged(lambda r: r.__setitem__("output_hash", "0" * 64))[1]
    assert "does not have" in forged(lambda r: r["review"].__setitem__("sentence", 40))[1]
    assert "malformed" in forged(lambda r: r["review"].__setitem__("decision", "fine"))[1]
    assert "unknown entry kind" in forged(lambda r: r.__setitem__("kind", "note"))[1]


def test_keyed_record_macs_the_decision(ledger, tmp_path, two_models, monkeypatch):
    monkeypatch.setenv("CRITERION_RECORD_KEY", "k1")
    p = tmp_path / "keyed.jsonl"
    verify(SOURCE, OUTPUT, mode="rag", record_path=p)
    (verdict,) = read_record(p)
    entry = review.decide(verdict.entry_id, "unsupported", "Aisha", sentence=2, record_path=p)
    assert entry.mac and verify_chain(p)[0]
    assert not verify_chain(p, key="other")[0]


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------

def test_stats_compare_machine_and_reviewer_by_label(ledger):
    (verdict,) = read_record(ledger)
    eid = verdict.entry_id
    review.decide(eid, "supported", "Aisha", sentence=2, record_path=ledger)      # first thought
    review.decide(eid, "unsupported", "Aisha", sentence=2, record_path=ledger)    # corrected: later wins
    review.decide(eid, "supported", "Bilal", sentence=1, record_path=ledger)
    review.decide(eid, "unclear", "Bilal", sentence=3, record_path=ledger)
    s = review.stats(ledger)
    assert s["sentences"] == 4 and s["reviewed"] == 3 and s["reviewers"] == ["Aisha", "Bilal"]
    unsure, probable, confident = (s["by_confidence"][k] for k in ("unsure", "probable", "confident"))
    assert (unsure["reviewed"], unsure["disagreed"], unsure["agreement"]) == (1, 1, 0.0)
    assert unsure["machine_supported_human_unsupported"] == 1
    assert (probable["agreed"], probable["agreement"]) == (1, 1.0)
    assert (confident["sentences"], confident["unclear"], confident["agreement"]) == (2, 1, None)
    assert len(read_record(ledger)) == 5 and verify_chain(ledger)[0]              # nothing overwritten


def test_stats_on_an_empty_or_unreviewed_record(ledger, tmp_path):
    s = review.stats(ledger)
    assert s["reviewed"] == 0 and s["by_confidence"]["unsure"]["agreement"] is None
    assert review.stats(tmp_path / "missing.jsonl")["sentences"] == 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_queue_decide_stats(ledger, capsys):
    assert main(["review", "queue", "--record", str(ledger), "--json", "--spot-check", "0"]) == 0
    rows = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert [r["confidence"] for r in rows] == ["unsure", "probable", "confident"]

    assert main(["review", "queue", "--record", str(ledger), "--limit", "1"]) == 0
    out = capsys.readouterr().out
    assert "The regulator fined the company." in out and "minicheck 0.90" in out and "1 waiting" in out

    eid = rows[0]["entry_id"][:12]
    assert main(["review", "decide", eid, "unsupported", "--sentence", "2",
                 "--reviewer", "Aisha", "--record", str(ledger)]) == 0
    assert main(["review", "decide", eid, "unsupported", "--sentence", "99",
                 "--reviewer", "Aisha", "--record", str(ledger)]) == 1
    capsys.readouterr()
    assert main(["review", "stats", "--record", str(ledger), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["reviewed"] == 1
    assert main(["review", "stats", "--record", str(ledger)]) == 0
    assert "unsure" in capsys.readouterr().out
    assert main(["chain", "--record", str(ledger)]) == 0


def test_cli_walk_records_answers_until_quit(ledger, monkeypatch, capsys):
    answers = iter(["x", "u", "no fine in the source", "n", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main(["review", "walk", "--reviewer", "Aisha", "--record", str(ledger), "--spot-check", "0"]) == 0
    assert "1 decision recorded" in capsys.readouterr().out
    entries = read_record(ledger)
    assert len(entries) == 2 and entries[1].review["note"] == "no fine in the source"
    assert entries[1].review["sentence"] == 2
    assert verify_chain(ledger)[0]


def test_cli_walk_stops_cleanly_when_input_ends(ledger, monkeypatch):
    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    assert main(["review", "walk", "--reviewer", "Aisha", "--record", str(ledger)]) == 0
    assert len(read_record(ledger)) == 1
