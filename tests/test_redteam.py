"""Red-team regression tests (v0.4.0).

Each test pins one attack that succeeded against v0.3.1. Tests that need
the NLI model are skipped when it is unavailable, and say so.
"""

from __future__ import annotations

import json
import threading

import pytest

import criterion
from criterion import verify, CapabilityError
from criterion.__main__ import main
from criterion.capture import capture
from criterion.divergence import _align_atoms, _extract_numbers, find_divergences
from criterion.record import (
    RecordCorruptError, chain_head, read_record, verify_chain,
)


def _nli_available() -> bool:
    try:
        from criterion.nli import classify_pair
        classify_pair("A cat sat.", "A cat sat.")
        return True
    except Exception:
        return False


needs_nli = pytest.mark.skipif(not _nli_available(), reason="NLI model not available")

DOCS = (
    "The trial enrolled 400 adults across six hospitals. Participants received either "
    "the drug or a placebo for twelve weeks. The drug is not safe for children under twelve. "
    "Blood pressure fell by an average of 9 points in the treatment arm. Side effects were "
    "mild and resolved without intervention. The sponsor funded the study but had no role "
    "in the analysis. "
)


def check(source, output, mode="strict", **kw):
    return verify(source, output, mode=mode, record=False, **kw)


# --------------------------------------------------------------------------
# Mode integrity: the output's length must not relax an explicit mode
# --------------------------------------------------------------------------

def test_strict_mode_is_not_relaxed_by_a_short_output():
    r = check(DOCS, "The trial enrolled 400 adults across six hospitals.", mode="strict")
    assert not r.is_trustworthy, r.to_dict()
    assert any(d.code == "omission" and d.severity.value == "critical" for d in r.divergences)


def test_find_divergences_rejects_unknown_mode():
    with pytest.raises(ValueError):
        find_divergences(capture("A.", "A."), mode="lenient")


# --------------------------------------------------------------------------
# Grounding: output that alignment cannot pair is still checked
# --------------------------------------------------------------------------

@needs_nli
@pytest.mark.parametrize("mode", ["rag", "summary", "strict"])
def test_paraphrased_contradiction_is_caught_in_every_mode(mode):
    r = check(DOCS, "Doctors can freely give this medicine to young kids.", mode=mode)
    assert not r.is_trustworthy, r.to_dict()
    assert r.critical_count >= 1


@needs_nli
def test_rag_single_fabricated_sentence_is_not_faithful():
    r = check(
        DOCS,
        "The trial enrolled 400 adults across six hospitals. "
        "The FDA approved the drug last year for all age groups.",
        mode="rag",
    )
    assert not r.is_trustworthy, r.to_dict()
    assert any(d.code in ("ungrounded", "contradicts_source") for d in r.divergences)


@needs_nli
def test_fabricated_clause_appended_to_a_true_sentence():
    r = check(
        "The drug reduced blood pressure in the trial.",
        "The drug reduced blood pressure in the trial and also cures cancer.",
    )
    assert not r.is_trustworthy, r.to_dict()


@needs_nli
def test_run_on_clause_hiding_a_new_claim():
    r = check(
        "The bridge is safe. Inspectors found no cracks.",
        "The bridge is safe. Inspectors found no cracks; however the bridge will collapse soon",
    )
    assert not r.is_trustworthy, r.to_dict()


@needs_nli
def test_faithful_rag_answer_and_summary_stay_trustworthy():
    rag = check(DOCS, "Blood pressure dropped by about 9 points on average in the group that took the drug.", mode="rag")
    assert rag.is_trustworthy, rag.to_dict()
    summ = check(
        DOCS,
        "A 400-adult, twelve-week trial found the drug lowered blood pressure by 9 points "
        "with mild side effects; it is not safe for children under twelve.",
        mode="summary",
    )
    assert summ.is_trustworthy, summ.to_dict()


@needs_nli
def test_fragment_answers_are_not_sent_to_nli():
    docs = "Todd Phillips directed the film. It was shot in Nevada over six weeks."
    assert check(docs, "Todd Phillips", mode="rag").is_trustworthy
    assert check(docs, "Nevada", mode="rag").critical_count == 0
    r = check(docs, "Martin Scorsese", mode="rag")
    assert not r.is_trustworthy
    assert any(d.code == "fragment_not_in_source" for d in r.divergences)


# --------------------------------------------------------------------------
# False positives that made strict mode cry wolf
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source,output", [
    ("The device isn't certified.", "The device is not certified."),
    ("Users can't delete the log.", "Users cannot delete the log."),
    ("The device isn’t certified.", "The device isn't certified."),
    ("Sales rose 10% in May.", "Sales rose 10 percent in May."),
])
def test_equivalent_spellings_are_not_divergences(source, output):
    r = check(source, output)
    assert r.is_trustworthy and r.critical_count == 0, r.to_dict()


@needs_nli
def test_double_negation_equivalence_is_not_flagged():
    r = check("The claim is not untrue.", "The claim is true.")
    assert r.is_trustworthy, r.to_dict()


def test_number_canonicalisation():
    assert _extract_numbers("1,500,000 residents") == _extract_numbers("1.5 million residents")
    assert _extract_numbers("$5M") == _extract_numbers("5 million dollars") == {"5000000"}
    assert _extract_numbers("10%") == _extract_numbers("10 percent") == {"10%"}
    assert _extract_numbers("2.50") == _extract_numbers("2.5")
    assert _extract_numbers("between 5-10 days") == {"5", "10"}
    assert _extract_numbers("growth of -5%") != _extract_numbers("growth of 5%")
    assert _extract_numbers("$1,000") != _extract_numbers("$2,000")


def test_sign_flip_is_a_numeric_change():
    r = check("Margins moved by -5% this quarter.", "Margins moved by 5% this quarter.")
    assert not r.is_trustworthy
    assert any(d.code == "numeric_change" for d in r.divergences)


# --------------------------------------------------------------------------
# Hidden content
# --------------------------------------------------------------------------

def test_zero_width_character_cannot_hide_a_changed_number():
    r = check("Revenue grew 60% last year.", "Revenue grew 9​0% last year.")
    codes = {d.code for d in r.divergences}
    assert "hidden_chars" in codes, r.to_dict()
    assert codes & {"numeric_change", "nli_contradiction"}, r.to_dict()
    assert not r.is_trustworthy


def test_bidi_override_and_homoglyph_are_reported():
    r = check("Pay the vendor today.", "Pay the vendor today.‮")
    assert any(d.code == "hidden_chars" for d in r.divergences)
    r = check("Send it to paypal support.", "Send it to pаypal support.")  # Cyrillic a
    assert any(d.code == "hidden_chars" for d in r.divergences)
    assert not r.is_trustworthy


def test_emoji_joiners_are_not_hidden_content():
    r = check("The family arrived.", "The family arrived. \U0001F468‍\U0001F469‍\U0001F467")
    assert not any(d.code == "hidden_chars" for d in r.divergences)


# --------------------------------------------------------------------------
# Capability honesty
# --------------------------------------------------------------------------

def test_result_reports_which_detectors_ran():
    r = check("The sky is blue.", "The sky is blue.")
    assert set(r.capabilities) >= {"nli", "semantic"}
    assert "capabilities" in r.to_dict()


def test_require_nli_fails_closed(monkeypatch):
    import criterion.divergence as dv
    monkeypatch.setattr(dv, "_USE_NLI", False)
    r = check("The sky is blue.", "The sky is blue.")
    assert r.capabilities["nli"] is False
    assert "NLI did not run" in r.summary
    with pytest.raises(CapabilityError):
        check("The sky is blue.", "The sky is blue.", require_nli=True)
    monkeypatch.setenv("CRITERION_REQUIRE_NLI", "1")
    with pytest.raises(CapabilityError):
        check("The sky is blue.", "The sky is blue.")
    assert main(["verify", "The sky is blue.", "The sky is blue.", "--no-record", "--require-nli", "-q"]) == 1


def test_nli_runtime_failure_is_reported_not_swallowed(monkeypatch):
    import criterion.divergence as dv

    def boom(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(dv, "_USE_NLI", True)
    monkeypatch.setattr(dv, "_nli_probs", boom, raising=False)
    r = check("The sky is blue.", "The sky is green.")
    assert r.capabilities["nli"] is False
    assert "model exploded" in r.capabilities["nli_error"]


def test_documented_api_is_exported():
    assert callable(criterion.verify_batch)
    assert criterion.__version__ == criterion._version.__version__


# --------------------------------------------------------------------------
# Record integrity
# --------------------------------------------------------------------------

def _ledger(tmp_path, n=3, **kw):
    p = tmp_path / "ledger.jsonl"
    verify("The sky is blue.", "The sky is blue.", mode="strict", record_path=p, **kw)
    if n > 1:
        verify(
            "The vaccine is safe. Clinical trials with 40,000 participants showed no serious adverse effects.",
            "The vaccine is safe.", mode="strict", record_path=p, **kw,
        )
    if n > 2:
        verify("Revenue grew 60%.", "Revenue grew 90%.", mode="strict", record_path=p, **kw)
    return p


def test_field_boundary_shift_is_detected(tmp_path):
    """v0.3.1 hashed fields with no delimiter: moving a digit from the start
    of `summary` onto the end of `broken_bonds` left the hash unchanged."""
    p = _ledger(tmp_path)
    lines = p.read_text(encoding="utf-8").splitlines()
    tampered = False
    for i, line in enumerate(lines):
        e = json.loads(line)
        if e["summary"][:1].isdigit():
            e["broken_bonds"] = int(str(e["broken_bonds"]) + e["summary"][0])
            e["summary"] = e["summary"][1:]
            lines[i] = json.dumps(e, ensure_ascii=False)
            tampered = True
            break
    assert tampered
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, msg = verify_chain(p)
    assert not ok, msg


def test_every_recorded_field_is_integrity_protected(tmp_path):
    p = _ledger(tmp_path, n=1)
    original = json.loads(p.read_text(encoding="utf-8"))
    for field_name, value in [
        ("mode", "summary"), ("detector_config", {"nli": True}),
        ("criterion_version", "9.9.9"), ("trust_level", "unreliable"),
        ("recorded_at", "2020-01-01T00:00:00+00:00"), ("hash_scheme", 1),
    ]:
        e = dict(original)
        e[field_name] = value
        p.write_text(json.dumps(e) + "\n", encoding="utf-8")
        ok, msg = verify_chain(p)
        assert not ok, f"tampering with {field_name} went undetected"


def test_record_states_the_real_version_and_mode(tmp_path):
    p = _ledger(tmp_path, n=1)
    e = json.loads(p.read_text(encoding="utf-8"))
    assert e["criterion_version"] == criterion.__version__
    assert e["mode"] == "strict"
    assert e["detector_config"]["nli"] in (True, False)


def test_tail_truncation_is_detected_with_an_anchor(tmp_path):
    p = _ledger(tmp_path)
    head, count = chain_head(p)
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert verify_chain(p)[0], "an unanchored chain cannot see truncation (documented)"
    assert not verify_chain(p, expected_head=head)[0]
    assert not verify_chain(p, expected_count=count)[0]
    assert main(["chain", "--record", str(p), "--expect-count", str(count)]) == 1


def test_keyed_chain_resists_rewrite(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITERION_RECORD_KEY", "s3cret")
    p = _ledger(tmp_path, n=2)
    assert verify_chain(p)[0]
    assert not verify_chain(p, key="wrong-key")[0]
    # An attacker without the key rewrites the whole ledger with fresh hashes.
    monkeypatch.delenv("CRITERION_RECORD_KEY")
    forged = tmp_path / "forged.jsonl"
    verify("The sky is blue.", "The sky is green.", mode="strict", record_path=forged)
    assert verify_chain(forged)[0]                       # self-consistent...
    assert not verify_chain(forged, key="s3cret")[0]     # ...but not ours


def test_append_after_corrupt_tail_is_refused(tmp_path):
    p = _ledger(tmp_path, n=1)
    with open(p, "a", encoding="utf-8") as f:
        f.write("{garbage\n")
    with pytest.raises(RecordCorruptError):
        verify("A.", "A.", mode="strict", record_path=p)


def test_concurrent_writers_do_not_fork_the_chain(tmp_path):
    p = tmp_path / "ledger.jsonl"
    pair_args = ("The sky is blue.", "The sky is blue.")

    def work():
        for _ in range(8):
            verify(*pair_args, mode="strict", record_path=p)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok, msg = verify_chain(p)
    assert ok, msg
    assert len(read_record(p)) == 48


def test_store_text_false_keeps_hashes_only(tmp_path):
    p = tmp_path / "ledger.jsonl"
    verify("Patient John Doe has a secret.", "John Doe has a secret.", mode="strict",
           record_path=p, store_text=False)
    raw = p.read_text(encoding="utf-8")
    e = json.loads(raw)
    assert e["source_text"] is None and e["output_text"] is None
    assert len(e["source_hash"]) == 64
    assert verify_chain(p)[0]


def test_legacy_v031_entries_still_verify(tmp_path):
    """Ledgers written by <= 0.3.1 (hash scheme 1) must keep verifying."""
    import hashlib
    src, out = "The sky is blue.", "The sky is blue."
    sh, oh = (hashlib.sha256(t.encode()).hexdigest() for t in (src, out))
    fields = [sh, oh, "2026-11-01T00:00:00+00:00", "genesis", "1.0", "faithful", "0",
              "[]", "[]", "0", "ok", "0.2.2"]
    entry = {
        "entry_id": hashlib.sha256("".join(fields).encode()).hexdigest(),
        "recorded_at": fields[2], "predecessor_hash": "genesis",
        "source_hash": sh, "output_hash": oh, "source_text": src, "output_text": out,
        "trust_score": 1.0, "trust_level": "faithful", "total_divergences": 0,
        "divergences": [], "bond_divergences": [], "broken_bonds": 0,
        "summary": "ok", "criterion_version": "0.2.2",
    }
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    assert verify_chain(p)[0]
    verify(src, out, mode="strict", record_path=p)      # new entry chains onto the old one
    ok, msg = verify_chain(p)
    assert ok, msg


# --------------------------------------------------------------------------
# CLI / batch / alignment robustness
# --------------------------------------------------------------------------

def test_cli_accepts_long_literal_text():
    assert main(["verify", "word " * 3000, "word " * 3000, "--no-record", "-q"]) == 0


def test_batch_rejects_non_string_fields(tmp_path):
    f = tmp_path / "pairs.jsonl"
    f.write_text('{"source": ["a"], "output": "b"}\n[1, 2]\n', encoding="utf-8")
    out = tmp_path / "results.jsonl"
    assert main(["batch", str(f), "--no-record", "-o", str(out)]) == 1
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert all("error" in r for r in rows)


def test_verify_rejects_non_string_input():
    with pytest.raises(TypeError):
        verify(["not", "text"], "b", record=False)


def test_alignment_is_best_first_not_first_come():
    pair = capture(
        "Q1 revenue was five million dollars. Q2 revenue was seven million dollars.",
        "Q2 revenue was seven million dollars. Q1 revenue was five million dollars.",
    )
    matched, us, uo = _align_atoms(pair.source.atoms, pair.output.atoms)
    assert not us and not uo
    assert all(s.text == o.text for s, o, _ in matched)


def test_alignment_of_identical_large_documents_is_fast():
    import time
    doc = " ".join(f"Sentence number {i} states fact {i * 7}." for i in range(1500))
    pair = capture(doc, doc)
    t0 = time.time()
    matched, us, uo = _align_atoms(pair.source.atoms, pair.output.atoms)
    assert len(matched) == 1500 and not us and not uo
    assert time.time() - t0 < 5


def test_oversized_fuzzy_alignment_is_refused_not_hung(monkeypatch):
    import criterion.divergence as dv
    monkeypatch.setattr(dv, "MAX_FUZZY_COMPARISONS", 100)
    src = " ".join(f"Alpha item {i} is red." for i in range(30))
    out = " ".join(f"Beta thing {i} was blue." for i in range(30))
    with pytest.raises(dv.InputTooLargeError):
        verify(src, out, mode="strict", record=False)
    assert main(["verify", src, out, "--no-record", "-q"]) == 1


@needs_nli
def test_strict_mode_reports_lost_precision():
    r = check("The contract expires on March 15, 2027.", "The contract expires in early 2027.")
    assert any(d.code == "detail_lost" for d in r.divergences), r.to_dict()
    lenient = check("The contract expires on March 15, 2027.", "The contract expires in early 2027.", mode="rag")
    assert not any(d.code == "detail_lost" for d in lenient.divergences)


def test_role_swaps_are_caught_without_any_model(monkeypatch):
    import criterion.divergence as dv
    monkeypatch.setattr(dv, "_USE_NLI", False)
    monkeypatch.setattr(dv, "_USE_SEMANTIC", False)
    for source, output in [
        ("The dosage is 5 mg and the limit is 10 mg.", "The dosage is 10 mg and the limit is 5 mg."),
        ("The treatment is safe for adults and unsafe for children.",
         "The treatment is unsafe for adults and safe for children."),
        ("Alice is not approved and Bob is approved.", "Alice is approved and Bob is not approved."),
    ]:
        r = check(source, output)
        assert any(d.code == "role_swap" for d in r.divergences), r.to_dict()
        assert not r.is_trustworthy
    benign = check("Alice and Bob are not approved.", "Bob and Alice are not approved.")
    assert not any(d.code == "role_swap" for d in benign.divergences)
