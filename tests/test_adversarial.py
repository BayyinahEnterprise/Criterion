"""Independent adversarial regression tests for Criterion v0.2.3.

Run from the repository root:
    python -m pytest -q /path/to/Bismillah_Criterion_RedTeam_Adversarial_v023_Bismillah.py

These are prospective acceptance tests, NOT claimed passing tests. Some are expected
failures on commit d1010f7d; fix the product rather than weakening assertions.
"""
import json
from pathlib import Path
from criterion.api import verify
from criterion.record import verify_chain
from criterion.__main__ import main


def check(source, output, mode="strict"):
    # Force deterministic word-only behavior so an optional model download
    # cannot mask a regression. Repeat the tests separately with semantic on.
    import criterion.divergence as divergence
    previous = divergence._USE_SEMANTIC
    try:
        divergence._USE_SEMANTIC = False
        return verify(source, output, mode=mode, record=False)
    finally:
        divergence._USE_SEMANTIC = previous


def test_num_swapped_roles_is_not_trustworthy():
    result = check("The dosage is 5 mg and the limit is 10 mg.",
                   "The dosage is 10 mg and the limit is 5 mg.")
    assert not result.is_trustworthy, result.to_dict()


def test_predicates_swapped_between_subjects_is_not_trustworthy():
    result = check("The treatment is safe for adults and unsafe for children.",
                   "The treatment is unsafe for adults and safe for children.")
    assert not result.is_trustworthy, result.to_dict()


def test_negation_moves_between_subjects_is_not_trustworthy():
    result = check("Alice is not approved and Bob is approved.",
                   "Alice is approved and Bob is not approved.")
    assert not result.is_trustworthy, result.to_dict()


def test_empty_both_is_invalid_not_faithful():
    result = check("", "")
    assert not result.is_trustworthy, result.to_dict()


def test_empty_output_is_recorded_when_requested(tmp_path):
    p = tmp_path / 'ledger.jsonl'
    result = verify("The report is complete.", "", mode="strict", record=True, record_path=p)
    assert result.record_id is not None
    assert p.exists() and p.stat().st_size > 0


def test_missing_ledger_fails_verification(tmp_path):
    ok, message = verify_chain(tmp_path / 'missing.jsonl')
    assert not ok, message


def test_empty_batch_must_fail(tmp_path):
    input_file = tmp_path / 'empty.jsonl'
    input_file.write_text('', encoding='utf-8')
    output_file = tmp_path / 'results.jsonl'
    assert main(['batch', str(input_file), '--no-record', '-o', str(output_file)]) != 0


def test_record_contains_mode_and_configuration(tmp_path):
    p = tmp_path / 'ledger.jsonl'
    result = verify('The patient may improve.', 'The patient will improve.',
                    mode='summary', record=True, record_path=p)
    entry = json.loads(p.read_text(encoding='utf-8').splitlines()[0])
    assert entry['mode'] == 'summary'
    assert 'detector_config' in entry
    assert entry['trust_score'] == result.trust_score
    assert entry['trust_level'] == result.trust_level


def test_rag_does_not_silently_drop_a_retained_claims_missing_support():
    source = ('The device is safe. Because a certified lab tested its insulation. '
              'The warranty is valid.')
    output = 'The device is safe. The warranty is valid.'
    result = check(source, output, mode='rag')
    # Current policy intentionally permits selective answers. A stronger
    # commercial policy should report material qualification/evidence loss
    # even if it does not automatically block every omitted source sentence.
    assert result.total_divergences > 0 or not result.is_trustworthy, result.to_dict()


def test_record_chain_has_external_anchor(tmp_path):
    # Policy-level test: plain hash chains cannot prevent a privileged writer
    # from regenerating every hash. An authentic product needs signed and
    # externally anchored checkpoints (not implemented in v0.2.3).
    # Placeholder for an integration test once anchor API exists.
    import pytest
    pytest.skip('Requires signed checkpoint / external anchor interface')
