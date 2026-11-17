# Changelog

## 0.4.0 — red-team hardening

Every item below was reproduced against v0.3.1 first and is pinned by a test in `tests/test_redteam.py`.

### Decisions taken after comparing with other groundedness checkers

- **Fails closed.** `require_nli` now defaults to on: a missing or failing NLI model raises `CapabilityError`. Heuristics-only is an explicit opt-in (`require_nli=False`, `CRITERION_REQUIRE_NLI=0`, `--allow-degraded`), and its best verdict is the new level `unverified`, which is never `is_trustworthy`.
- **Default NLI model is now `tasksource/deberta-base-long-nli`** (Apache-2.0, pinned). On held-out HalluMix the share of faithful answers flagged fell from 42% to 19% while recall went from 94% to 89%; on HaluEval recall fell from 72% to 60%. The old model's scores saturate at 0/1, so no threshold tried brought its false alarms below a third.
- **A groundedness score and caller-set thresholds.** `VerifyResult.groundedness`, `.sentence_support`, `.thresholds`; `verify(thresholds={...})`; `--fabrication-threshold`. Thresholds used are stored in each record's `detector_config`.
- **`question=`** on `verify()`: a fragment answer is judged as an answer to the question instead of merely being located in the source. Partial: about 45% of wrong-entity fragments caught in a synthetic test, against 0% without it.
- Known regression from the model change: "365 days" → "about a year" is scored as a contradiction (`tests/test_nli.py::test_faithful_numeric_paraphrase`, marked xfail).
- README stated the old NLI model was 140 MB; its weights are ~570 MB.

### Detection

- **An explicit mode is no longer relaxed by a short output.** `find_divergences` inferred "this is a summary" from the sentence-count ratio regardless of the mode passed to `verify`. In `strict` mode, dropping five of six sentences returned *adapted / trustworthy*; in `rag` mode — where answers are always shorter than the documents — every fabricated sentence was downgraded to MINOR and a wholly invented answer scored *faithful (94%)*. The mode is now passed down and decides leniency.
- **Output that fails lexical alignment is now grounded with NLI.** NLI used to run only on sentence pairs with ≥ 50% character overlap, so a contradiction phrased in different words was never judged. Each unsettled output sentence is now tested against its most relevant source sentences, singly and concatenated: supported → no finding (and the source sentence is not reported as omitted); contradicted → CRITICAL; unsupported → CRITICAL or MAJOR by how little support exists.
- **Fabricated clauses inside an otherwise true sentence are caught** ("…reduced blood pressure *and also cures cancer*"). Aligned pairs the source does not entail go through the same grounding.
- **Fragments are not sent to NLI.** Bare noun phrases produce confident nonsense from an NLI model; they are checked by containment in the source instead.
- **Strict mode reports lost precision** (`detail_lost`) when the source implies the output but not the reverse.
- **Role swaps are caught without any model** (`role_swap`): same words, different order, where a number, negation or opposite moved.
- **Hidden content** (`hidden_chars`): zero-width characters next to ASCII text, bidi overrides, tag characters and mixed-script look-alike words in the output are CRITICAL. All checks and model calls now see NFKC-normalised text with format characters removed.
- **False positives removed:** `isn't` vs `is not`, `can't` vs `cannot`, typographic vs straight apostrophes, `10%` vs `10 percent`, `1,500,000` vs `1.5 million`, and double negatives NLI judges equivalent no longer raise CRITICAL. Numbers are compared by value; a leading minus is now significant (`-5%` ≠ `5%`), and both ends of a range (`5-10`) are extracted.
- **Alignment is best-first with a verbatim fast path** instead of first-come greedy, and refuses inputs past a comparison budget (`InputTooLargeError`) instead of hanging.
- `Divergence.code`: a stable machine-readable finding code. Mode policy keys off it; it previously matched the substring `"set()"` in the human-readable explanation.

### Honesty about what ran

- `VerifyResult.capabilities`, `to_dict()["capabilities"]` and the record's `detector_config` state whether NLI and embeddings actually ran, with the error if not. Model failures were previously swallowed by `except Exception: pass`.
- The NLI label order is read from the model's config. It was hard-coded, so a `CRITERION_NLI_MODEL` with a different order silently swapped entailment and neutral.
- Default models are pinned to Hub revisions.
- One version string (`criterion/_version.py`). Records claimed `0.2.2`, captures `0.2.0`, SARIF `0.2.0`, the package `0.3.1`.
- `bench/run_benchmark.py` measures `verify()` in an explicit mode, not the kernel pipeline.
- `criterion.verify_batch`, documented in the README, is now actually exported.
- The `_align_atoms` docstring described a semantic second pass that did not exist.

### Record

- **Entry ids are computed over canonical JSON** (hash scheme 2). Scheme 1 concatenated fields with no delimiter: moving a digit from the start of `summary` onto the end of `broken_bonds` changed the record and left the hash, and `verify_chain`, unchanged. Scheme-1 ledgers still verify; new entries are always scheme 2.
- Entries record `mode` and `detector_config`; both are hashed.
- Optional keyed chain (`CRITERION_RECORD_KEY`, HMAC-SHA256) and external anchoring (`verify_chain(expected_head=, expected_count=)`, `criterion chain --expect-head/--expect-count`). An unkeyed, unanchored chain still cannot detect a full rewrite or deletion of the newest entries; the README now says exactly that.
- Appends take an exclusive file lock; a corrupt last line raises `RecordCorruptError` instead of silently restarting the chain at `genesis`.
- Empty-output and empty-source verdicts are recorded.
- `store_text=False` / `--no-store-text` keep hashes only.

### CLI, CI, SARIF

- `criterion verify <long literal text>` no longer crashes with `OSError: File name too long`.
- `criterion batch` accepts `--record`, reads and writes UTF-8, and rejects non-string fields.
- Workflow: installs `[all]` (it installed `[semantic]`, i.e. ran the gate without NLI), passes `--require-nli`, fails closed on rows without a verdict (was `r.get('is_trustworthy', True)`), passes inputs through `env`, and declares `permissions: contents: read`.
- SARIF: removed the invalid `fixes` objects (no `artifactChanges`), optional `artifact_uris` so results carry the location GitHub needs to display them.

### Behaviour changes to be aware of

- RAG and strict verdicts are stricter than v0.3.1. On held-out HalluMix the share of faithful answers flagged is 19% (4% in v0.3.1, which caught 13% of hallucinations against 89% now).
- A core-only install (`pip install criterion-ai`) raises `CapabilityError` from `verify()` unless degraded mode is requested.
- The NLI model download is larger (~740 MB) and inference is slower (about one second per answer on two CPU cores).
- `verify()` raises `TypeError` for non-string input, `RecordCorruptError` for a corrupt ledger tail, and `InputTooLargeError` past the comparison budget.
- With `mode=None`, `find_divergences` keeps its legacy length-based leniency (used by `criterion.pipeline`).
