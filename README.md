# Criterion

**Your AI's output looks right. Criterion checks whether it is.**

Criterion is a trust layer for AI-generated content. It decomposes text into semantic atoms, traces their relationships, and detects where an AI's output diverges from its source — not just that it changed, but *how* it changed and *what to fix*.

No API keys. No external services. Runs locally.

> **Status: Alpha (v0.7.0).** Suitable for developer evaluation and assisted review. Not yet validated for autonomous high-stakes trust decisions. See [Limitations](#limitations) for what it catches and what it misses.

## Install

`criterion-ai` is not on PyPI yet. Until it is published, install from this repository:

```bash
pip install "criterion-ai[all] @ git+https://github.com/BayyinahEnterprise/Criterion"
```

| Install | Adds | What it can do |
|---------|------|----------------|
| `criterion-ai` | ~2 MB | Deterministic checks only: changed numbers, sign flips, negation, role swaps, hidden characters. Can find a problem; can never clear an output (see below) |
| `criterion-ai[nli]` | PyTorch, plus three models downloaded on first use: two grounding scorers (~1.4 GB and ~1.7 GB) and a ~740 MB NLI model | Decides, sentence by sentence, whether the source supports the output; labels contradictions; strict-mode omission checks |
| `criterion-ai[all]` | + a ~90 MB embedding model | Better passage ranking and paraphrase recognition |

### The models, and what happens without them

| Model | Job | Licence |
|-------|-----|---------|
| `lytang/MiniCheck-RoBERTa-Large` | **Grounding scorer 1.** P(supported) for each output sentence, taken as the maximum over ~400-token chunks of the source. | MIT |
| `yaxili96/FactCG-DeBERTa-v3-Large` | **Grounding scorer 2.** The same question, asked of a differently trained model. The verdict is decided by the average of the two; where they disagree the sentence is labelled *unsure*. | MIT |
| `tasksource/deberta-base-long-nli` | **Labeller.** Says whether a failed sentence contradicts the source or merely lacks support; powers strict-mode omission checks and `question=` | Apache-2.0 |
| `all-MiniLM-L6-v2` | Passage ranking | Apache-2.0 |

All four are pinned to fixed revisions, and the scorers are loaded from safetensors only (no pickle is ever deserialised; the pinned MiniCheck revision is the Hub's safetensors conversion, verified tensor-for-tensor against the authors' original).

`CRITERION_SCORER` chooses the scorer: `ensemble` (default, both models), `minicheck` or `factcg` alone (roughly a third and two thirds of the time, and no confidence label), or `nli` (smaller download, about eight points less accurate). Several can be joined with `+`. If any configured model cannot load, the scorer as a whole fails; it never quietly carries on with fewer.

Criterion **fails closed**. If the grounding scorer is not installed or cannot load, `verify()` raises `CapabilityError`, because without it nothing checks whether the source supports the output — and that is the job. A working NLI model alone does not count.

To run the deterministic checks alone, opt in: `require_nli=False`, `CRITERION_REQUIRE_NLI=0`, or `--allow-degraded`. A problem they find is still reported as `divergent` / `unreliable`. A clean result is reported as **`unverified`**, which is never `is_trustworthy`.

## Quick start

```python
import criterion

result = criterion.verify(
    source="The vaccine is safe. Clinical trials with 40,000 participants showed no serious adverse effects.",
    output="The vaccine is safe.",
    mode="strict",
)
print(result.trust_level)        # "divergent"
print(result.trust_score)        # 0.38
print(result.is_trustworthy)     # False — evidence was dropped
print(result.summary)            # "2 divergences detected. 1 critical. 1 broken bond. ..."
print(result.capabilities)       # {"nli": True, "semantic": True, ...} — which detectors actually ran
```

`result.capabilities` and the record's `detector_config` state which detectors and model revisions produced each verdict.

### Groundedness score and thresholds

```python
result.groundedness        # 0-1: the weakest output sentence's support in the source
result.sentence_support    # ({"text": ..., "support": 0.97}, ...) one per judged sentence
result.thresholds          # the thresholds this verdict used

# Demand stronger support (more flags) ...
criterion.verify(source=docs, output=answer, mode="rag", thresholds={"grounded": 0.7, "fabrication": 0.7})
# ... or accept weaker support (fewer flags)
criterion.verify(source=docs, output=answer, mode="rag", thresholds={"grounded": 0.3, "fabrication": 0.3})
```

- `grounded` (default 0.5): the support a sentence needs to count as supported. 0.5 is the fixed threshold the fact-checking literature reports with.
- `fabrication` (default 0.5): below this an unsupported sentence is critical and blocks trust on its own; between `fabrication` and `grounded` it is reported as major and does not block alone. Equal to `grounded` by default, so every unsupported sentence blocks.
- `verbatim` (default 0.9): how close to a word-for-word copy a sentence must be before a deterministic check (changed number, flipped negation, role swap) may overrule the scorer. Below it such a check is shown as a note, because a summary sentence is not wrong for differing from whichever source sentence it happens to resemble.

On the CLI: `--fabrication-threshold`. The Benchmarks section shows what the `grounded` dial buys.

### Confidence: a third answer

Two grounding models score every sentence. Their average decides the verdict; whether they agree says how much weight it will bear.

```python
result.confidence          # "confident" | "probable" | "unsure" | None (one model only)
result.unsure_count        # sentences the two models disagree on
result.sentence_support    # each row also carries "models": {"minicheck": .., "factcg": ..}, "confidence" and "evidence"
```

| Label | Meaning | Share of claims | Verdict was right |
|-------|---------|-----------------|-------------------|
| `confident` | both models agree, each at least 0.3 clear of the threshold | 58% | 89% |
| `probable` | both agree, at least one narrowly | 25% | 69% |
| `unsure` | they land on opposite sides of the threshold | 18% | 59% |

(440 fresh LLM-AggreFact test rows; see Benchmarks.) An output takes its weakest sentence's label. `unsure` does not change `is_trustworthy`; it tells a reviewer where to look first. To make a gate stop on it, use `--fail-on-unsure` (exit code 3) or test `result.confidence`. The 0.3 margin is `thresholds["confident"]`.

## Three modes for three jobs

### RAG verification

> Is the LLM answer grounded in the retrieved documents?

```python
result = criterion.verify(source=retrieved_docs, output=llm_answer, mode="rag")
```

RAG mode is strict on **additions** (fabricated content) and **strengthening** (overclaiming), lenient on omissions — the answer doesn't need to repeat everything. Every answer sentence is scored for support against every chunk of the source; a sentence no chunk supports is flagged, and labelled a contradiction when the NLI model finds one.

**Pass the question when the answer is a fragment.** A bare `"Delhi"` or `"yes"` is not a statement; on its own Criterion can only check that its words occur in the source, so a wrong name that the source happens to mention passes. With the question, the fragment is judged as an answer:

```python
criterion.verify(source=retrieved_docs, output="Martin Scorsese", mode="rag",
                 question="Who directed the film?")
```

This is a partial fix, not a solved problem — see Benchmarks.

### Summary verification

> Does this summary faithfully represent the source?

```python
result = criterion.verify(source=long_article, output=ai_summary, mode="summary")
```

Summary mode knows compression is the job. Flags fabricated details, changed numbers, negation inversions. Permits omissions and rewording.

### Strict comparison

> Flag every deviation from the source.

```python
result = criterion.verify(source=original, output=revised, mode="strict")
```

Everything counts — paraphrases, compression, emphasis shifts, lost precision ("March 15, 2027" → "early 2027"). Use for contracts, legal, medical, regulatory content.

**The mode you pass is the mode you get.** A short output never relaxes `strict`. `mode="auto"` (the default) *does* choose the lenient summary mode when the output is much shorter than the source — and the output is the thing under test — so pass an explicit mode in any gate.

## What it catches

### Deterministic checks (always available)

| Check | Example | Severity |
|-------|---------|----------|
| Numeric change | "60%" → "90%" | Critical |
| Currency change | "$1,000" → "$2,000" | Critical |
| Negation inversion | "was not present" → "was present" | Critical |
| Prefix antonym | "safe" → "unsafe", "legal" → "illegal" | Critical |
| Common antonym | "innocent" → "guilty", "true" → "false" | Critical |
| Hedge removal | "may improve" → "will improve" | Major |
| Intensifier added | "improved" → "always improved" | Major |
| Evidence dropped | Claim retained, supporting evidence omitted | Critical |
| Content added | Output contains claims not in source | Major |
| Content omitted | Source material not in output | Varies by mode |
| Role swap | "5 mg dose, 10 mg limit" → "10 mg dose, 5 mg limit" | Critical |
| Sign flip | "-5%" → "5%" | Critical |
| Hidden characters | zero-width characters, bidi overrides, mixed-script look-alike words | Critical |

Numbers are compared by value, so `1,500,000` = `1.5 million`, `10%` = `10 percent`, `$5M` = `5 million dollars`; contractions are expanded, so `isn't` = `is not`.

### Model checks (requires `[nli]` install)

| Check | Example | Severity |
|-------|---------|----------|
| Entity substitution | "Alice signed" → "Bob signed" | Critical |
| Relation reversal | "Alice sold to Bob" → "Bob sold to Alice" | Critical |
| Predicate swap | "safe for adults, unsafe for children" → reversed | Critical |
| Negation scope | "Alice not approved, Bob approved" → reversed | Critical |
| Inflected antonym | "revenue increased" → "revenue decreased" | Critical |
| Unit change | "5 mg" → "5 g" | Critical |
| Location change | "meeting in Paris" → "meeting in London" | Critical |
| Currency unit | "$100" → "€100" | Critical |
| Paraphrased contradiction | "not safe for children" → "doctors can give it to young kids" | Critical |
| Fabricated sentence | An answer sentence no source passage supports | Critical / Major |
| Fabricated clause | "…reduced blood pressure" → "…reduced blood pressure and cures cancer" | Critical / Major |

### What it does NOT catch (known limitations)

Without the models installed, Criterion **cannot detect**:
- Entity substitution (Alice → Bob) — same sentence structure, different names
- Relation reversal (sold-to vs bought-from) — same words, different direction
- Inflected antonyms (increased → decreased) — not in the static antonym table
- Unit/currency unit changes ($100 → €100) — symbols are stripped for numeric comparison

These require the models. The grounding scorer decides whether each sentence is supported; the NLI model labels why a failed sentence fails.

**Even with the models, Criterion does not catch:**
- More than half of wrong fragment answers that name something the source mentions, even with `question=` (and all of them without it)
- Errors that span sentences, and claims whose support needs world knowledge the source does not state
- Support that only emerges from more than about five source sentences taken together (such a sentence is reported as unsupported — a false positive)
- Factual errors that require world knowledge (the source itself may be wrong)
- Multi-sentence reasoning chains where the contradiction emerges from combining claims
- Non-English text (all heuristics are English-only; NLI has limited multilingual capability)

**Criterion is a structural consistency checker, not a truth oracle.** It tells you whether the output is faithful to the source, not whether the source is correct.

## Every divergence is actionable

```python
for d in result.divergences:
    print(d.explanation)            # What went wrong
    print(d.suggested_correction)   # What it should say
    print(d.verification_method)    # How to confirm this is real
    print(d.severity)               # critical, major, minor, cosmetic
    print(d.confidence)             # 0.0–1.0
    print(d.code)                   # stable machine-readable code, e.g. "numeric_change",
                                    # "negation", "role_swap", "ungrounded", "contradicts_source",
                                    # "unsupported_detail", "detail_lost", "hidden_chars", "omission"
```

Filter and route on `d.code`, not on the wording of `d.explanation`.

## CLI

```bash
# Verify a single pair
criterion verify source.txt output.txt
criterion verify source.txt output.txt --mode rag
criterion verify source.txt output.txt --mode strict --json

# Batch verification
criterion batch pairs.jsonl --mode rag -o results.jsonl

# Verify the append-only audit chain
criterion chain --record .criterion/record.jsonl
criterion chain --record .criterion/record.jsonl --expect-head <id> --expect-count <n>

# CI gate: explicit mode (a missing grounding model is an error by default)
criterion batch pairs.jsonl --mode rag -o results.jsonl

# Fewer false alarms / heuristics only
criterion verify source.txt output.txt --mode rag --fabrication-threshold 0.05
criterion verify source.txt output.txt --allow-degraded

# Keep hashes only (no source/output text on disk)
criterion verify source.txt output.txt --no-store-text

# Stop when the two grounding models disagree
criterion verify source.txt output.txt --mode rag --fail-on-unsure
```

Exit codes: `0` = trustworthy, `2` = untrustworthy or unverified, `3` = trustworthy but unsure (only with `--fail-on-unsure`), `1` = error (including a missing grounding model).

## Python API

```python
import criterion

# Single verification
result = criterion.verify(source=src, output=out, mode="rag")

# Batch
results = criterion.verify_batch(pairs=[(s1,o1), (s2,o2)], mode="rag")

# JSON serialization
print(result.to_dict())

# SARIF output for GitHub Code Scanning
from criterion.github import write_sarif
write_sarif(results, "criterion.sarif")

# Reviewer workflow (see "Review" below)
from criterion import review
review.queue(); review.stats()
```

## How it works

1. **Capture** — Decompose text into typed Atoms (claim, evidence, definition, reference, conclusion, metadata, observation), connected by typed Bonds (supports, contradicts, defines, cites, follows, elaborates, qualifies), composed into Molecules.

2. **Divergence** — Align atoms by similarity. Run deterministic checks (numeric, negation, antonym) before similarity gates. Run NLI contradiction detection on aligned pairs. Classify into 8 divergence types × 4 severities.

3. **Verdict** — Score trustworthiness with mode-specific filtering (RAG/summary/strict).

4. **Record** — Append-only JSONL with content-hashed chain for tamper detection.

## The record: what it proves

By default every `verify()` call appends an entry (including the full source and output text) to `.criterion/record.jsonl`. Pass `record=False` to skip it or `store_text=False` to keep only SHA-256 hashes.

| Setup | Detects |
|-------|---------|
| Default (unkeyed) | Any edit to an entry; reordering; removal from the middle. **Not** a full rewrite by someone with write access, and **not** deletion of the newest entries. |
| `CRITERION_RECORD_KEY` set | The above, plus a rewrite by anyone who lacks the key (HMAC-SHA256 per entry). |
| `--expect-head` / `--expect-count` | The above, plus truncation — provided you store the head id and count somewhere the writer cannot reach. `criterion chain` prints both. |

Each entry records the mode, the Criterion version that wrote it, and a `detector_config` (which detectors ran, model names and pinned revisions, thresholds). Appends take a file lock, and Criterion refuses to append after a corrupt last line rather than silently starting a new chain. Ledgers written by any earlier version still verify.

From v0.7.0 each entry also records every judged sentence: its support score, each model's own score, the confidence label, and (unless `store_text=False`) the sentence and the source passage that supported it best. These are what a reviewer is later shown.

## Review: a person where the machine is weakest

About one verdict in five is wrong, so someone has to look at some of them. `criterion review` decides which, records what the person concluded, and keeps score.

```bash
criterion review queue                      # what needs a person, most doubtful first
criterion review walk --reviewer "Aisha"    # go through it: [s]upported / [u]nsupported / [?] unclear
criterion review decide 86bd62e8 unsupported --sentence 2 --reviewer "Aisha" --note "cost is never mentioned"
criterion review stats                      # how often machine and reviewers agreed, by confidence label
```

```python
from criterion import review
items = review.queue()                                        # ReviewItem: text, evidence, support, models, confidence
review.decide(items[0].entry_id, "unsupported", "Aisha", sentence=items[0].sentence)
review.stats()
```

- **Order.** `unsure` first, then `probable`, then `confident`; within each, the score nearest the threshold first. That is the order of the machine's measured reliability (59%, 69%, 89%), reversed.
- **What is queued.** Every sentence the machine did not clear, and every sentence it cleared without being `confident`. Confident, supported sentences are left out except for a spot-check sample (`--spot-check`, default 5%), chosen by hash so the same record always gives the same sample. Without that sample the error rate of confident verdicts on your data could never be measured.
- **A decision never changes a verdict.** It is a new entry in the same hash chain that names the verdict it concerns and repeats its source and output hashes. `criterion chain` rejects a decision that has been edited, points at no earlier verdict, or is bound to different text. Deciding again appends again; the later decision is the one `stats` counts and the earlier one stays in the record.
- **What it proves, and what it does not.** That a decision was recorded at that position and not altered since. Not who made it: the reviewer's name is whatever was typed. With `CRITERION_RECORD_KEY`, that the entry was written by someone holding the key.
- **`stats` is not a random sample.** Doubtful sentences are reviewed first, so agreement on `unsure` and `probable` describes those groups, and only the spot-checked share speaks for `confident` verdicts in general.

## Benchmarks

All figures are measured through the public API, `criterion.verify(source, output, mode="rag")`, with `[all]` installed and default thresholds. "Flagged" means `is_trustworthy == False`.

### LLM-AggreFact

LLM-AggreFact (Tang et al., 2024) is the benchmark this field reports: eleven datasets of real model outputs with human labels, scored by balanced accuracy averaged over datasets at a fixed 0.5 threshold.

| Version | Balanced accuracy | 95% interval | Sample |
|---------|-------------------|--------------|--------|
| 0.4.0 | 64.8% | 61.8–67.7 | 1,650 test rows (150 per dataset) |
| 0.5.0 | 71.8% | 67.7–75.7 | 440 fresh test rows (20 supported + 20 unsupported per dataset) |
| 0.6.0 | 79.1% | 75.5–82.3 | 440 further fresh test rows, drawn the same way |

**Do not read 71.8 → 79.1 as a seven-point gain.** The two samples are different rows, and the 0.6.0 sample is easier: MiniCheck alone, the 0.5.0 scorer, reaches 78.9% on it. Averaging in the second model added 0.2 points there (interval −2.3 to +2.7). On 528 dev rows the same change added 3.0 points (0.8 to 5.3). The honest summary is a small gain, somewhere between nothing and three points.

What the second model reliably buys is the confidence label. It was designed on dev and held on the test sample:

| | Dev (528 rows) | Test (440 rows) |
|---|---|---|
| Models agree: share of claims / verdict accuracy | 82% / 77.1% | 82% / 83.3% (79.8–86.4) |
| Models disagree (`unsure`): accuracy | 46.1% | 59.0% (48.1–70.3) |
| `confident`: share / accuracy | ~50% / ~84% | 58% / 89.2% (85.5–92.5) |

On one shared set of 264 dev rows the gain from 0.4.0 to 0.5.0 is 9.1 points (interval 3.0–15.2). For orientation, published fixed-threshold averages on the full test set are 73.5 for MiniCheck-RoBERTa and 75.6 for FactCG-DeBERTa (the two scorers Criterion averages), 75.0 for MiniCheck-Flan-T5 and 75.9 for GPT-4o.

These are **samples, not a leaderboard entry**: the full test set has 29,320 rows and would take about 37 hours on the two CPU cores this was measured on. Per-dataset results at 40 rows each are too noisy to report. The benchmark's terms allow evaluation only; no row of it was used to train or fine-tune anything, and thresholds were left at 0.5.

**What the `grounded` threshold buys** (264 dev rows):

| `grounded` | Supported claims accepted | Unsupported claims caught |
|------------|---------------------------|---------------------------|
| 0.3 | 73% | 72% |
| 0.5 (default) | 66% | 78% |
| 0.7 | 53% | 83% |

### HalluMix and HaluEval

300 held-out pairs each (HaluEval QA rows 300–599; from HalluMix rows 600–1,199, the first 300 pairs whose source is under 6,000 characters).

| Dataset | Version | Precision | Recall | F1 | Faithful answers flagged |
|---------|---------|-----------|--------|----|--------------------------|
| HalluMix | 0.3.1 | 86.7% | 13.1% | 22.8% | 4 / 102 |
| HalluMix | 0.4.0 | 90.3% | 89.4% | 89.8% | 19 / 102 |
| HalluMix | 0.5.0 | 86.0% | 96.0% | 90.7% | 31 / 102 |
| HalluMix | 0.6.0 | 85.5% | 98.0% | 91.3% | 33 / 102 |
| HaluEval QA | 0.3.1 | 85.7% | 7.7% | 14.2% | 2 / 145 |
| HaluEval QA | 0.4.0 | 96.9% | 60.0% | 74.1% | 3 / 145 |
| HaluEval QA | 0.5.0 | 96.7% | 75.5% | 84.8% | 4 / 145 |
| HaluEval QA | 0.6.0 | 96.7% | 76.8% | 85.6% | 4 / 145 |

The second model changed little here: 0.6.0 differs from 0.5.0 by a handful of rows on either dataset. The confidence label again separated good verdicts from doubtful ones — on HalluMix the verdict was right for 92% of `confident` outputs (204), 78% of `probable` (51) and 72% of `unsure` (32); on HaluEval, where half the answers are fragments the label does not cover, 88%, 57% and 47% (94, 37 and 19 outputs).

Two cautions. On HalluMix, 0.5.0 and 0.6.0 catch more than 0.4.0 but also flag more faithful answers (about 31% against 19%); lower `grounded` to trade that back. And HaluEval's errors are synthetic — the LLM-AggreFact authors left it out of their benchmark for that reason — so weigh it accordingly.

### Fragment answers and `question=`

In a synthetic test (150 HaluEval questions whose correct answer is a fragment, each also paired with a different name taken from the same source): without the question every wrong name passed; with it 43% of wrong names were flagged, and 9% of correct answers were flagged too. Useful, far from solved.

Figures quoted for v0.2–v0.3.1 came from the mode-less kernel pipeline rather than `verify()` and are not comparable. `python bench/run_benchmark.py` reproduces the HalluMix/HaluEval rows.

## Limitations

This is an alpha release that has been through four rounds of independent adversarial auditing. Known limitations:

- **NLI is optional but critical.** Without it, entity/relation/unit swaps pass undetected. Install with `[nli]`.
- **About a fifth to a quarter of verdicts on hard benchmarks are wrong.** 72–79% balanced accuracy on LLM-AggreFact samples means Criterion is a reviewer's aid, not an oracle. The best published systems reach about 76–77% on the full test set. Even `confident` verdicts were wrong about one time in nine.
- **Slow on CPU.** Every output sentence is scored against every chunk of the source: with both models, about 7 seconds per claim on two CPU cores for a typical document (median), 10.5 on average and 33 at the 95th percentile. `CRITERION_SCORER=minicheck` is about three times faster and gives no confidence label. Cost grows with (output sentences) × (source length) × (models).
- **Supported or unsupported only.** The grounding scorer has no contradiction class; "contradicts" labels come from the smaller NLI model and are less reliable than the support decision.
- **Sentence-level.** Each output sentence is judged on its own. An error that only appears across sentences (a pronoun that now points at the wrong person) can pass.
- **Model limits.** No world or unit knowledge beyond the source, 512-token window per chunk, English only, and it can be fooled by adversarial phrasing. Near-verbatim copies with a changed number or flipped negation are also caught deterministically.
- **Alignment is quadratic in rewritten sentences.** Verbatim sentences align in linear time; the rest are compared pairwise. Past 1,000,000 comparisons (about 1,000 × 1,000 rewritten sentences) Criterion raises `InputTooLargeError` instead of hanging — verify long documents in sections or raise `CRITERION_MAX_COMPARISONS`.
- **Audit chain is symmetric-keyed at best.** There are no public-key signatures or built-in external anchoring; see "The record: what it proves".
- **Models are downloaded from the Hugging Face Hub.** The default models are pinned to fixed revisions. `CRITERION_NLI_MODEL` / `CRITERION_MODEL` load whatever you point them at, unpinned unless you also set `CRITERION_NLI_REVISION` / `CRITERION_MODEL_REVISION`.
- **Not validated for autonomous trust decisions.** Use as a review assistant with human oversight, not as a sole gate for medical, legal, or financial content.
- **English only.** All heuristic checks assume English text.

## Audit history

Criterion v0.2.0–v0.3.1 was independently audited by four AI systems across six rounds. The audits identified and drove fixes for: audit-chain integrity, RAG mode crashes, batch exit codes, empty input handling, currency extraction, NLI integration, antonym detection, numeric canonicalization, and mode-specific filtering policies. The auditors' consensus at v0.3.1: "experimental developer use: viable; controlled customer pilots: conditional; high-assurance commercial release: not approved."

v0.4.0 added the fixes from a further red-team pass, and v0.5.0 rebuilt the verdict around a grounding scorer after measuring on LLM-AggreFact; v0.6.0 added a second grounding model and the confidence label, and v0.7.0 the reviewer workflow; see [CHANGELOG.md](CHANGELOG.md). None of the four has been independently reviewed.

## Requirements

- Python ≥ 3.10
- numpy
- sentence-transformers (optional, `[semantic]`)
- transformers + torch (optional, `[nli]` — strongly recommended)

## License

MIT
