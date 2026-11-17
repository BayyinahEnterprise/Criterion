# Criterion

**Your AI's output looks right. Criterion checks whether it is.**

Criterion is a trust layer for AI-generated content. It decomposes text into semantic atoms, traces their relationships, and detects where an AI's output diverges from its source — not just that it changed, but *how* it changed and *what to fix*.

No API keys. No external services. Runs locally.

> **Status: Alpha (v0.4.0).** Suitable for developer evaluation and assisted review. Not yet validated for autonomous high-stakes trust decisions. See [Limitations](#limitations) for what it catches and what it misses.

## Install

```bash
pip install "criterion-ai[all]"         # recommended: NLI model + sentence embeddings
pip install "criterion-ai[nli]"         # NLI model only
pip install criterion-ai                # heuristics only — see "Without the NLI model"
```

| Install | Adds | Catches |
|---------|------|---------|
| `criterion-ai` | ~2 MB | Changed numbers, currency ($1K→$2K), sign flips, negation inversion, prefix antonyms (safe→unsafe), role swaps, hidden characters, broken evidence bonds |
| `criterion-ai[nli]` | PyTorch + a ~740 MB model downloaded on first use | Everything above **plus** contradictions in any wording, fabricated sentences and clauses, entity swaps (Alice→Bob), relation reversals, unit changes (mg→g), location substitution |
| `criterion-ai[all]` | + an ~90 MB embedding model | Everything above + better passage retrieval for grounding and paraphrase recognition |

### Without the NLI model

Criterion **fails closed**. If the NLI model is not installed or cannot load, `verify()` raises `CapabilityError`, because without it contradictions and fabricated claims are never checked — and that is the job.

To run the heuristics alone, opt in: `require_nli=False`, `CRITERION_REQUIRE_NLI=0`, or `--allow-degraded`. A problem the heuristics find is still reported as `divergent` / `unreliable`. A clean result is reported as **`unverified`**, which is never `is_trustworthy`: heuristics can find a problem, they cannot clear an output.

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

# Tune the trade-off: fewer false alarms ...
criterion.verify(source=docs, output=answer, mode="rag", thresholds={"fabrication": 0.05})
# ... or catch more
criterion.verify(source=docs, output=answer, mode="rag", thresholds={"fabrication": 0.5})
```

`fabrication` is the main dial: an unsupported sentence blocks trust on its own only when its best source support is below it (default 0.2). `grounded` (default 0.5) is the support a sentence needs to count as supported at all; `contradiction` (default 0.7) is the confidence a contradiction needs. On the CLI: `--fabrication-threshold`. The Benchmarks section shows what the dial buys.

## Three modes for three jobs

### RAG verification

> Is the LLM answer grounded in the retrieved documents?

```python
result = criterion.verify(source=retrieved_docs, output=llm_answer, mode="rag")
```

RAG mode is strict on **additions** (fabricated content) and **strengthening** (overclaiming), lenient on omissions — the answer doesn't need to repeat everything. With NLI installed, every answer sentence is checked for support against the most relevant source passages: unsupported sentences are flagged, sentences a passage contradicts are critical, and paraphrases the aligner missed are recognised as supported.

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

### NLI checks (requires `[nli]` install)

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

Without the NLI model installed, Criterion **cannot detect**:
- Entity substitution (Alice → Bob) — same sentence structure, different names
- Relation reversal (sold-to vs bought-from) — same words, different direction
- Inflected antonyms (increased → decreased) — not in the static antonym table
- Unit/currency unit changes ($100 → €100) — symbols are stripped for numeric comparison

These require the NLI model, which classifies sentence pairs as entailment, contradiction, or neutral. The default is `tasksource/deberta-base-long-nli` (Apache-2.0, ~740 MB, pinned to a fixed revision).

**Even with NLI, Criterion does not catch:**
- About half of wrong fragment answers that name something the source mentions, even with `question=` (and all of them without it)
- Equivalences that need world knowledge: the default model scores "365 days" → "about a year" as a contradiction
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

# CI gate: explicit mode (a missing NLI model is an error by default)
criterion batch pairs.jsonl --mode rag -o results.jsonl

# Fewer false alarms / heuristics only
criterion verify source.txt output.txt --mode rag --fabrication-threshold 0.05
criterion verify source.txt output.txt --allow-degraded

# Keep hashes only (no source/output text on disk)
criterion verify source.txt output.txt --no-store-text
```

Exit codes: `0` = trustworthy, `2` = untrustworthy or unverified, `1` = error (including a missing NLI model).

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

Each entry records the mode, the Criterion version that wrote it, and a `detector_config` (which detectors ran, model names and pinned revisions, thresholds). Appends take a file lock, and Criterion refuses to append after a corrupt last line rather than silently starting a new chain. Ledgers written by v0.3.1 and earlier still verify.

## Benchmarks

Measured through the public API, `criterion.verify(source, answer, mode="rag")`, with `[all]` installed and default thresholds. "Flagged" means `is_trustworthy == False`. Each row is 300 pairs that were not used for any design or tuning decision: HaluEval QA rows 300–599, and from HalluMix rows 600–1,199 the first 300 pairs whose source is under 6,000 characters.

| Dataset | Version | Precision | Recall | F1 | Faithful answers flagged |
|---------|---------|-----------|--------|----|--------------------------|
| HalluMix | 0.3.1 | 86.7% | 13.1% | 22.8% | 4 / 102 |
| HalluMix | 0.4.0 | 90.3% | 89.4% | 89.8% | 19 / 102 |
| HaluEval QA | 0.3.1 | 85.7% | 7.7% | 14.2% | 2 / 145 |
| HaluEval QA | 0.4.0 | 96.9% | 60.0% | 74.1% | 3 / 145 |

**What the threshold dial buys** (HalluMix, a separate 300-pair development slice):

| `fabrication` | Hallucinations caught | Faithful answers flagged |
|---------------|-----------------------|--------------------------|
| 0.05 | 82% | 14% |
| 0.2 (default) | 86% | 17% |
| 0.5 | 92% | 25% |

**Why the default model changed.** With the previous model (`cross-encoder/nli-deberta-v3-small`) the same pipeline caught 94% of HalluMix hallucinations but flagged 42% of faithful answers, and no threshold setting tried moved that below a third: its scores sit at 0 or 1. The current default is graded, so the dial works. The cost is on HaluEval, where recall fell from 72% to 60%, and a larger download (~740 MB against ~570 MB).

**Fragment answers and `question=`.** In a synthetic test (150 HaluEval questions whose correct answer is a fragment, each also paired with a different name taken from the same source), without the question every wrong name passed; with it 45% of wrong names were flagged, at the cost of flagging 9% of correct answers. Useful, far from solved.

Read all of this with care: 300 pairs per dataset is a small sample; none of it is LLM-AggreFact, the benchmark the field reports, which is gated and has not been run; and figures quoted for v0.2–v0.3.1 came from the mode-less kernel pipeline rather than `verify()`, so they are not comparable. Reproduce with `python bench/run_benchmark.py`, which now calls the public API.

## Limitations

This is an alpha release that has been through four rounds of independent adversarial auditing. Known limitations:

- **NLI is optional but critical.** Without it, entity/relation/unit swaps pass undetected. Install with `[nli]`.
- **Grounding still raises false alarms.** Sentence-level NLI flags some faithful answers as unsupported (see Benchmarks). Tune with `thresholds=`.
- **NLI model limits.** The default model has no world or unit knowledge, is weak on arithmetic and domain jargon, truncates inputs at 512 tokens, and can be fooled by adversarial phrasing. It runs at roughly one second per answer on two CPU cores.
- **Not measured on LLM-AggreFact.** That is the benchmark the field reports; the dataset is gated and has not been run. Numbers here are not comparable to leaderboard figures.
- **Atom typing is heuristic.** Sentence splitting and claim/evidence classification use regex patterns, not a trained model. Complex documents may misclassify atoms.
- **Alignment is quadratic in rewritten sentences.** Verbatim sentences align in linear time; the rest are compared pairwise. Past 1,000,000 comparisons (about 1,000 × 1,000 rewritten sentences) Criterion raises `InputTooLargeError` instead of hanging — verify long documents in sections or raise `CRITERION_MAX_COMPARISONS`.
- **Audit chain is symmetric-keyed at best.** There are no public-key signatures or built-in external anchoring; see "The record: what it proves".
- **Models are downloaded from the Hugging Face Hub.** The default models are pinned to fixed revisions. `CRITERION_NLI_MODEL` / `CRITERION_MODEL` load whatever you point them at, unpinned unless you also set `CRITERION_NLI_REVISION` / `CRITERION_MODEL_REVISION`.
- **Not validated for autonomous trust decisions.** Use as a review assistant with human oversight, not as a sole gate for medical, legal, or financial content.
- **English only.** All heuristic checks assume English text.

## Audit history

Criterion v0.2.0–v0.3.1 was independently audited by four AI systems across six rounds. The audits identified and drove fixes for: audit-chain integrity, RAG mode crashes, batch exit codes, empty input handling, currency extraction, NLI integration, antonym detection, numeric canonicalization, and mode-specific filtering policies. The auditors' consensus at v0.3.1: "experimental developer use: viable; controlled customer pilots: conditional; high-assurance commercial release: not approved."

v0.4.0 adds the fixes from a further red-team pass; see [CHANGELOG.md](CHANGELOG.md). That pass has not itself been independently reviewed.

## Requirements

- Python ≥ 3.10
- numpy
- sentence-transformers (optional, `[semantic]`)
- transformers + torch (optional, `[nli]` — strongly recommended)

## License

MIT
