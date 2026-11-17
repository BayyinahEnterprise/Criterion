# Criterion

**Your AI's output looks right. Criterion checks whether it is.**

Criterion is a trust layer for AI-generated content. It decomposes text into semantic atoms, traces their relationships, and detects where an AI's output diverges from its source — not just that it changed, but *how* it changed and *what to fix*.

No API keys. No external services. Runs locally.

> **Status: Alpha (v0.4.0).** Suitable for developer evaluation and assisted review. Not yet validated for autonomous high-stakes trust decisions. See [Limitations](#limitations) for what it catches and what it misses.

## Install

```bash
pip install criterion-ai                # core: numeric, negation, structural checks
pip install criterion-ai[nli]           # + NLI contradiction detection (recommended)
pip install criterion-ai[all]           # + NLI + sentence embeddings
```

**Which install do you need?**

| Install | Size | Catches |
|---------|------|---------|
| `criterion-ai` | ~2 MB | Changed numbers, currency ($1K→$2K), negation inversion, prefix antonyms (safe→unsafe), broken evidence bonds, additions, omissions |
| `criterion-ai[nli]` | ~500 MB | Everything above **plus** entity swaps (Alice→Bob), relation reversals (sold-to→bought-from), predicate swaps, inflected antonyms (increased→decreased), unit changes (mg→g), location substitution |
| `criterion-ai[all]` | ~600 MB | Everything above + sentence-embedding similarity for paraphrase detection |

**The `[nli]` install is strongly recommended.** Without it, Criterion cannot detect meaning-level contradictions where the words are similar but the facts are reversed.

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

**Check `result.capabilities["nli"]`.** If it is `False`, contradictions and fabricated claims were not checked and a clean verdict means much less. For any automated gate, pass `require_nli=True` (or set `CRITERION_REQUIRE_NLI=1`, or `--require-nli` on the CLI) so a missing model is an error instead of a quiet downgrade.

## Three modes for three jobs

### RAG verification

> Is the LLM answer grounded in the retrieved documents?

```python
result = criterion.verify(source=retrieved_docs, output=llm_answer, mode="rag")
```

RAG mode is strict on **additions** (fabricated content) and **strengthening** (overclaiming), lenient on omissions — the answer doesn't need to repeat everything. With NLI installed, every answer sentence is checked for support against the most relevant source passages: unsupported sentences are flagged, sentences a passage contradicts are critical, and paraphrases the aligner missed are recognised as supported.

Bare fragment answers (`"Delhi"`, `"yes"`) are not statements, so they cannot be verified against a source without the question. Criterion checks only that a fragment's words occur in the source. Pass full-sentence answers for a real check.

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

These require the NLI model, which classifies sentence pairs as entailment, contradiction, or neutral using a 140MB DeBERTa model.

**Even with NLI, Criterion does not catch:**
- Fragment answers that name something the source mentions but that is the wrong answer to the question (the question is not an input)
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

# CI gate: explicit mode, fail if the NLI model cannot run
criterion batch pairs.jsonl --mode rag --require-nli -o results.jsonl

# Keep hashes only (no source/output text on disk)
criterion verify source.txt output.txt --no-store-text
```

Exit codes: `0` = trustworthy, `2` = untrustworthy, `1` = error.

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

Measured through the public API, `criterion.verify(source, answer, mode="rag")`, with `[all]` installed. "Flagged" means `is_trustworthy == False`. Rows 300–599 of each dataset (300 pairs each); rows 0–299 were looked at while building v0.4.0 and are not reported.

| Dataset | Version | Precision | Recall | F1 | Faithful answers flagged |
|---------|---------|-----------|--------|----|--------------------------|
| HaluEval QA | 0.3.1 | 85.7% | 7.7% | 14.2% | 2 / 145 |
| HaluEval QA | 0.4.0 | 96.5% | 71.6% | 82.2% | 4 / 145 |
| HalluMix | 0.3.1 | 86.7% | 13.1% | 22.8% | 4 / 102 |
| HalluMix | 0.4.0 | 81.3% | 94.4% | 87.4% | 43 / 102 |

Read these with care: 300 pairs per dataset is a small sample; HaluEval answers are fragments, which Criterion can only locate in the source, not verify; and on HalluMix the gain in recall is paid for with false alarms on 42% of faithful answers. Figures quoted for v0.2–v0.3.1 came from the mode-less kernel pipeline rather than `verify()`, and are not comparable. Reproduce with `python bench/run_benchmark.py`, which now calls the public API.

## Limitations

This is an alpha release that has been through four rounds of independent adversarial auditing. Known limitations:

- **NLI is optional but critical.** Without it, entity/relation/unit swaps pass undetected. Install with `[nli]`.
- **NLI fallback is reported, not blocked, by default.** If the NLI model is missing or fails, `result.capabilities["nli"]` is `False` and the summary says so, but `is_trustworthy` can still be `True`. Use `require_nli=True` to make it an error.
- **Grounding costs precision.** Sentence-level NLI flags some faithful answers as unsupported (see Benchmarks). The thresholds are constants in `criterion/divergence.py`.
- **NLI model limits.** A 140 MB model is weak on arithmetic, long sentences (inputs are truncated at 512 tokens) and domain jargon, and can be fooled by adversarial phrasing.
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
