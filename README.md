# Criterion

**Your AI's output looks right. Criterion checks whether it is.**

Criterion is a trust layer for AI-generated content. It decomposes text into semantic atoms, traces their relationships, and detects where an AI's output diverges from its source — not just that it changed, but *how* it changed and *what to fix*.

No API keys. No external services. Runs locally.

> **Status: Alpha (v0.3.1).** Suitable for developer evaluation and assisted review. Not yet validated for autonomous high-stakes trust decisions. See [Limitations](#limitations) for what it catches and what it misses.

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
print(result.summary)            # "2 divergences detected. 1 critical..."
```

## Three modes for three jobs

### RAG verification

> Is the LLM answer grounded in the retrieved documents?

```python
result = criterion.verify(source=retrieved_docs, output=llm_answer, mode="rag")
```

RAG mode is strict on **additions** (fabricated content) and **strengthening** (overclaiming), lenient on omissions — the answer doesn't need to repeat everything.

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

Everything counts — paraphrases, compression, emphasis shifts. Use for contracts, legal, medical, regulatory content.

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

### What it does NOT catch (known limitations)

Without the NLI model installed, Criterion **cannot detect**:
- Entity substitution (Alice → Bob) — same sentence structure, different names
- Relation reversal (sold-to vs bought-from) — same words, different direction
- Inflected antonyms (increased → decreased) — not in the static antonym table
- Unit/currency unit changes ($100 → €100) — symbols are stripped for numeric comparison

These require the NLI model, which classifies sentence pairs as entailment, contradiction, or neutral using a 140MB DeBERTa model.

**Even with NLI, Criterion does not catch:**
- Fabricated claims that are lexically unrelated to any source content and don't align with any source atom
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
```

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

## Limitations

This is an alpha release that has been through four rounds of independent adversarial auditing. Known limitations:

- **NLI is optional but critical.** Without it, entity/relation/unit swaps pass undetected. Install with `[nli]`.
- **NLI fallback is silent.** If the NLI model fails to load, Criterion falls back to heuristics without reporting the degradation in the result. A future release will add explicit capability reporting.
- **Atom typing is heuristic.** Sentence splitting and claim/evidence classification use regex patterns, not a trained model. Complex documents may misclassify atoms.
- **Alignment is greedy and quadratic.** O(n²) in atom count. Large documents (500+ sentences) may be slow. No chunking or truncation warning.
- **Audit chain is not cryptographically signed.** The hash chain detects accidental modification but not a determined attacker who can recompute hashes. Concurrent writers can corrupt the chain.
- **Not validated for autonomous trust decisions.** Use as a review assistant with human oversight, not as a sole gate for medical, legal, or financial content.
- **English only.** All heuristic checks assume English text.

## Audit history

Criterion v0.2.0–v0.3.1 was independently audited by four AI systems across six rounds. The audits identified and drove fixes for: audit-chain integrity, RAG mode crashes, batch exit codes, empty input handling, currency extraction, NLI integration, antonym detection, numeric canonicalization, and mode-specific filtering policies. The auditors' consensus at v0.3.1: "experimental developer use: viable; controlled customer pilots: conditional; high-assurance commercial release: not approved."

## Requirements

- Python ≥ 3.10
- numpy
- sentence-transformers (optional, `[semantic]`)
- transformers + torch (optional, `[nli]` — strongly recommended)

## License

MIT
