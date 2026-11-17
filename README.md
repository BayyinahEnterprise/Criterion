# Criterion

**Your AI's output looks right. Criterion checks whether it is.**

Criterion is a trust layer for AI-generated content. It decomposes text into semantic atoms, traces their relationships, and detects where an AI's output diverges from its source — not just that it changed, but *how* it changed and *what to fix*.

No API keys. No external services. Runs locally.

## Install

```bash
pip install criterion-ai            # core (word-level only)
pip install criterion-ai[semantic]   # + sentence embeddings for better accuracy
```

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
result = criterion.verify(
    source=retrieved_documents,
    output=llm_answer,
    mode="rag",
)

if not result.is_trustworthy:
    print("Hallucination detected:")
    for d in result.divergences:
        print(f"  {d.explanation}")
        print(f"  Fix: {d.suggested_correction}")
```

RAG mode is strict on **additions** (fabricated content not in the docs) and **strengthening** (overclaiming), but lenient on omissions — the answer doesn't need to repeat everything from the source.

### Summary verification

> Does this summary faithfully represent the source?

```python
result = criterion.verify(
    source=long_article,
    output=ai_summary,
    mode="summary",
)
```

Summary mode knows that compression and rewording are the job of a summary. It only flags changes that alter meaning: fabricated details, changed numbers, inverted negations, and dropped evidence for retained claims.

### Strict comparison

> Flag every deviation from the source.

```python
result = criterion.verify(
    source=original_contract,
    output=revised_contract,
    mode="strict",
)
```

Strict mode catches everything — including paraphrases, compression, and emphasis shifts. Use it for contracts, legal documents, medical records, and regulatory content.

## What it catches

Criterion classifies divergences into eight types on four paired axes:

| Axis | Inflating | Deflating |
|------|-----------|-----------|
| **Expansion** | Addition — content added | Omission — content removed |
| **Intensity** | Strengthening — claims made stronger | Weakening — claims made softer |
| **Emphasis** | Advancing — emphasis shifted forward | Deferring — emphasis buried |
| **Visibility** | Surfacing — implicit made explicit | Concealing — explicit hidden |

Plus two special detectors that run before everything else:
- **Numeric change** — catches "60%" → "90%" even when sentences are otherwise identical
- **Negation inversion** — catches "was not present" → "was present"

And structural analysis:
- **Broken bonds** — evidence dropped while claims retained (the claim is now unsupported)

## Every divergence is actionable

Every divergence carries two fields inspired by structured diagnostics:

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
criterion chain
```

Exit codes: `0` = trustworthy, `2` = untrustworthy. Use in CI:

```bash
criterion verify source.txt output.txt --quiet || echo "FAILED"
```

## CI / GitHub Actions

Add to your workflow:

```yaml
- name: Install Criterion
  run: pip install criterion-ai[semantic]

- name: Verify AI outputs
  run: criterion batch pairs.jsonl --mode rag -o results.jsonl
```

See `.github/workflows/criterion.yml` for a complete example.

## Python API

```python
import criterion

# Single verification
result = criterion.verify(source=src, output=out, mode="rag")

# Batch verification
results = criterion.verify_batch(
    pairs=[(src1, out1), (src2, out2)],
    mode="rag",
)

# JSON serialization
import json
print(json.dumps(result.to_dict(), indent=2))

# SARIF output for GitHub Code Scanning
from criterion.github import write_sarif
write_sarif(results, "criterion.sarif")
```

## How it works

Criterion decomposes text using an **Atom/Bond/Molecule** architecture:

1. **Capture** — Decompose source and output into typed semantic units (Atoms: claim, evidence, definition, reference, conclusion, metadata, observation), connected by typed relationships (Bonds: supports, contradicts, defines, cites, follows, elaborates, qualifies), composed into Molecules.

2. **Divergence** — Align source and output atoms by content similarity. Classify every difference by type (8 types) and severity (critical/major/minor/cosmetic). Detect broken bonds (evidence dropped while claims retained). Run numeric and negation checks before similarity gates.

3. **Verdict** — Score trustworthiness from divergence penalties with sublinear scaling. Classify as faithful (≥90%), adapted (≥70%), divergent (≥40%), or unreliable (<40%).

4. **Record** — Preserve every verification in an append-only JSONL file with a hash chain, so the audit trail is tamper-evident.

## Benchmark results (v0.2.0)

| Dataset | Type | Precision | Recall | F1 |
|---------|------|-----------|--------|----|
| HalluMix | Mixed hallucination | 57.6% | 70.7% | 63.4% |
| HaluEval | QA hallucination | 50.2% | 77.3% | 60.9% |
| GuardrailsAI | Factual errors in summaries | 49.9% | 97.0% | 65.9% |
| SummEval | Summary consistency | 10.5% | 68.8% | 18.3% |

Criterion achieves these results with **no ML training, no fine-tuning, and no API calls** — just structural decomposition, word-level heuristics, and a small sentence embedding model (all-MiniLM-L6-v2, 80MB).

SummEval's lower precision reflects the fundamental challenge of summarization: every summary omits and compresses, and distinguishing "valid compression" from "meaning-altering omission" requires deeper semantic understanding. The `summary` mode addresses this with more lenient thresholds.

## Requirements

- Python ≥ 3.10
- numpy
- sentence-transformers (optional, for semantic similarity)

## License

MIT
