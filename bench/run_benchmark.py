"""
Criterion Benchmark Runner

Runs Criterion against hallucination detection datasets and measures
how well our divergence detection aligns with ground-truth labels.

Metrics:
    - Accuracy: does Criterion agree with the ground-truth label?
    - Precision: of the examples Criterion flags as divergent, how many are truly hallucinated?
    - Recall: of the truly hallucinated examples, how many does Criterion flag?
    - F1: harmonic mean of precision and recall
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



@dataclass
class BenchResult:
    dataset: str
    total: int
    true_positive: int    # Criterion says divergent, label says hallucinated
    true_negative: int    # Criterion says faithful, label says not hallucinated
    false_positive: int   # Criterion says divergent, label says not hallucinated
    false_negative: int   # Criterion says faithful, label says hallucinated
    errors: int
    elapsed_seconds: float

    @property
    def accuracy(self) -> float:
        n = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        return (self.true_positive + self.true_negative) / n if n else 0

    @property
    def precision(self) -> float:
        d = self.true_positive + self.false_positive
        return self.true_positive / d if d else 0

    @property
    def recall(self) -> float:
        d = self.true_positive + self.false_negative
        return self.true_positive / d if d else 0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0

    def report(self) -> str:
        return (
            f"\n{'=' * 60}\n"
            f"  CRITERION BENCHMARK: {self.dataset}\n"
            f"{'=' * 60}\n"
            f"  Examples:  {self.total} ({self.errors} errors skipped)\n"
            f"  Time:      {self.elapsed_seconds:.1f}s ({self.total / max(self.elapsed_seconds, 0.01):.0f} examples/sec)\n"
            f"\n"
            f"  Accuracy:  {self.accuracy:.1%}\n"
            f"  Precision: {self.precision:.1%}\n"
            f"  Recall:    {self.recall:.1%}\n"
            f"  F1:        {self.f1:.1%}\n"
            f"\n"
            f"  Confusion matrix:\n"
            f"                    Criterion: Divergent  Criterion: Faithful\n"
            f"  Label: Halluc.    {self.true_positive:>8}  TP        {self.false_negative:>8}  FN\n"
            f"  Label: Faithful   {self.false_positive:>8}  FP        {self.true_negative:>8}  TN\n"
            f"{'=' * 60}\n"
        )


def run_criterion_on_pair(source_text: str, output_text: str, mode: str = "rag") -> bool:
    """Run Criterion and return True if the output is flagged as divergent.

    Goes through the PUBLIC API (`criterion.verify`) in an explicit mode,
    with NLI required, so the published numbers describe what a user who
    calls `verify(..., mode=...)` actually gets. (Through v0.3.1 this
    called the kernel pipeline directly, which no mode of the public API
    reproduces, so the reported F1 did not describe the shipped product.)
    """
    from criterion.api import verify
    result = verify(source_text, output_text, mode=mode, record=False, require_nli=True)
    return not result.is_trustworthy  # True = flagged as divergent


def bench_hallumix(limit: int = 0) -> BenchResult:
    """Benchmark against HalluMix (quotientai/HalluMix)."""
    from datasets import load_dataset
    ds = load_dataset("quotientai/HalluMix", split="train")

    tp = tn = fp = fn = errors = 0
    n = min(len(ds), limit) if limit else len(ds)
    t0 = time.time()

    for i in range(n):
        ex = ds[i]
        try:
            # Build source from documents
            docs = ex.get("documents") or []
            source = " ".join(str(d) for d in docs).strip()
            if not source:
                errors += 1
                continue

            # Build output from answer
            answer = str(ex.get("answer", "")).strip()
            if not answer:
                errors += 1
                continue

            is_hallucinated = bool(ex["hallucination_label"])
            criterion_says_divergent = run_criterion_on_pair(source, answer)

            if criterion_says_divergent and is_hallucinated:
                tp += 1
            elif not criterion_says_divergent and not is_hallucinated:
                tn += 1
            elif criterion_says_divergent and not is_hallucinated:
                fp += 1
            else:
                fn += 1

        except Exception as e:
            errors += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  HalluMix: {i + 1}/{n} ({elapsed:.1f}s)", flush=True)

    return BenchResult(
        dataset="HalluMix",
        total=n,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        errors=errors,
        elapsed_seconds=time.time() - t0,
    )


def bench_halueval(limit: int = 0) -> BenchResult:
    """Benchmark against HaluEval (pminervini/HaluEval)."""
    from datasets import load_dataset

    # HaluEval has multiple configs; qa is the most comparable
    try:
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    except Exception:
        try:
            ds = load_dataset("pminervini/HaluEval", "qa_samples")
            ds = ds[list(ds.keys())[0]]
        except Exception as e:
            print(f"  HaluEval: could not load ({e})")
            return BenchResult("HaluEval", 0, 0, 0, 0, 0, 0, 0.0)

    print(f"  HaluEval loaded: {len(ds)} examples, columns: {ds.column_names}")

    tp = tn = fp = fn = errors = 0
    n = min(len(ds), limit) if limit else len(ds)
    t0 = time.time()

    for i in range(n):
        ex = ds[i]
        try:
            # HaluEval qa_samples: knowledge (source), answer (output),
            # hallucination = "yes"/"no" (label)
            knowledge = str(ex.get("knowledge", "")).strip()
            answer = str(ex.get("answer", "")).strip()
            hallu_label = str(ex.get("hallucination", "")).strip().lower()

            if not knowledge or not answer:
                errors += 1
                continue

            is_hallu = hallu_label == "yes"
            criterion_flag = run_criterion_on_pair(knowledge, answer)

            if criterion_flag and is_hallu:
                tp += 1
            elif not criterion_flag and not is_hallu:
                tn += 1
            elif criterion_flag and not is_hallu:
                fp += 1
            else:
                fn += 1

        except Exception:
            errors += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  HaluEval: {i + 1}/{n} ({elapsed:.1f}s)", flush=True)

    return BenchResult(
        dataset="HaluEval",
        total=n,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        errors=errors,
        elapsed_seconds=time.time() - t0,
    )


def bench_summeval(limit: int = 0) -> BenchResult:
    """Benchmark against SummEval (mteb/summeval).

    SummEval has 100 source texts, each with 16 machine summaries
    rated for consistency on a 1-5 scale by human judges.
    We threshold: consistency < 3.0 = hallucinated, >= 3.0 = faithful.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("mteb/summeval", split="test")
    except Exception as e:
        print(f"  SummEval: could not load ({e})")
        return BenchResult("SummEval", 0, 0, 0, 0, 0, 0, 0.0)

    print(f"  SummEval loaded: {len(ds)} source texts × ~16 summaries each")

    tp = tn = fp = fn = errors = 0
    total_pairs = 0
    t0 = time.time()

    for i, ex in enumerate(ds):
        source = ex["text"]
        summaries = ex["machine_summaries"]
        consistency_scores = ex["consistency"]

        pairs_this_doc = min(len(summaries), len(consistency_scores))
        if limit and total_pairs + pairs_this_doc > limit:
            pairs_this_doc = limit - total_pairs

        for j in range(pairs_this_doc):
            summary = summaries[j]
            score = consistency_scores[j]
            is_hallu = score < 3.0  # Low consistency = hallucinated

            try:
                criterion_flag = run_criterion_on_pair(source, summary, mode="summary")

                if criterion_flag and is_hallu:
                    tp += 1
                elif not criterion_flag and not is_hallu:
                    tn += 1
                elif criterion_flag and not is_hallu:
                    fp += 1
                else:
                    fn += 1
            except Exception:
                errors += 1

            total_pairs += 1

        if limit and total_pairs >= limit:
            break

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  SummEval: {total_pairs} pairs from {i + 1} docs ({elapsed:.1f}s)", flush=True)

    return BenchResult(
        dataset="SummEval",
        total=total_pairs,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        errors=errors,
        elapsed_seconds=time.time() - t0,
    )


def bench_hdmbench(limit: int = 0) -> BenchResult:
    """Benchmark against HDM-Bench (dataframer/HDM-Bench).

    HDM-Bench has context (source), response (output), and is_hallucination label.
    Uses the 'mr' (human-reviewed) split when available, else 'synthetic'.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("dataframer/HDM-Bench", split="mr")
    except Exception:
        try:
            ds = load_dataset("dataframer/HDM-Bench", split="synthetic")
        except Exception as e:
            print(f"  HDM-Bench: could not load ({e})")
            return BenchResult("HDM-Bench", 0, 0, 0, 0, 0, 0, 0.0)

    print(f"  HDM-Bench loaded: {len(ds)} examples, columns: {ds.column_names}")

    tp = tn = fp = fn = errors = 0
    n = min(len(ds), limit) if limit else len(ds)
    t0 = time.time()

    for i in range(n):
        ex = ds[i]
        try:
            context = str(ex.get("context", "")).strip()
            response = str(ex.get("response", "")).strip()
            is_hallu_raw = ex.get("is_hallucination")

            if not context or not response:
                errors += 1
                continue

            # Handle various label formats
            if isinstance(is_hallu_raw, bool):
                is_hallu = is_hallu_raw
            elif isinstance(is_hallu_raw, str):
                is_hallu = is_hallu_raw.lower() in ("yes", "true", "1")
            else:
                is_hallu = bool(is_hallu_raw)

            criterion_flag = run_criterion_on_pair(context, response)

            if criterion_flag and is_hallu:
                tp += 1
            elif not criterion_flag and not is_hallu:
                tn += 1
            elif criterion_flag and not is_hallu:
                fp += 1
            else:
                fn += 1

        except Exception:
            errors += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  HDM-Bench: {i + 1}/{n} ({elapsed:.1f}s)", flush=True)

    return BenchResult(
        dataset="HDM-Bench",
        total=n,
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        errors=errors,
        elapsed_seconds=time.time() - t0,
    )


def bench_guardrails(limit: int = 0) -> BenchResult:
    """Benchmark against GuardrailsAI/hallucination.

    Each example has an article (source), reference_summary (faithful),
    and edited_summary (hallucinated). We test both: the reference should
    pass, the edited should fail.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("GuardrailsAI/hallucination", split="train")
    except Exception as e:
        print(f"  GuardrailsAI: could not load ({e})")
        return BenchResult("GuardrailsAI", 0, 0, 0, 0, 0, 0, 0.0)

    print(f"  GuardrailsAI loaded: {len(ds)} examples, columns: {ds.column_names}")

    tp = tn = fp = fn = errors = 0
    n = min(len(ds), limit // 2 if limit else len(ds))
    t0 = time.time()

    for i in range(n):
        ex = ds[i]
        try:
            article = str(ex.get("article", "")).strip()
            ref = str(ex.get("reference_summary", "")).strip()
            edited = str(ex.get("edited_summary", "")).strip()

            if not article or not ref or not edited:
                errors += 1
                continue

            # Reference summary should be faithful (not flagged)
            ref_flag = run_criterion_on_pair(article, ref, mode="summary")
            if not ref_flag:
                tn += 1  # Correctly identified as faithful
            else:
                fp += 1  # Incorrectly flagged faithful summary

            # Edited summary should be hallucinated (flagged)
            edit_flag = run_criterion_on_pair(article, edited, mode="summary")
            if edit_flag:
                tp += 1  # Correctly caught hallucination
            else:
                fn += 1  # Missed hallucination

        except Exception:
            errors += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  GuardrailsAI: {i + 1}/{n} ({elapsed:.1f}s)", flush=True)

    return BenchResult(
        dataset="GuardrailsAI",
        total=n * 2,  # Each example produces 2 test cases
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        errors=errors,
        elapsed_seconds=time.time() - t0,
    )


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Criterion Benchmark Suite {'(limit=' + str(limit) + ')' if limit else '(full)'}")
    print()

    results = []

    print("1/5 HalluMix...")
    r = bench_hallumix(limit)
    print(r.report())
    results.append(r)

    print("2/5 HaluEval...")
    r = bench_halueval(limit)
    print(r.report())
    results.append(r)

    print("3/5 SummEval...")
    r = bench_summeval(limit)
    print(r.report())
    results.append(r)

    print("4/5 HDM-Bench...")
    r = bench_hdmbench(limit)
    print(r.report())
    results.append(r)

    print("5/5 GuardrailsAI...")
    r = bench_guardrails(limit)
    print(r.report())
    results.append(r)

    # Combined
    print("\n" + "=" * 60)
    print("  COMBINED RESULTS")
    print("=" * 60)
    for r in results:
        if r.total:
            print(f"  {r.dataset:15s}  Acc={r.accuracy:.1%}  P={r.precision:.1%}  R={r.recall:.1%}  F1={r.f1:.1%}  n={r.total}")
    total_tp = sum(r.true_positive for r in results)
    total_tn = sum(r.true_negative for r in results)
    total_fp = sum(r.false_positive for r in results)
    total_fn = sum(r.false_negative for r in results)
    total_n = total_tp + total_tn + total_fp + total_fn
    if total_n:
        acc = (total_tp + total_tn) / total_n
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f"  {'COMBINED':15s}  Acc={acc:.1%}  P={prec:.1%}  R={rec:.1%}  F1={f1:.1%}  n={total_n}")
    print("=" * 60)
