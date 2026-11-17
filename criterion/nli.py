"""
Module 6: NLI (the Judge)

Natural Language Inference for claim-level contradiction detection.
This is the precision layer that catches what heuristics cannot:
numeric role swaps, entity substitutions, relation reversals,
inflected antonyms, and all cases where wording is similar but
meaning is reversed.

Uses tasksource/deberta-base-long-nli (Apache-2.0, ~740 MB) — classifies
each pair as entailment, contradiction, or neutral. It was chosen over
the previous default, cross-encoder/nli-deberta-v3-small, by measurement:
on HalluMix it halves the rate of faithful answers flagged, its scores
are graded rather than saturated at 0/1 (so thresholds actually move the
trade-off), and it can judge a fragment answer phrased with its question.
The small model remains available:
    CRITERION_NLI_MODEL=cross-encoder/nli-deberta-v3-small

Architecture:
    - Lazy model loading (first call loads, subsequent calls reuse)
    - Batch inference for efficiency
    - Three-class output: contradiction / entailment / neutral
    - Contradiction → CRITICAL divergence, always

Supply-chain note: the default model is pinned to a specific Hub
revision so a verdict cannot change because upstream weights moved.
Override with CRITERION_NLI_MODEL / CRITERION_NLI_REVISION.
"""

from __future__ import annotations

import os
from typing import NamedTuple, Sequence

# torch is imported lazily — not needed at module level
# This allows the module to be imported even without torch installed,
# so _USE_NLI in divergence.py can be set correctly and degrade gracefully.

_DEFAULT_NLI_MODEL = "tasksource/deberta-base-long-nli"
_DEFAULT_NLI_REVISION = "04dcf11f844b07bc57015169fca2b7d6df8299d5"
# Known-good revisions for models Criterion has been measured with. Any
# other model is unpinned unless CRITERION_NLI_REVISION is set.
_KNOWN_REVISIONS = {
    _DEFAULT_NLI_MODEL: _DEFAULT_NLI_REVISION,
    "cross-encoder/nli-deberta-v3-small": "fa2804872c3b4bd748f38c0185cc85775361e735",
}

_NLI_MODEL_NAME = os.environ.get("CRITERION_NLI_MODEL", _DEFAULT_NLI_MODEL)
_NLI_REVISION = os.environ.get(
    "CRITERION_NLI_REVISION", _KNOWN_REVISIONS.get(_NLI_MODEL_NAME, ""),
) or None

# The three classes. Their ORDER in the logits differs between checkpoints
# and is always read from the loaded model's config (_resolve_label_index).
_LABELS = ["contradiction", "entailment", "neutral"]

_model = None
_tokenizer = None
_label_index: dict[str, int] | None = None
_load_failed = False  # Cache load failures to avoid repeated slow attempts
_load_error: str | None = None


class NLIProbs(NamedTuple):
    """Class probabilities for one premise-hypothesis pair."""

    contradiction: float
    entailment: float
    neutral: float

    @property
    def label(self) -> str:
        best = max(range(3), key=lambda i: self[i])
        return ("contradiction", "entailment", "neutral")[best]

    @property
    def confidence(self) -> float:
        return max(self)


def _resolve_label_index(model) -> dict[str, int]:
    """Map label names to logit positions using the model's own config.

    NLI checkpoints do not share a label order (e.g. roberta-large-mnli is
    contradiction/neutral/entailment). Assuming a fixed order would silently
    swap "entailment" and "neutral" for any non-default model, so the order
    is read from id2label and the load fails closed if it cannot be resolved.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    index: dict[str, int] = {}
    for idx, name in id2label.items():
        key = str(name).strip().lower()
        for canonical in _LABELS:
            if key.startswith(canonical[:6]):  # contra / entail / neutra
                index[canonical] = int(idx)
    if set(index) == set(_LABELS) and len(set(index.values())) == 3:
        return index
    raise RuntimeError(
        f"NLI model {_NLI_MODEL_NAME!r} does not expose "
        f"contradiction/entailment/neutral labels (id2label={id2label!r})"
    )


def _load_nli():
    """Load the NLI model and tokenizer once, lazily.

    Returns (model, tokenizer) or raises ImportError/RuntimeError.
    Caches failures to avoid repeated slow attempts.
    """
    global _model, _tokenizer, _label_index, _load_failed, _load_error
    if _load_failed:
        raise RuntimeError(f"NLI model previously failed to load: {_load_error}")
    if _model is None:
        try:
            import torch  # noqa: F401 — lazy import
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(_NLI_MODEL_NAME, revision=_NLI_REVISION)
            model = AutoModelForSequenceClassification.from_pretrained(
                _NLI_MODEL_NAME, revision=_NLI_REVISION
            )
            model.eval()
            _label_index = _resolve_label_index(model)
            _model, _tokenizer = model, tokenizer
        except Exception as e:
            _load_failed = True
            _load_error = f"{type(e).__name__}: {e}"
            raise
    return _model, _tokenizer


def model_info() -> dict[str, str | None]:
    """Which NLI model this process is configured to use."""
    return {"model": _NLI_MODEL_NAME, "revision": _NLI_REVISION}


def classify_batch_probs(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    batch_size: int = 32,
) -> list[NLIProbs]:
    """Full class probabilities for multiple premise-hypothesis pairs."""
    if len(premises) != len(hypotheses):
        raise ValueError("premises and hypotheses must have the same length")
    if not premises:
        return []

    import torch
    model, tokenizer = _load_nli()
    li = _label_index
    if li is None:
        raise RuntimeError("NLI label order was not resolved")
    results: list[NLIProbs] = []

    for i in range(0, len(premises), batch_size):
        batch_p = list(premises[i:i + batch_size])
        batch_h = list(hypotheses[i:i + batch_size])
        inputs = tokenizer(
            batch_p, batch_h,
            return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        for j in range(len(batch_p)):
            row = probs[j]
            results.append(NLIProbs(
                contradiction=float(row[li["contradiction"]]),
                entailment=float(row[li["entailment"]]),
                neutral=float(row[li["neutral"]]),
            ))

    return results


def classify_batch(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """Classify multiple premise-hypothesis pairs in batches.

    Returns list of (label, confidence) tuples.
    """
    return [
        (p.label, p.confidence)
        for p in classify_batch_probs(premises, hypotheses, batch_size)
    ]


def classify_pair(premise: str, hypothesis: str) -> tuple[str, float]:
    """Classify a single premise-hypothesis pair.

    Returns (label, confidence) where label is one of
    'contradiction', 'entailment', 'neutral'.
    """
    return classify_batch([premise], [hypothesis])[0]


def is_contradiction(premise: str, hypothesis: str, threshold: float = 0.7) -> bool:
    """Quick check: does the hypothesis contradict the premise?"""
    label, conf = classify_pair(premise, hypothesis)
    return label == "contradiction" and conf >= threshold


def is_entailment(premise: str, hypothesis: str, threshold: float = 0.7) -> bool:
    """Quick check: does the premise entail the hypothesis?"""
    label, conf = classify_pair(premise, hypothesis)
    return label == "entailment" and conf >= threshold
