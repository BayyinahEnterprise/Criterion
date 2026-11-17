"""
Module 6: NLI (the Judge)

Natural Language Inference for claim-level contradiction detection.
This is the precision layer that catches what heuristics cannot:
numeric role swaps, entity substitutions, relation reversals,
inflected antonyms, and all cases where wording is similar but
meaning is reversed.

Uses cross-encoder/nli-deberta-v3-small (140MB) — classifies
each atom pair as entailment, contradiction, or neutral.

Architecture:
    - Lazy model loading (first call loads, subsequent calls reuse)
    - Batch inference for efficiency
    - Three-class output: contradiction / entailment / neutral
    - Contradiction → CRITICAL divergence, always
"""

from __future__ import annotations

import os
from typing import Sequence

# torch is imported lazily — not needed at module level
# This allows the module to be imported even without torch installed,
# so _USE_NLI in divergence.py can be set correctly and degrade gracefully.

_NLI_MODEL_NAME = os.environ.get(
    "CRITERION_NLI_MODEL", "cross-encoder/nli-deberta-v3-small"
)

# Labels in the order the model outputs them
_LABELS = ["contradiction", "entailment", "neutral"]

_model = None
_tokenizer = None
_load_failed = False  # Cache load failures to avoid repeated slow attempts


def _load_nli():
    """Load the NLI model and tokenizer once, lazily.

    Returns (model, tokenizer) or raises ImportError/RuntimeError.
    Caches failures to avoid repeated slow attempts.
    """
    global _model, _tokenizer, _load_failed
    if _load_failed:
        raise RuntimeError("NLI model previously failed to load")
    if _model is None:
        try:
            import torch  # noqa: F811 — lazy import
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            _tokenizer = AutoTokenizer.from_pretrained(_NLI_MODEL_NAME)
            _model = AutoModelForSequenceClassification.from_pretrained(_NLI_MODEL_NAME)
            _model.eval()
        except Exception as e:
            _load_failed = True
            raise
    return _model, _tokenizer


def classify_pair(premise: str, hypothesis: str) -> tuple[str, float]:
    """Classify a single premise-hypothesis pair.

    Returns (label, confidence) where label is one of
    'contradiction', 'entailment', 'neutral'.
    """
    import torch
    model, tokenizer = _load_nli()
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt", truncation=True, max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = probs.argmax().item()
    return _LABELS[idx], float(probs[idx])


def classify_batch(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """Classify multiple premise-hypothesis pairs in batches.

    Returns list of (label, confidence) tuples.
    """
    if not premises:
        return []

    import torch
    model, tokenizer = _load_nli()
    results = []

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
            idx = probs[j].argmax().item()
            results.append((_LABELS[idx], float(probs[j][idx])))

    return results


def is_contradiction(premise: str, hypothesis: str, threshold: float = 0.7) -> bool:
    """Quick check: does the hypothesis contradict the premise?"""
    label, conf = classify_pair(premise, hypothesis)
    return label == "contradiction" and conf >= threshold


def is_entailment(premise: str, hypothesis: str, threshold: float = 0.7) -> bool:
    """Quick check: does the premise entail the hypothesis?"""
    label, conf = classify_pair(premise, hypothesis)
    return label == "entailment" and conf >= threshold
