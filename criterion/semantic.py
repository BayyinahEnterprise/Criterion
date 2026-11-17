"""
Module 5: Semantic (the Discerner)

Furqan reading order: 5 (first descent module, beyond the kernel)
Dependency depth: 1 — depends only on capture

Provides semantic similarity between atoms using sentence embeddings.
This module is the precision fix: it distinguishes faithful paraphrase
from unfaithful divergence by comparing MEANING rather than WORDS.

The kernel (modules 1–4) uses word-level similarity and catches 88%
of hallucinations but cries wolf 57% of the time. This module brings
precision up by telling the divergence engine: "these two sentences
use different words but mean the same thing — do not flag them."

Architecture:
    - Lazy model loading (first call loads, subsequent calls reuse)
    - Batch encoding for efficiency
    - Two thresholds:
        FAITHFUL_THRESHOLD = 0.75  — above this, the pair is faithful
        DIVERGENT_THRESHOLD = 0.35 — below this, the pair is divergent
        Between them: use word-level heuristics as tiebreaker
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

import numpy as np

from criterion.capture import Atom


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Semantic similarity thresholds
FAITHFUL_THRESHOLD = 0.75   # Above this: paraphrase, not a divergence
DIVERGENT_THRESHOLD = 0.35  # Below this: genuinely different meaning

# Model name (can be overridden via env var)
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
_MODEL_NAME = os.environ.get("CRITERION_MODEL", _DEFAULT_MODEL)
# Pinned so a verdict cannot change because upstream weights moved.
# The pin only applies to the default model; set CRITERION_MODEL_REVISION
# to pin a custom one.
_MODEL_REVISION = os.environ.get(
    "CRITERION_MODEL_REVISION",
    _DEFAULT_REVISION if _MODEL_NAME == _DEFAULT_MODEL else "",
) or None


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_model = None


def _get_model():
    """Load the sentence transformer model once, lazily."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME, revision=_MODEL_REVISION)
    return _model


def model_info() -> dict[str, str | None]:
    """Which embedding model this process is configured to use."""
    return {"model": _MODEL_NAME, "revision": _MODEL_REVISION}


def encode_texts(texts: Sequence[str]) -> np.ndarray:
    """Encode texts into L2-normalised embeddings, shape (n, dim)."""
    if not texts:
        return np.empty((0, 0))
    embs = _get_model().encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-9)


# ---------------------------------------------------------------------------
# Embedding and similarity
# ---------------------------------------------------------------------------

def encode_atoms(atoms: Sequence[Atom]) -> np.ndarray:
    """Encode a sequence of atoms into embedding vectors.

    Returns an (n, dim) numpy array where n = len(atoms).
    """
    model = _get_model()
    texts = [a.text for a in atoms]
    if not texts:
        return np.empty((0, 0))
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity between two text strings.

    Returns a float in [-1, 1], where 1 = identical meaning.
    In practice, positive values above 0.3 indicate related content.
    """
    model = _get_model()
    embs = model.encode([text_a, text_b], convert_to_numpy=True, show_progress_bar=False)
    # Cosine similarity
    a, b = embs[0], embs[1]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def batch_semantic_similarity(
    sources: Sequence[str],
    outputs: Sequence[str],
) -> list[float]:
    """Compute pairwise semantic similarity for aligned pairs.

    sources[i] is compared to outputs[i]. Returns a list of floats.
    More efficient than calling semantic_similarity() in a loop
    because all texts are encoded in one batch.
    """
    if not sources or not outputs:
        return []

    model = _get_model()
    all_texts = list(sources) + list(outputs)
    all_embs = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)

    n = len(sources)
    src_embs = all_embs[:n]
    out_embs = all_embs[n:]

    similarities = []
    for i in range(n):
        a, b = src_embs[i], out_embs[i]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        sim = float(np.dot(a, b) / denom) if denom > 0 else 0.0
        similarities.append(sim)

    return similarities


def is_faithful_paraphrase(source_text: str, output_text: str) -> bool:
    """Quick check: is the output a faithful paraphrase of the source?

    Returns True if semantic similarity is above FAITHFUL_THRESHOLD.
    """
    return semantic_similarity(source_text, output_text) >= FAITHFUL_THRESHOLD


def classify_semantic_distance(source_text: str, output_text: str) -> str:
    """Classify the semantic relationship between source and output.

    Returns one of:
        "faithful"  — meaning preserved (sim >= 0.75)
        "uncertain" — might be paraphrase or might be divergent (0.35–0.75)
        "divergent" — meaning changed (sim < 0.35)
    """
    sim = semantic_similarity(source_text, output_text)
    if sim >= FAITHFUL_THRESHOLD:
        return "faithful"
    if sim < DIVERGENT_THRESHOLD:
        return "divergent"
    return "uncertain"
