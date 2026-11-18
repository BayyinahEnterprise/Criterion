"""
Module 7: Grounding (the Witness)

Answers one question per output sentence: does the source support it?

The method is the one the fact-checking literature converged on
(SummaC 2021, AlignScore 2023, MiniCheck 2024, FactCG 2025):

    1. Cut the source into chunks of a few hundred tokens, at sentence
       boundaries.
    2. Score the sentence against every chunk with a model trained for
       document-grounded fact-checking.
    3. The sentence's support is the MAXIMUM over chunks.

A generic NLI model is the wrong tool for step 2: MiniCheck's ablation
puts an NLI-only model about fifteen points below one trained on
grounding data, and Criterion's own measurements on LLM-AggreFact agree.
So the scorer here is a grounding model, and the NLI model (nli.py) is
kept only to label WHY an unsupported sentence fails (contradiction or
no support) and for strict-mode omission checks.

Scorers (select with CRITERION_SCORER):

    minicheck  lytang/MiniCheck-RoBERTa-Large   MIT, ~1.4 GB   (default)
    factcg     yaxili96/FactCG-DeBERTa-v3-Large MIT, ~1.7 GB   (about 2x slower)
    nli        the configured NLI model, used chunk-and-max    (smaller, weaker)

All are pinned to fixed Hub revisions and loaded from safetensors only,
so no pickle is ever deserialised. The MiniCheck revision is the Hub's
safetensors conversion of the authors' weights (their repository ships a
pickle); every tensor in it was verified identical to the original.

Each scorer returns P(supported) in [0, 1]. They output supported /
unsupported only — there is no contradiction class.
"""

from __future__ import annotations

import os
import re
from typing import Any, Sequence

_SPECS: dict[str, dict[str, Any]] = {
    "minicheck": {
        "model": "lytang/MiniCheck-RoBERTa-Large",
        # refs/pr/1: safetensors conversion, verified tensor-for-tensor
        # against pytorch_model.bin @ 74c8919647e61ed0f71bc177d94f10930f090068
        "revision": "b773ffe285110d560f3261ad1f0183f14a954d5f",
        "chunk_tokens": 400,
        "max_length": 512,
    },
    "factcg": {
        "model": "yaxili96/FactCG-DeBERTa-v3-Large",
        "revision": "0430e3509dbd28d2dff7a117c0eae25359ff3e80",
        "chunk_tokens": 550,     # counted in words, as in the FactCG repository
        "max_length": 2048,
    },
    "nli": {"chunk_tokens": 350, "max_length": 1024},
}

_FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above can we conclude "
    "that \"{text_b}\"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is "
)

SCORER_NAME = os.environ.get("CRITERION_SCORER", "minicheck").strip().lower()
BATCH_SIZE = int(os.environ.get("CRITERION_SCORER_BATCH", "8"))

_model = None
_tokenizer = None
_entail_index: int | None = None
_load_failed = False
_load_error: str | None = None

_WORDISH = re.compile(r"\w+|[^\w\s]")


def scorer_info() -> dict[str, Any]:
    """Which grounding scorer this process is configured to use."""
    spec = _SPECS.get(SCORER_NAME, {})
    if SCORER_NAME == "nli":
        from criterion.nli import model_info
        return {"scorer": "nli", **model_info()}
    return {"scorer": SCORER_NAME, "model": spec.get("model"), "revision": spec.get("revision")}


def _load():
    """Load the configured scorer once. Raises on failure; failures are cached."""
    global _model, _tokenizer, _entail_index, _load_failed, _load_error
    if _load_failed:
        raise RuntimeError(f"grounding scorer previously failed to load: {_load_error}")
    if _model is not None:
        return _model, _tokenizer
    try:
        if SCORER_NAME not in _SPECS:
            raise ValueError(
                f"unknown CRITERION_SCORER {SCORER_NAME!r}; expected one of {sorted(_SPECS)}"
            )
        import torch  # noqa: F401 — lazy import
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        if SCORER_NAME == "nli":
            from criterion import nli
            model, tokenizer = nli._load_nli()
            _entail_index = nli._label_index["entailment"]
        else:
            spec = _SPECS[SCORER_NAME]
            config = AutoConfig.from_pretrained(spec["model"], revision=spec["revision"], num_labels=2)
            config.problem_type = "single_label_classification"
            tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"], use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                spec["model"], revision=spec["revision"], config=config,
                use_safetensors=True,          # never fall back to a pickle
            )
            model.eval()
            _entail_index = 1                  # label 1 = supported, in both models
        _model, _tokenizer = model, tokenizer
    except Exception as e:
        _load_failed = True
        _load_error = f"{type(e).__name__}: {e}"
        raise
    return _model, _tokenizer


def _count(text: str) -> int:
    """Token count in the unit the scorer's chunk size is defined in."""
    if SCORER_NAME == "minicheck":
        # No truncation here: a truncated count would make an over-long run
        # of text look as if it fits.
        return len(_tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])
    if SCORER_NAME == "factcg":
        return len(_WORDISH.findall(text))
    return len(text.split())


_SOURCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def build_chunks(source: str | Sequence[str], reserve: int = 0) -> list[str]:
    """Pack the source into consecutive chunks of the scorer's size.

    `source` is the raw source text (preferred) or a list of sentences.
    Chunks break at sentence ends where possible; a single run of text
    longer than a chunk (tokenised or unpunctuated input) is cut by length,
    so nothing in the source is silently truncated away.

    `reserve` shrinks the chunk so that chunk + claim still fits the model's
    window (MiniCheck's own code truncates the END of the input, i.e. the
    claim, when they do not fit; reserving room avoids that).
    """
    _load()
    spec = _SPECS[SCORER_NAME]
    limit = spec["chunk_tokens"]
    if SCORER_NAME == "minicheck":
        limit = max(64, min(limit, spec["max_length"] - reserve - 4))
    pieces = _SOURCE_SPLIT.split(source) if isinstance(source, str) else list(source)
    chunks: list[str] = []
    current: list[str] = []
    used = 0

    def flush() -> None:
        nonlocal current, used
        if current:
            chunks.append(" ".join(current))
        current, used = [], 0

    for piece in pieces:
        piece = (piece or "").strip()
        if not piece:
            continue
        n = _count(piece)
        if n > limit:
            # Too long to be one sentence (tokenised or unpunctuated text):
            # cut by words into runs that fit.
            flush()
            words = piece.split()
            step = max(8, int(limit * len(words) / n * 0.9))
            k = 0
            while k < len(words):
                size = step
                while size > 8 and _count(" ".join(words[k:k + size])) > limit:
                    size //= 2
                chunks.append(" ".join(words[k:k + size]))
                k += size
            continue
        if current and used + n > limit:
            flush()
        current.append(piece)
        used += n
    flush()
    return chunks


def _encode(chunks: list[str], claims: list[str]):
    spec = _SPECS[SCORER_NAME]
    if SCORER_NAME == "minicheck":
        texts = [_tokenizer.eos_token.join([c, h]) for c, h in zip(chunks, claims)]
        return _tokenizer(texts, max_length=spec["max_length"], truncation=True, padding=True, return_tensors="pt")
    if SCORER_NAME == "factcg":
        texts = [_FACTCG_TEMPLATE.format(text_a=c, text_b=h) for c, h in zip(chunks, claims)]
        return _tokenizer(texts, max_length=spec["max_length"], truncation=True, padding="longest", return_tensors="pt")
    return _tokenizer(
        chunks, claims, max_length=spec["max_length"], truncation="only_first",
        padding=True, return_tensors="pt",
    )


def score_pairs(chunks: Sequence[str], claims: Sequence[str]) -> list[float]:
    """P(supported) for each (chunk, claim) pair."""
    if len(chunks) != len(claims):
        raise ValueError("chunks and claims must have the same length")
    if not chunks:
        return []
    import torch
    model, _ = _load()
    # Batch similar lengths together: padding to the longest wastes quadratic work.
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]) + len(claims[i]))
    out = [0.0] * len(chunks)
    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        inputs = _encode([chunks[i] for i in idx], [claims[i] for i in idx])
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[:, _entail_index]
        for i, p in zip(idx, probs.tolist()):
            out[i] = float(p)
    return out


def support(source: str | Sequence[str], claims: Sequence[str]) -> list[tuple[float, str]]:
    """Best support for each claim over all chunks of the source.

    Returns one (score, best_chunk) per claim, score in [0, 1]. Every claim
    is scored against every chunk — the published protocol — so cost grows
    with (number of claims) x (source length).
    """
    if not claims:
        return []
    _load()
    longest_claim = max(_count(c) for c in claims) if SCORER_NAME == "minicheck" else 0
    chunks = build_chunks(source, reserve=min(longest_claim, 200))
    if not chunks:
        return [(0.0, "")] * len(claims)
    pair_chunks = [c for _ in claims for c in chunks]
    pair_claims = [h for h in claims for _ in chunks]
    scores = score_pairs(pair_chunks, pair_claims)
    result: list[tuple[float, str]] = []
    for k in range(len(claims)):
        row = scores[k * len(chunks):(k + 1) * len(chunks)]
        best = max(range(len(row)), key=row.__getitem__)
        result.append((row[best], chunks[best]))
    return result
