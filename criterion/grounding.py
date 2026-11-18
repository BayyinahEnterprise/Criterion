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

    ensemble   minicheck + factcg, scores averaged             (default)
    minicheck  lytang/MiniCheck-RoBERTa-Large   MIT, ~1.4 GB
    factcg     yaxili96/FactCG-DeBERTa-v3-Large MIT, ~1.7 GB   (about 2x slower)
    nli        the configured NLI model, used chunk-and-max    (smaller, weaker)

Why two models by default. On 528 held-out LLM-AggreFact dev claims the
average of the two was 3.0 points more accurate than MiniCheck alone
(95% interval 0.8 to 5.3), and — more useful — their disagreement is
informative: where they agree the verdict was right 77% of the time,
where they disagree 46%, a coin flip. So `support` also reports each
model's own score, and the caller labels a sentence "unsure" when they
land on opposite sides of the threshold. Any combination can be named
explicitly with "+", e.g. CRITERION_SCORER=minicheck+nli.

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

SCORER_NAME = os.environ.get("CRITERION_SCORER", "ensemble").strip().lower()
BATCH_SIZE = int(os.environ.get("CRITERION_SCORER_BATCH", "8"))
# Cap on (batch size) x (tokens per input)^2: eight inputs of 512 tokens.
BATCH_TOKEN_SQUARES = int(os.environ.get("CRITERION_SCORER_BATCH_AREA", str(8 * 512 * 512)))
_ENSEMBLE = ("minicheck", "factcg")

_WORDISH = re.compile(r"\w+|[^\w\s]")
_SOURCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def member_names() -> tuple[str, ...]:
    """The scorers the configured name stands for, in order."""
    if SCORER_NAME == "ensemble":
        return _ENSEMBLE
    names = tuple(n.strip() for n in SCORER_NAME.split("+") if n.strip())
    unknown = [n for n in names if n not in _SPECS]
    if unknown or not names or len(set(names)) != len(names):
        raise ValueError(
            f"unknown CRITERION_SCORER {SCORER_NAME!r}; expected 'ensemble', one of "
            f"{sorted(_SPECS)}, or several of them joined with '+'"
        )
    return names


class Support(tuple):
    """(score, best_chunk) with each model's own score in `.members`.

    A plain 2-tuple to every existing caller; `.members` maps scorer name to
    that model's best score, and is what a caller needs to tell agreement
    from disagreement.
    """

    members: dict[str, float]

    def __new__(cls, score: float, chunk: str, members: dict[str, float] | None = None):
        self = super().__new__(cls, (score, chunk))
        self.members = dict(members or {})
        return self


class _Scorer:
    """One loaded grounding model with its own chunking and prompt."""

    def __init__(self, name: str):
        import torch  # noqa: F401 — lazy import
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.name = name
        self.spec = _SPECS[name]
        if name == "nli":
            from criterion import nli
            self.model, self.tokenizer = nli._load_nli()
            self.entail_index = nli._label_index["entailment"]
        else:
            spec = self.spec
            config = AutoConfig.from_pretrained(spec["model"], revision=spec["revision"], num_labels=2)
            config.problem_type = "single_label_classification"
            self.tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"], use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                spec["model"], revision=spec["revision"], config=config,
                use_safetensors=True,          # never fall back to a pickle
            )
            self.model.eval()
            self.entail_index = 1              # label 1 = supported, in both models

    def count(self, text: str) -> int:
        """Token count in the unit this scorer's chunk size is defined in."""
        if self.name == "minicheck":
            # No truncation here: a truncated count would make an over-long
            # run of text look as if it fits.
            return len(self.tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])
        if self.name == "factcg":
            return len(_WORDISH.findall(text))
        return len(text.split())

    def build_chunks(self, source: str | Sequence[str], reserve: int = 0) -> list[str]:
        limit = self.spec["chunk_tokens"]
        if self.name == "minicheck":
            limit = max(64, min(limit, self.spec["max_length"] - reserve - 4))
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
            n = self.count(piece)
            if n > limit:
                # Too long to be one sentence (tokenised or unpunctuated
                # text): cut by words into runs that fit.
                flush()
                words = piece.split()
                step = max(8, int(limit * len(words) / n * 0.9))
                k = 0
                while k < len(words):
                    size = step
                    while size > 8 and self.count(" ".join(words[k:k + size])) > limit:
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

    def _encode(self, chunks: list[str], claims: list[str]):
        spec, tok = self.spec, self.tokenizer
        if self.name == "minicheck":
            texts = [tok.eos_token.join([c, h]) for c, h in zip(chunks, claims)]
            return tok(texts, max_length=spec["max_length"], truncation=True, padding=True, return_tensors="pt")
        if self.name == "factcg":
            texts = [_FACTCG_TEMPLATE.format(text_a=c, text_b=h) for c, h in zip(chunks, claims)]
            return tok(texts, max_length=spec["max_length"], truncation=True, padding="longest", return_tensors="pt")
        return tok(
            chunks, claims, max_length=spec["max_length"], truncation="only_first",
            padding=True, return_tensors="pt",
        )

    def _batches(self, order: list[int], chunks: Sequence[str], claims: Sequence[str]):
        """Split `order` (shortest first) into batches bounded by count AND size.

        Attention memory grows with batch x length squared. Eight inputs of
        2,048 tokens through a large model need several gigabytes, which
        killed the process on an 8 GB machine; long inputs therefore go in
        small batches, down to one at a time.
        """
        limit = self.spec["max_length"]
        batch: list[int] = []
        for i in order:
            # Rough token estimate; exactness is not needed for a memory bound.
            tokens = min(limit, (len(chunks[i]) + len(claims[i])) // 3 + 16)
            fits = len(batch) < BATCH_SIZE and (len(batch) + 1) * tokens ** 2 <= BATCH_TOKEN_SQUARES
            if batch and not fits:
                yield batch
                batch = []
            batch.append(i)
        if batch:
            yield batch

    def score_pairs(self, chunks: Sequence[str], claims: Sequence[str]) -> list[float]:
        if len(chunks) != len(claims):
            raise ValueError("chunks and claims must have the same length")
        if not chunks:
            return []
        import torch
        # Batch similar lengths together: padding to the longest wastes quadratic work.
        order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]) + len(claims[i]))
        out = [0.0] * len(chunks)
        for idx in self._batches(order, chunks, claims):
            inputs = self._encode([chunks[i] for i in idx], [claims[i] for i in idx])
            with torch.no_grad():
                probs = torch.softmax(self.model(**inputs).logits, dim=-1)[:, self.entail_index]
            for i, p in zip(idx, probs.tolist()):
                out[i] = float(p)
        return out

    def support(self, source: str | Sequence[str], claims: Sequence[str]) -> list[tuple[float, str]]:
        longest_claim = max(self.count(c) for c in claims) if self.name == "minicheck" else 0
        chunks = self.build_chunks(source, reserve=min(longest_claim, 200))
        if not chunks:
            return [(0.0, "")] * len(claims)
        pair_chunks = [c for _ in claims for c in chunks]
        pair_claims = [h for h in claims for _ in chunks]
        scores = self.score_pairs(pair_chunks, pair_claims)
        result: list[tuple[float, str]] = []
        for k in range(len(claims)):
            row = scores[k * len(chunks):(k + 1) * len(chunks)]
            best = max(range(len(row)), key=row.__getitem__)
            result.append((row[best], chunks[best]))
        return result


_scorers: list[_Scorer] | None = None
_load_failed = False
_load_error: str | None = None


def scorer_info() -> dict[str, Any]:
    """Which grounding scorer(s) this process is configured to use."""
    try:
        names = member_names()
    except ValueError:
        return {"scorer": SCORER_NAME, "model": None, "revision": None}

    def one(name: str) -> dict[str, Any]:
        if name == "nli":
            from criterion.nli import model_info
            return {"scorer": "nli", **model_info()}
        spec = _SPECS[name]
        return {"scorer": name, "model": spec.get("model"), "revision": spec.get("revision")}

    if len(names) == 1:
        return one(names[0])
    return {"scorer": SCORER_NAME, "combine": "mean", "members": [one(n) for n in names]}


def _load() -> list[_Scorer]:
    """Load every configured scorer once. Raises on failure; failures are cached.

    All or nothing: if one member of an ensemble cannot load, the scorer as a
    whole has failed. Quietly carrying on with the other would be a weaker
    check presented as the configured one.
    """
    global _scorers, _load_failed, _load_error
    if _load_failed:
        raise RuntimeError(f"grounding scorer previously failed to load: {_load_error}")
    if _scorers is not None:
        return _scorers
    try:
        _scorers = [_Scorer(name) for name in member_names()]
    except Exception as e:
        _scorers = None
        _load_failed = True
        _load_error = f"{type(e).__name__}: {e}"
        raise
    return _scorers


def _count(text: str) -> int:
    """Token count as the first configured scorer counts it."""
    return _load()[0].count(text)


def build_chunks(source: str | Sequence[str], reserve: int = 0) -> list[str]:
    """Pack the source into consecutive chunks of the first scorer's size.

    `source` is the raw source text (preferred) or a list of sentences.
    Chunks break at sentence ends where possible; a single run of text
    longer than a chunk (tokenised or unpunctuated input) is cut by length,
    so nothing in the source is silently truncated away.

    `reserve` shrinks the chunk so that chunk + claim still fits the model's
    window (MiniCheck's own code truncates the END of the input, i.e. the
    claim, when they do not fit; reserving room avoids that).
    """
    return _load()[0].build_chunks(source, reserve)


def score_pairs(chunks: Sequence[str], claims: Sequence[str]) -> list[float]:
    """P(supported) for each (chunk, claim) pair, by the first configured scorer."""
    return _load()[0].score_pairs(chunks, claims)


def support(source: str | Sequence[str], claims: Sequence[str]) -> list[Support]:
    """Best support for each claim over all chunks of the source.

    Returns one Support per claim: (score, best_chunk), score in [0, 1],
    with each model's own score in `.members`. Every claim is scored against
    every chunk — the published protocol — so cost grows with
    (number of claims) x (source length) x (number of models). With several
    models the score is their mean and the chunk is the first model's.
    """
    if not claims:
        return []
    scorers = _load()
    per_model = [s.support(source, claims) for s in scorers]
    result: list[Support] = []
    for k in range(len(claims)):
        members = {s.name: per_model[j][k][0] for j, s in enumerate(scorers)}
        score = sum(members.values()) / len(members)
        result.append(Support(score, per_model[0][k][1], members))
    return result
