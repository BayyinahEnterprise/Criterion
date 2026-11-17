"""
Criterion — a trust layer for AI-generated content.

Your AI's output looks right. Criterion checks whether it is.

Quick start::

    import criterion

    result = criterion.verify(
        source="The contract expires on March 15, 2027.",
        output="The contract expires in early 2028.",
        mode="strict",
    )
    print(result.trust_level)     # "divergent"
    print(result.is_trustworthy)  # False — the date changed
    print(result.capabilities)    # which detectors actually ran
    for d in result.divergences:
        print(d.code, d.explanation)
        print(d.suggested_correction)

Modes::

    # RAG verification — is the LLM answer grounded in the retrieved docs?
    result = criterion.verify(source=docs, output=answer, mode="rag")

    # Summary verification — does this summary faithfully represent the source?
    result = criterion.verify(source=article, output=summary, mode="summary")

    # Strict comparison — flag any deviation, even paraphrases
    result = criterion.verify(source=original, output=revised, mode="strict")

Architecture (Furqan reading order):
    1. capture    — decompose source and output into Atoms, Bonds, Molecules
    2. divergence — enumerate every difference by type and severity
    3. verdict    — score the trustworthiness
    4. record     — preserve everything append-only with hash chain
"""

from criterion._version import __version__
from criterion.api import verify, verify_batch, VerifyResult, Mode, CapabilityError
from criterion.verdict import TrustLevel
from criterion.divergence import DivergenceType, Severity

__all__ = [
    "verify",
    "verify_batch",
    "CapabilityError",
    "VerifyResult",
    "Mode",
    "TrustLevel",
    "DivergenceType",
    "Severity",
    "__version__",
]
