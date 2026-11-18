"""Shared test configuration.

Criterion fails closed when its grounding model is unavailable. So that the
deterministic checks can still be tested on a core-only install, the
suite opts into degraded mode when (and only when) NLI cannot load.
In that mode a clean result is "unverified", so tests that assert a
trustworthy verdict are skipped via the `needs_nli` marker.
"""

import os

import pytest


def nli_available() -> bool:
    """True when both models Criterion uses can load: the grounding scorer
    (which decides support) and the NLI model (which labels and serves
    strict mode)."""
    try:
        from criterion import grounding
        from criterion.nli import classify_pair
        grounding.support(["A cat sat."], ["A cat sat."])
        classify_pair("A cat sat.", "A cat sat.")
        return True
    except Exception:
        return False


NLI = nli_available()
if not NLI:
    os.environ.setdefault("CRITERION_REQUIRE_NLI", "0")

needs_nli = pytest.mark.skipif(not NLI, reason="NLI model not available")
