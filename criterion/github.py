"""
GitHub integration for Criterion.

Provides SARIF output for GitHub Code Scanning and helpers for
CI/CD pipelines (GitHub Actions, pre-commit hooks, etc.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from criterion._version import __version__
from criterion.api import VerifyResult


def to_sarif(
    results: list[VerifyResult],
    tool_name: str = "criterion",
    artifact_uris: list[str] | None = None,
) -> dict[str, Any]:
    """Convert Criterion results to SARIF 2.1.0 format.

    SARIF (Static Analysis Results Interchange Format) is the standard
    format for GitHub Code Scanning. Upload the output to show Criterion
    findings in the Security tab of your repository.

    Args:
        results: List of VerifyResult objects.
        tool_name: Name of the tool in the SARIF output.
        artifact_uris: Optional repo-relative path per result (the file
            holding that AI output). GitHub Code Scanning only displays
            results that have a location, so pass this when uploading.

    Returns:
        A SARIF 2.1.0 JSON-serializable dictionary.
    """
    rules = {}
    sarif_results = []

    if artifact_uris is not None and len(artifact_uris) != len(results):
        raise ValueError("artifact_uris must have one entry per result")

    def _locations(i: int) -> dict[str, Any]:
        if artifact_uris is None:
            return {}
        return {"locations": [{"physicalLocation": {
            "artifactLocation": {"uri": artifact_uris[i]},
            "region": {"startLine": 1},
        }}]}

    for idx, result in enumerate(results):
        for d in result.divergences:
            rule_id = f"criterion/{d.dtype.value}"
            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": d.dtype.value.replace("_", " ").title(),
                    "shortDescription": {"text": f"AI output {d.dtype.value}"},
                    "defaultConfiguration": {
                        "level": _severity_to_sarif_level(d.severity.value),
                    },
                }

            sarif_results.append({
                "ruleId": rule_id,
                "level": _severity_to_sarif_level(d.severity.value),
                "message": {
                    "text": d.explanation,
                },
                **_locations(idx),
                "properties": {
                    # SARIF `fixes` requires artifactChanges; a free-text
                    # suggestion belongs in properties.
                    "suggested_correction": d.suggested_correction[:200],
                    "code": d.code,
                    "nli_ran": bool(result.capabilities.get("nli")),
                    "confidence": d.confidence,
                    "verification_method": d.verification_method,
                    "trust_score": result.trust_score,
                    "trust_level": result.trust_level,
                },
            })

        # Also report broken bonds
        for bd in result.broken_bonds:
            rule_id = "criterion/broken-bond"
            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": "Broken Evidential Bond",
                    "shortDescription": {"text": "Evidence dropped while claim retained"},
                    "defaultConfiguration": {"level": "error"},
                }
            sarif_results.append({
                "ruleId": rule_id,
                "level": "error",
                "message": {"text": bd.explanation},
                **_locations(idx),
            })

    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": tool_name,
                        "version": __version__,
                        "informationUri": "https://github.com/BayyinahEnterprise/criterion",
                        "rules": list(rules.values()),
                    },
                },
                "results": sarif_results,
            }
        ],
    }


def write_sarif(
    results: list[VerifyResult],
    path: str | Path,
    artifact_uris: list[str] | None = None,
) -> None:
    """Write SARIF output to a file."""
    sarif = to_sarif(results, artifact_uris=artifact_uris)
    Path(path).write_text(json.dumps(sarif, indent=2), encoding="utf-8")


def _severity_to_sarif_level(severity: str) -> str:
    """Map Criterion severity to SARIF level."""
    return {
        "critical": "error",
        "major": "warning",
        "minor": "note",
        "cosmetic": "none",
    }.get(severity, "warning")
