"""
GitHub integration for Criterion.

Provides SARIF output for GitHub Code Scanning and helpers for
CI/CD pipelines (GitHub Actions, pre-commit hooks, etc.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from criterion.api import VerifyResult


def to_sarif(results: list[VerifyResult], tool_name: str = "criterion") -> dict[str, Any]:
    """Convert Criterion results to SARIF 2.1.0 format.

    SARIF (Static Analysis Results Interchange Format) is the standard
    format for GitHub Code Scanning. Upload the output to show Criterion
    findings in the Security tab of your repository.

    Args:
        results: List of VerifyResult objects.
        tool_name: Name of the tool in the SARIF output.

    Returns:
        A SARIF 2.1.0 JSON-serializable dictionary.
    """
    rules = {}
    sarif_results = []

    for result in results:
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
                "fixes": [
                    {
                        "description": {"text": d.suggested_correction[:200]},
                    }
                ] if d.suggested_correction else [],
                "properties": {
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
            })

    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": tool_name,
                        "version": "0.2.0",
                        "informationUri": "https://github.com/BayyinahEnterprise/criterion",
                        "rules": list(rules.values()),
                    },
                },
                "results": sarif_results,
            }
        ],
    }


def write_sarif(results: list[VerifyResult], path: str | Path) -> None:
    """Write SARIF output to a file."""
    sarif = to_sarif(results)
    Path(path).write_text(json.dumps(sarif, indent=2))


def _severity_to_sarif_level(severity: str) -> str:
    """Map Criterion severity to SARIF level."""
    return {
        "critical": "error",
        "major": "warning",
        "minor": "note",
        "cosmetic": "none",
    }.get(severity, "warning")
