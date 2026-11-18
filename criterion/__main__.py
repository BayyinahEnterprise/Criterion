"""
Criterion CLI — verify AI-generated content against its source.

Usage:
    criterion verify source.txt output.txt
    criterion verify source.txt output.txt --mode rag
    criterion verify source.txt output.txt --mode summary --json
    criterion batch pairs.jsonl --mode rag -o results.jsonl
    criterion chain [--record PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import os

from criterion.api import verify, Mode, CapabilityError
from criterion.divergence import InputTooLargeError
from criterion.record import verify_chain, chain_head, RecordCorruptError, _KEY_ENV


def _read_file_or_text(path_or_text: str) -> str:
    """Read from a file path if it exists, otherwise treat as literal text."""
    try:
        p = Path(path_or_text)
        if p.is_file():
            return p.read_text(encoding="utf-8")
    except (OSError, ValueError):
        pass  # too long / not a valid path: it is literal text
    return path_or_text


def _add_policy_flags(sub_parser) -> None:
    sub_parser.add_argument(
        "--allow-degraded", action="store_true",
        help="Run heuristics only if the NLI model is unavailable. Clean results are then "
             "'unverified' (exit 2), never trustworthy. Without this flag a missing model is an error.",
    )
    sub_parser.add_argument(
        "--require-nli", action="store_true",
        help="(Default behaviour; kept for compatibility.) Fail if the NLI model cannot run.",
    )
    sub_parser.add_argument(
        "--fabrication-threshold", type=float, default=None, metavar="0-1",
        help="An unsupported sentence blocks trust on its own when its best source support is "
             "below this. Higher flags more, lower flags less.",
    )
    sub_parser.add_argument(
        "--fail-on-unsure", action="store_true",
        help="Exit 3 when the verdict is trustworthy but the two grounding models disagree on "
             "some sentence. Without this flag such a result exits 0 and is only labelled unsure.",
    )
    sub_parser.add_argument(
        "--no-store-text", action="store_true",
        help="Record only SHA-256 hashes of source/output, not the texts.",
    )


def _policy_kwargs(args) -> dict:
    kwargs = {
        "require_nli": False if args.allow_degraded else (True if args.require_nli else None),
        "store_text": not args.no_store_text,
    }
    if args.fabrication_threshold is not None:
        kwargs["thresholds"] = {"fabrication": args.fabrication_threshold}
    return kwargs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="criterion",
        description="Criterion — a trust layer for AI-generated content.",
        epilog="Your AI's output looks right. Criterion checks whether it is.",
    )
    sub = parser.add_subparsers(dest="command")

    # verify
    verify_p = sub.add_parser("verify", help="Verify an AI output against its source.")
    verify_p.add_argument("source", help="Source file path or text.")
    verify_p.add_argument("output", help="AI output file path or text.")
    verify_p.add_argument(
        "--mode", "-m",
        choices=["rag", "summary", "strict", "auto"],
        default="auto",
        help="Verification mode (default: auto-detect).",
    )
    verify_p.add_argument("--record", default=None, help="Path to the record file.")
    verify_p.add_argument("--no-record", action="store_true", help="Skip recording.")
    verify_p.add_argument("--json", action="store_true", help="Output as JSON.")
    verify_p.add_argument("--quiet", "-q", action="store_true", help="Exit code only.")
    _add_policy_flags(verify_p)

    # chain
    chain_p = sub.add_parser("chain", help="Verify the append-only record chain.")
    chain_p.add_argument("--record", default=None, help="Path to the record file.")
    chain_p.add_argument("--expect-head", default=None,
                         help="Head entry id you anchored elsewhere; fail if the ledger does not end there.")
    chain_p.add_argument("--expect-count", type=int, default=None,
                         help="Entry count you anchored elsewhere; fail on mismatch (detects truncation).")

    # batch
    batch_p = sub.add_parser("batch", help="Verify multiple pairs from a JSONL file.")
    batch_p.add_argument("input_file", help="JSONL with {source, output} per line.")
    batch_p.add_argument("--mode", "-m", choices=["rag", "summary", "strict", "auto"], default="auto")
    batch_p.add_argument("--output", "-o", help="Output JSONL file.")
    batch_p.add_argument("--no-record", action="store_true")
    batch_p.add_argument("--record", default=None, help="Path to the record file.")
    _add_policy_flags(batch_p)

    args = parser.parse_args(argv)

    if args.command is None or args.command == "verify":
        if not hasattr(args, "source"):
            parser.print_help()
            return 1

        source_text = _read_file_or_text(args.source)
        output_text = _read_file_or_text(args.output)
        try:
            result = verify(
                source=source_text,
                output=output_text,
                mode=args.mode,
                record_path=args.record,
                record=not args.no_record,
                **_policy_kwargs(args),
            )
        except (CapabilityError, RecordCorruptError, InputTooLargeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        if hasattr(args, "quiet") and args.quiet:
            return _exit_code(result, args)

        if hasattr(args, "json") and args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            _print_human(result)

        return _exit_code(result, args)

    elif args.command == "batch":
        return _run_batch(args)

    elif args.command == "chain":
        ok, msg = verify_chain(
            args.record, expected_head=args.expect_head, expected_count=args.expect_count,
        )
        if ok:
            print(f"✓ {msg}")
            head, count = chain_head(args.record)
            print(f"  head: {head}  entries: {count}")
            if not os.environ.get(_KEY_ENV):
                print(f"  note: unkeyed chain — detects edits, not a full rewrite. Set {_KEY_ENV} to key it.")
            if args.expect_head is None and args.expect_count is None:
                print("  note: store the head and count somewhere the writer cannot reach, then pass "
                      "--expect-head/--expect-count to detect truncation.")
            return 0
        else:
            print(f"✗ CHAIN BROKEN: {msg}")
            return 1

    parser.print_help()
    return 1


def _exit_code(result, args) -> int:
    """0 trustworthy, 2 not, 3 trustworthy-but-unsure when --fail-on-unsure."""
    if not result.is_trustworthy:
        return 2
    if getattr(args, "fail_on_unsure", False) and result.confidence == "unsure":
        return 3
    return 0


def _print_human(result) -> None:
    sym = {"faithful": "✓", "adapted": "~", "divergent": "!", "unreliable": "✗", "unverified": "?"}.get(result.trust_level, "?")
    print(f"\n  {sym} {result.trust_level.upper()} — {result.trust_score:.0%} trust  [mode: {result.mode}]\n")
    print(f"  {result.summary}\n")
    if result.groundedness is not None:
        print(f"  Groundedness (weakest sentence's source support): {result.groundedness:.2f}")
        if result.confidence:
            note = {
                "confident": "both grounding models agree, clearly",
                "probable": "both grounding models agree, at least one narrowly",
                "unsure": f"the grounding models disagree on {result.unsure_count} sentence(s) — check by hand",
            }[result.confidence]
            print(f"  Confidence: {result.confidence} ({note})")
        print()
    caps = result.capabilities
    if caps and not caps.get("model_checked"):
        print(f"  ⚠ The grounding model did not run ({caps.get('scorer_error') or 'unavailable'}).")
        print("    Contradictions and fabricated claims were NOT checked. pip install criterion-ai[nli]\n")

    if result.total_divergences > 0:
        print(f"  Divergences ({len(result.divergences)} atom-level, {len(result.broken_bonds)} broken bonds):")
        for d in result.divergences:
            if d.severity.value in ("critical", "major"):
                sev = d.severity.value.upper()[:4]
                atype = f"({d.source_type})" if d.source_type != "n/a" else f"(+{d.output_type})"
                print(f"    [{sev}] {d.dtype.value} {atype}: {d.explanation}")
                print(f"           fix: {d.suggested_correction[:100]}")
        for bd in result.broken_bonds:
            print(f"    [BOND] {bd.explanation}")
        print()

    if result.corrections:
        print(f"  Suggested corrections ({len(result.corrections)}):")
        for i, c in enumerate(result.corrections[:5], 1):
            print(f"    {i}. {c[:120]}")
        if len(result.corrections) > 5:
            print(f"    ... and {len(result.corrections) - 5} more")
        print()

    if result.record_id:
        print(f"  Recorded: {result.record_id[:12]}...")


def _run_batch(args) -> int:
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        return 1

    results = []
    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    results.append({"error": "line is not a JSON object", "line": line_num})
                    continue
                source = obj.get("source", "")
                output = obj.get("output", "")
                # Reject empty/missing/non-string source or output
                if not isinstance(source, str) or not isinstance(output, str) or not source or not output:
                    results.append({"error": "missing or non-string source or output", "line": line_num})
                    continue
                result = verify(
                    source=source,
                    output=output,
                    mode=args.mode,
                    record_path=args.record,
                    record=not args.no_record,
                    **_policy_kwargs(args),
                )
                results.append(result.to_dict())
            except (CapabilityError, RecordCorruptError) as e:
                # Not a per-pair problem: the gate itself cannot run. Stop.
                print(f"Error: {e}", file=sys.stderr)
                return 1
            except Exception as e:
                results.append({"error": str(e), "line": line_num})

    # Reject empty batches
    if not results:
        print("Error: batch file is empty or contains no valid entries", file=sys.stderr)
        return 1

    output_path = args.output or str(input_path.with_suffix(".results.jsonl"))
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    trustworthy = sum(1 for r in results if r.get("is_trustworthy"))
    untrustworthy = total - errors - trustworthy
    unsure = sum(1 for r in results if r.get("is_trustworthy") and r.get("confidence") == "unsure")
    print(f"\n  Batch: {total} pairs → {trustworthy} trustworthy, {untrustworthy} flagged, {errors} errors")
    if unsure:
        print(f"  Of the trustworthy, {unsure} are unsure: the grounding models disagree on some sentence.")
    print(f"  Results: {output_path}")
    if errors:
        return 1  # Processing errors
    if untrustworthy > 0:
        return 2  # Untrustworthy content found
    if unsure and getattr(args, "fail_on_unsure", False):
        return 3  # Trustworthy on average, but the models disagree
    return 0


if __name__ == "__main__":
    sys.exit(main())
