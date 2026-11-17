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

from criterion.api import verify, Mode
from criterion.record import verify_chain


def _read_file_or_text(path_or_text: str) -> str:
    """Read from a file path if it exists, otherwise treat as literal text."""
    p = Path(path_or_text)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return path_or_text


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

    # chain
    chain_p = sub.add_parser("chain", help="Verify the append-only record chain.")
    chain_p.add_argument("--record", default=None, help="Path to the record file.")

    # batch
    batch_p = sub.add_parser("batch", help="Verify multiple pairs from a JSONL file.")
    batch_p.add_argument("input_file", help="JSONL with {source, output} per line.")
    batch_p.add_argument("--mode", "-m", choices=["rag", "summary", "strict", "auto"], default="auto")
    batch_p.add_argument("--output", "-o", help="Output JSONL file.")
    batch_p.add_argument("--no-record", action="store_true")

    args = parser.parse_args(argv)

    if args.command is None or args.command == "verify":
        if not hasattr(args, "source"):
            parser.print_help()
            return 1

        source_text = _read_file_or_text(args.source)
        output_text = _read_file_or_text(args.output)
        result = verify(
            source=source_text,
            output=output_text,
            mode=args.mode,
            record_path=args.record,
            record=not args.no_record,
        )

        if hasattr(args, "quiet") and args.quiet:
            return 0 if result.is_trustworthy else 2

        if hasattr(args, "json") and args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            _print_human(result)

        return 0 if result.is_trustworthy else 2

    elif args.command == "batch":
        return _run_batch(args)

    elif args.command == "chain":
        ok, msg = verify_chain(args.record)
        if ok:
            print(f"✓ {msg}")
            return 0
        else:
            print(f"✗ CHAIN BROKEN: {msg}")
            return 1

    parser.print_help()
    return 1


def _print_human(result) -> None:
    sym = {"faithful": "✓", "adapted": "~", "divergent": "!", "unreliable": "✗"}.get(result.trust_level, "?")
    print(f"\n  {sym} {result.trust_level.upper()} — {result.trust_score:.0%} trust  [mode: {result.mode}]\n")
    print(f"  {result.summary}\n")

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
    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                result = verify(
                    source=obj.get("source", ""),
                    output=obj.get("output", ""),
                    mode=args.mode,
                    record=not args.no_record,
                )
                results.append(result.to_dict())
            except Exception as e:
                results.append({"error": str(e), "line": line_num})

    output_path = args.output or str(input_path.with_suffix(".results.jsonl"))
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    trustworthy = sum(1 for r in results if r.get("is_trustworthy"))
    print(f"\n  Batch: {total} pairs → {trustworthy} trustworthy, {total - errors - trustworthy} flagged, {errors} errors")
    print(f"  Results: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
