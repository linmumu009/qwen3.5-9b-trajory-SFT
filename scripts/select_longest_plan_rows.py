#!/usr/bin/env python3
"""Select the longest eligible train rows from a trajectory dataset plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def token_upper_bound(row: dict[str, Any]) -> int:
    return int(row.get("input_tokens_estimate_high", row["input_tokens"]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=16384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    rows = [
        row
        for _, row in iter_jsonl(args.plan)
        if row.get("split") == "train"
        and row.get("phase") != "long_32k_review"
        and token_upper_bound(row) <= args.max_tokens
    ]
    rows.sort(
        key=lambda row: (
            token_upper_bound(row),
            int(row.get("input_tokens", 0)),
            row["source_file"],
            int(row["source_line"]),
        ),
        reverse=True,
    )
    selected = rows[: args.limit]
    if len(selected) != args.limit:
        raise RuntimeError(
            f"requested {args.limit} rows but only {len(selected)} are eligible"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, args.output)

    summary = {
        "source_plan": str(args.plan),
        "output": str(args.output),
        "records": len(selected),
        "max_tokens_limit": args.max_tokens,
        "selected_token_min": min(token_upper_bound(row) for row in selected),
        "selected_token_max": max(token_upper_bound(row) for row in selected),
        "output_sha256": sha256_file(args.output),
        "locations": [
            {
                "source_file": row["source_file"],
                "source_line": int(row["source_line"]),
                "tokens": token_upper_bound(row),
            }
            for row in selected
        ],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = args.summary_output.with_suffix(
        args.summary_output.suffix + ".tmp"
    )
    with temporary_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_summary, args.summary_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
