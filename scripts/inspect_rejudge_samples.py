"""Print deterministic, truncated samples for calibrating rejudge decisions.

This diagnostic writes nothing.  It is intentionally separate from the metadata-
only production output because its terminal output may contain dataset content.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from prepare_trajectory_sft import iter_jsonl
from rejudge_trajectory_candidates import (
    final_assistant_content,
    gold_text,
    manifest_fingerprint,
    recorded_tool_evidence,
)


def shorten(value: Any, limit: int) -> str:
    text = str(value or "").replace("\x00", "")
    return text if len(text) <= limit else text[:limit] + "…"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--per-reason", type=int, default=2)
    parser.add_argument("--text-limit", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decisions = [row for _, row in iter_jsonl(args.decisions)]
    chosen: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for row in decisions:
        reason = str(row["reason"])
        if counts[reason] >= args.per_reason:
            continue
        counts[reason] += 1
        chosen.append(row)

    by_source: dict[str, set[int]] = defaultdict(set)
    for row in chosen:
        by_source[row["source_file"]].add(int(row["source_line"]))
    raw: dict[tuple[str, int], dict[str, Any]] = {}
    for source_file, lines in by_source.items():
        for line, row in iter_jsonl(args.data_dir / source_file):
            if line in lines:
                raw[(source_file, line)] = row

    manifests: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(args.manifests_dir.glob("*.jsonl")):
        for _, row in iter_jsonl(path):
            key = (row.get("v"), row.get("task_id"))
            if all(isinstance(item, str) for item in key):
                manifests[key].append((path.name, row))

    output = []
    for decision in chosen:
        record = raw[(decision["source_file"], int(decision["source_line"]))]
        variants = manifests.get((decision["version"], decision["task_id"]), [])
        unique_variants: dict[str, tuple[str, dict[str, Any]]] = {}
        for path_name, manifest in variants:
            unique_variants.setdefault(manifest_fingerprint(manifest), (path_name, manifest))
        manifest = next(iter(unique_variants.values()))[1] if unique_variants else {}
        responses, calls = recorded_tool_evidence(record)
        output.append(
            {
                "decision": decision,
                "manifest_variant_count": len(unique_variants),
                "manifest_files": sorted({name for name, _ in variants}),
                "final_answer": shorten(final_assistant_content(record), args.text_limit),
                "gold_answer": shorten(gold_text(manifest), args.text_limit),
                "tool_responses": shorten(responses, args.text_limit),
                "tool_calls": shorten(calls, args.text_limit),
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
