"""Select and materialize one ms-swift trajectory near each token target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_trajectory_tokens import benchmark_rank, parse_targets
from prepare_trajectory_sft import convert_messages, iter_jsonl, make_training_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="24576")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [row for _, row in iter_jsonl(args.catalog)]
    targets = parse_targets(args.targets)
    eligible = [row for row in rows if row.get("token_method") == "exact_reused_qwen35"]
    chosen = {
        target: min(eligible or rows, key=lambda row: benchmark_rank(row, target))
        for target in targets
    }

    by_source: dict[str, dict[int, tuple[int, dict[str, Any]]]] = {}
    for target, row in chosen.items():
        by_source.setdefault(row["source_file"], {})[int(row["source_line"])] = (target, row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for source_file, wanted in by_source.items():
        found = set()
        for line_number, record in iter_jsonl(args.data_dir / source_file):
            selected = wanted.get(line_number)
            if selected is None:
                continue
            found.add(line_number)
            target, metadata = selected
            converted, _ = convert_messages(record.get("messages") or [])
            actual = int(metadata["input_tokens"])
            stem = f"target_{target}_actual_{actual}"
            data_path = args.output_dir / f"{stem}.jsonl"
            metadata_path = args.output_dir / f"{stem}.metadata.json"
            with data_path.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        make_training_row(converted), ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                )
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            results[str(target)] = {
                "actual_tokens": actual,
                "data": str(data_path),
                "metadata": str(metadata_path),
            }
        missing = set(wanted) - found
        if missing:
            raise RuntimeError(f"benchmark locations missing in {source_file}: {sorted(missing)}")
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
