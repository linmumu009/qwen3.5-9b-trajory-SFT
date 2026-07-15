"""Materialize a metadata dataset plan into isolated ms-swift JSONL sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_trajectory_sft import convert_messages, iter_jsonl, make_training_row


def output_group(row: dict[str, Any]) -> str:
    phase = row["phase"]
    split = row["split"]
    if phase == "long_32k_review":
        return "long_32k_review"
    if phase != "core_8k" and not phase.startswith("extension_"):
        raise ValueError(f"unexpected plan phase: {phase}")
    if split != "train":
        return "heldout"
    if row["quality_tier"] == "sql_result_verified":
        return "train_strong_verified"
    return "train_review"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    selected: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for _, row in iter_jsonl(path):
        source_file = row["source_file"]
        source_line = int(row["source_line"])
        if source_line in selected[source_file]:
            raise RuntimeError(f"duplicate plan location: {source_file}:{source_line}")
        selected[source_file][source_line] = row
    return selected


def materialize(
    plan: Path, data_dir: Path, output_dir: Path, training_label: str = "16k"
) -> dict[str, Any]:
    selected = load_plan(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {
        "train_strong_verified": f"train_strong_verified_{training_label}.jsonl",
        "train_review": f"train_review_{training_label}.jsonl",
        "heldout": f"heldout_{training_label}.jsonl",
        "long_32k_review": "long_32k_review.jsonl",
    }
    final_paths = {key: output_dir / value for key, value in output_names.items()}
    temporary_paths = {
        key: path.with_suffix(path.suffix + ".tmp") for key, path in final_paths.items()
    }
    handles = {}
    counts: Counter[str] = Counter()
    expected: Counter[str] = Counter()
    for lines in selected.values():
        for row in lines.values():
            expected[output_group(row)] += 1

    try:
        handles = {
            key: path.open("w", encoding="utf-8", newline="\n")
            for key, path in temporary_paths.items()
        }
        for source_file, wanted in selected.items():
            source_path = data_dir / source_file
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            found: set[int] = set()
            for line_number, record in iter_jsonl(source_path):
                plan_row = wanted.get(line_number)
                if plan_row is None:
                    continue
                found.add(line_number)
                converted, _ = convert_messages(record.get("messages") or [])
                group = output_group(plan_row)
                handles[group].write(
                    json.dumps(
                        make_training_row(converted),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                counts[group] += 1
            missing = set(wanted) - found
            if missing:
                raise RuntimeError(
                    f"plan locations missing in {source_file}: {sorted(missing)[:10]}"
                )
        for handle in handles.values():
            handle.close()
        handles = {}
        if counts != expected:
            raise RuntimeError(f"output count mismatch: expected={expected}, actual={counts}")
        for key, temporary in temporary_paths.items():
            os.replace(temporary, final_paths[key])
    except BaseException:
        for handle in handles.values():
            handle.close()
        for path in temporary_paths.values():
            if path.exists():
                path.unlink()
        raise

    outputs = {}
    for key, path in final_paths.items():
        outputs[key] = {
            "path": str(path),
            "records": counts[key],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {"plan_records": sum(counts.values()), "outputs": outputs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--training-label", default="16k")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        args.plan, args.data_dir, args.output_dir, training_label=args.training_label
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary_output.with_suffix(args.summary_output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, args.summary_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
