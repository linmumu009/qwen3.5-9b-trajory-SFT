"""Join verdict=correct indexes to raw trajectories and write one OpenAI JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from analyze_trajectory_tokens import resolve_task_id
from catalog_trajectory_candidates import source_mapping
from prepare_trajectory_sft import detect_version, iter_jsonl, load_manifest_data, load_verdicts


def verdict_filename(source_name: str, version: str) -> str:
    if source_name.startswith("qwen3.6-27B_"):
        prefix = "qwen3.6-27B"
    elif source_name.startswith("deepseek_"):
        prefix = "aliyun-deepseek-v4-pro"
    elif source_name.startswith("glm52_"):
        prefix = "aliyun-glm-5.2"
    elif source_name.startswith("qwen37max_"):
        prefix = "aliyun-qwen3.7-max"
    else:
        raise ValueError(f"no boss verdict mapping for source: {source_name}")
    return f"{prefix}_{version}_openai.jsonl"


def resolve_verdict_file(directory: Path, source_name: str, version: str) -> Path:
    current = directory / verdict_filename(source_name, version)
    if current.is_file():
        return current
    suffix = version.rsplit("_v", 1)[-1]
    legacy = directory / verdict_filename(source_name, version).replace(
        f"_{version}_", f"_v{suffix}_"
    )
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(current)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_correct(
    data_dir: Path,
    manifests_dir: Path,
    verdicts_dir: Path,
    output: Path,
) -> dict[str, Any]:
    source_files = sorted(data_dir.glob("*.jsonl"))
    if not source_files:
        raise FileNotFoundError(f"no raw JSONL files found: {data_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    total_expected = 0
    total_written = 0
    total_verdict_rows = 0
    total_raw_rows = 0
    missing_verdict_rows = 0
    by_source: dict[str, dict[str, int]] = {}

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for data_file in source_files:
                version = detect_version(data_file)
                manifest_file, _ = source_mapping(data_file.name, manifests_dir, verdicts_dir)
                verdict_file = resolve_verdict_file(verdicts_dir, data_file.name, version)
                manifest_lookup, manifest_order, manifest_source_lookup = load_manifest_data(
                    manifest_file, version
                )
                verdict_rows, conflicts = load_verdicts(verdict_file, version)
                if conflicts:
                    raise RuntimeError(f"conflicting verdicts in {verdict_file.name}")
                expected_ids = {
                    task_id
                    for task_id, row in verdict_rows.items()
                    if row.get("verdict") == "correct"
                }
                total_expected += len(expected_ids)
                total_verdict_rows += len(verdict_rows)
                matched_ids: set[str] = set()
                raw_rows = 0
                selected = 0
                for line_number, record in iter_jsonl(data_file):
                    raw_rows += 1
                    task_id, _ = resolve_task_id(
                        record,
                        line_number,
                        manifest_lookup,
                        manifest_order,
                        manifest_source_lookup,
                    )
                    if task_id is None:
                        continue
                    verdict = verdict_rows.get(task_id)
                    if verdict is None:
                        continue
                    if verdict.get("verdict") != "correct":
                        continue
                    if task_id in matched_ids:
                        raise RuntimeError(
                            f"duplicate raw task identity: {data_file.name}:{task_id}"
                        )
                    matched_ids.add(task_id)
                    handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    selected += 1
                    total_written += 1
                missing_correct = expected_ids - matched_ids
                if missing_correct:
                    raise RuntimeError(
                        f"correct verdicts missing raw trajectories in {data_file.name}: "
                        f"{sorted(missing_correct)[:10]}"
                    )
                total_raw_rows += raw_rows
                missing_verdict_rows += raw_rows - len(verdict_rows)
                by_source[data_file.name] = {
                    "raw_records": raw_rows,
                    "verdict_records": len(verdict_rows),
                    "correct_records": selected,
                }
        if total_written != total_expected:
            raise RuntimeError(
                f"selected row mismatch: expected={total_expected}, written={total_written}"
            )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "output": str(output),
        "format": "OpenAI trajectory JSONL",
        "source_files": len(source_files),
        "raw_records": total_raw_rows,
        "verdict_records": total_verdict_rows,
        "raw_records_without_verdict": missing_verdict_rows,
        "correct_records": total_written,
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "by_source": by_source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--verdicts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize_correct(
        args.data_dir,
        args.manifests_dir,
        args.verdicts_dir,
        args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
