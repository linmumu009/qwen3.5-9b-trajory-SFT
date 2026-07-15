"""Reuse exact Qwen3.5 token counts and estimate only changed candidates."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from analyze_trajectory_tokens import (
    length_bucket,
    quality_tier,
    ratio_bucket,
    resolve_task_id,
    source_family,
)
from catalog_trajectory_candidates import canonical_tools, source_mapping
from prepare_trajectory_sft import (
    convert_messages,
    detect_version,
    iter_jsonl,
    load_manifest_data,
    load_verdicts,
)


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("calibration values are empty")
    return ordered[round((len(ordered) - 1) * ratio)]


def calibration_ratios(rows: list[dict[str, Any]]) -> dict[str, float]:
    token_per_character = [
        float(row["input_tokens"]) / max(1, int(row["characters"])) for row in rows
    ]
    supervised_ratios = [float(row["supervised_ratio"]) for row in rows]
    return {
        "token_per_character_p05": percentile(token_per_character, 0.05),
        "token_per_character_p50": percentile(token_per_character, 0.50),
        "token_per_character_p95": percentile(token_per_character, 0.95),
        "supervised_ratio_p50": percentile(supervised_ratios, 0.50),
    }


def estimate_token_metadata(
    characters: int, calibration: dict[str, float]
) -> dict[str, Any]:
    input_tokens = max(1, round(characters * calibration["token_per_character_p50"]))
    supervised_ratio = calibration["supervised_ratio_p50"]
    supervised_tokens = max(1, round(input_tokens * supervised_ratio))
    return {
        "input_tokens": input_tokens,
        "supervised_tokens": supervised_tokens,
        "supervised_ratio": round(supervised_tokens / input_tokens, 6),
        "input_tokens_estimate_low": max(
            1, round(characters * calibration["token_per_character_p05"])
        ),
        "input_tokens_estimate_high": max(
            1, round(characters * calibration["token_per_character_p95"])
        ),
        "token_method": "estimated_from_qwen35_calibration",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--verdicts-dir", type=Path, required=True)
    parser.add_argument(
        "--verdict-layout", choices=("fixed", "upstream_openai"), default="fixed"
    )
    parser.add_argument("--calibration-catalog", type=Path, required=True)
    parser.add_argument("--catalog-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_rows = [row for _, row in iter_jsonl(args.calibration_catalog)]
    calibration_by_location = {
        (row["source_file"], int(row["source_line"])): row for row in calibration_rows
    }
    calibration_locations = set(calibration_by_location)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        grouped[(str(row["source_family"]), str(row["type"]))].append(row)
        grouped_source[str(row["source_family"])].append(row)
    global_calibration = calibration_ratios(calibration_rows)
    group_calibrations = {key: calibration_ratios(rows) for key, rows in grouped.items()}
    source_calibrations = {
        key: calibration_ratios(rows) for key, rows in grouped_source.items()
    }

    args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    catalog_tmp = args.catalog_output.with_suffix(args.catalog_output.suffix + ".tmp")
    summary_tmp = args.summary_output.with_suffix(args.summary_output.suffix + ".tmp")
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    current_locations: set[tuple[str, int]] = set()

    manifest_cache: dict[tuple[Path, str], Any] = {}
    verdict_cache: dict[tuple[Path, str], Any] = {}
    with catalog_tmp.open("w", encoding="utf-8") as output:
        for data_file in sorted(args.data_dir.glob("*.jsonl")):
            version = detect_version(data_file)
            manifest_file, verdict_file = source_mapping(
                data_file.name,
                args.manifests_dir,
                args.verdicts_dir,
                args.verdict_layout,
            )
            manifest_key = (manifest_file, version)
            verdict_key = (verdict_file, version)
            if manifest_key not in manifest_cache:
                manifest_cache[manifest_key] = load_manifest_data(manifest_file, version)
            if verdict_key not in verdict_cache:
                verdict_cache[verdict_key] = load_verdicts(verdict_file, version)
            manifest_lookup, manifest_order, manifest_source_lookup = manifest_cache[manifest_key]
            verdict_rows, conflicts = verdict_cache[verdict_key]
            if conflicts:
                raise RuntimeError(f"{data_file.name}: conflicting verdicts")

            for line_number, record in iter_jsonl(data_file):
                stats["source_records"] += 1
                task_id, alignment = resolve_task_id(
                    record,
                    line_number,
                    manifest_lookup,
                    manifest_order,
                    manifest_source_lookup,
                )
                stats[f"alignment_{alignment}"] += 1
                if task_id is None:
                    continue
                verdict_row = verdict_rows.get(task_id)
                if not verdict_row or verdict_row.get("verdict") != "correct":
                    continue
                stats["correct"] += 1
                try:
                    converted, conversion = convert_messages(record.get("messages") or [])
                except Exception:
                    stats["excluded_conversion_error"] += 1
                    continue
                if converted[-1]["role"] != "assistant":
                    stats["excluded_no_final_assistant"] += 1
                    continue
                if not conversion.get("tool_calls"):
                    stats["excluded_no_tool_roundtrip"] += 1
                    continue
                tools_ok, _ = canonical_tools(converted)
                if not tools_ok:
                    stats["excluded_noncanonical_tools"] += 1
                    continue

                family = source_family(data_file.name)
                task_type = str(verdict_row.get("type"))
                location = (data_file.name, line_number)
                previous = calibration_by_location.get(location)
                if previous is not None:
                    token_metadata = {
                        "input_tokens": int(previous["input_tokens"]),
                        "supervised_tokens": int(previous["supervised_tokens"]),
                        "supervised_ratio": float(previous["supervised_ratio"]),
                        "input_tokens_estimate_low": int(previous["input_tokens"]),
                        "input_tokens_estimate_high": int(previous["input_tokens"]),
                        "token_method": "exact_reused_qwen35",
                    }
                    stats["exact_reused_qwen35"] += 1
                else:
                    calibration = group_calibrations.get(
                        (family, task_type), source_calibrations.get(family, global_calibration)
                    )
                    token_metadata = estimate_token_metadata(
                        int(conversion["characters"]), calibration
                    )
                    stats["estimated_from_qwen35_calibration"] += 1

                metadata = {
                    "source_file": data_file.name,
                    "source_line": line_number,
                    "source_family": family,
                    "version": version,
                    "task_id": task_id,
                    "type": verdict_row.get("type"),
                    "quality_tier": quality_tier(verdict_row),
                    **token_metadata,
                    "length_bucket": length_bucket(token_metadata["input_tokens"]),
                    "supervised_ratio_bucket": ratio_bucket(
                        token_metadata["supervised_ratio"]
                    ),
                    "characters": conversion["characters"],
                    "messages": conversion["output_messages"],
                    "tool_calls": conversion["tool_calls"],
                    "assistant_messages": conversion.get("assistant_content_messages", 0),
                }
                output.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows.append(metadata)
                current_locations.add(location)
                stats["candidates"] += 1

    if not rows:
        raise RuntimeError("no candidates produced")
    os.replace(catalog_tmp, args.catalog_output)
    input_tokens = [int(row["input_tokens"]) for row in rows]
    boundary_sensitive = sum(
        any(
            int(row["input_tokens_estimate_low"]) <= boundary
            < int(row["input_tokens_estimate_high"])
            for boundary in (8192, 16384, 32768)
        )
        for row in rows
        if row["token_method"] != "exact_reused_qwen35"
    )
    token_per_character = [
        float(row["input_tokens"]) / max(1, int(row["characters"]))
        for row in calibration_rows
    ]
    summary = {
        "records": len(rows),
        "stats": dict(sorted(stats.items())),
        "exact_reused_records": sum(
            row["token_method"] == "exact_reused_qwen35" for row in rows
        ),
        "estimated_records": sum(
            row["token_method"] != "exact_reused_qwen35" for row in rows
        ),
        "estimated_boundary_sensitive_records": boundary_sensitive,
        "calibration_only_locations": len(calibration_locations - current_locations),
        "current_only_locations": len(current_locations - calibration_locations),
        "input_tokens_total": sum(input_tokens),
        "input_tokens_p50": int(percentile([float(v) for v in input_tokens], 0.50)),
        "input_tokens_p95": int(percentile([float(v) for v in input_tokens], 0.95)),
        "input_tokens_max": max(input_tokens),
        "qwen35_calibration": {
            "records": len(calibration_rows),
            "token_per_character_p05": percentile(token_per_character, 0.05),
            "token_per_character_p50": percentile(token_per_character, 0.50),
            "token_per_character_p95": percentile(token_per_character, 0.95),
            "aggregate_token_per_character": sum(
                int(row["input_tokens"]) for row in calibration_rows
            )
            / sum(int(row["characters"]) for row in calibration_rows),
        },
        "by_quality_tier": dict(sorted(Counter(row["quality_tier"] for row in rows).items())),
        "by_length_bucket": dict(sorted(Counter(row["length_bucket"] for row in rows).items())),
    }
    with summary_tmp.open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(summary_tmp, args.summary_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
