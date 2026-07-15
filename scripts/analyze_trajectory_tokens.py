"""Profile Qwen3.5 trajectory candidates without emitting trajectory text.

The script reuses the strict candidate gate from ``catalog_trajectory_candidates``
and applies the real ms-swift Qwen3.5 training template.  It writes a metadata-only
catalog, an aggregate summary, and (optionally) one server-local benchmark sample
near each requested token target.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from catalog_trajectory_candidates import canonical_tools, source_mapping
from prepare_trajectory_sft import (
    convert_messages,
    detect_version,
    first_user_content,
    iter_jsonl,
    load_manifest_data,
    load_verdicts,
    make_training_row,
    normalize_text,
)


LENGTH_LIMITS = (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
RATIO_LIMITS = (0.05, 0.10, 0.20, 0.40)


def as_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def percentile(values: Iterable[int | float], ratio: float) -> int | float:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = round((len(ordered) - 1) * ratio)
    return ordered[index]


def length_bucket(tokens: int) -> str:
    lower = 0
    for limit in LENGTH_LIMITS:
        if tokens <= limit:
            return f"{lower + 1}-{limit}"
        lower = limit
    return f">{LENGTH_LIMITS[-1]}"


def ratio_bucket(ratio: float) -> str:
    lower = 0.0
    for limit in RATIO_LIMITS:
        if ratio < limit:
            return f"{lower:.0%}-<{limit:.0%}"
        lower = limit
    return f">={RATIO_LIMITS[-1]:.0%}"


def source_family(filename: str) -> str:
    if filename.startswith("qwen3.6-27B_"):
        return "qwen3.6-27b"
    if filename.startswith("deepseek_"):
        return "deepseek-v4-pro"
    if filename.startswith("glm52_"):
        return "glm-5.2"
    if filename.startswith("qwen37max_"):
        return "qwen3.7-max"
    return "unknown"


def quality_tier(verdict_row: dict[str, Any]) -> str:
    evidence = verdict_row.get("evidence") or {}
    if evidence.get("agent_sql_ok") is True:
        return "sql_result_verified"
    if evidence.get("is_report") is True:
        return "report_rule_only"
    if verdict_row.get("type") == "kb":
        return "kb_rule_only"
    return "judge_rule_only"


def resolve_task_id(
    record: dict[str, Any],
    line_number: int,
    manifest_lookup: dict[str, set[str]],
    manifest_order: list[tuple[str, str]],
    manifest_source_lookup: dict[str, set[str]],
) -> tuple[str | None, str]:
    normalized_user = normalize_text(first_user_content(record))
    source = record.get("_source")
    source_candidates = (
        manifest_source_lookup.get(source, set()) if isinstance(source, str) else set()
    )
    if len(source_candidates) == 1:
        return next(iter(source_candidates)), "source"
    if line_number <= len(manifest_order):
        positional_instruction, positional_task_id = manifest_order[line_number - 1]
        if positional_instruction == normalized_user:
            return positional_task_id, "position"
    candidates = manifest_lookup.get(normalized_user, set())
    if len(candidates) == 1:
        return next(iter(candidates)), "unique_text"
    return None, "unresolved"


def group_summary(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field))].append(row)
    result: dict[str, dict[str, Any]] = {}
    for name, values in sorted(groups.items()):
        input_tokens = [row["input_tokens"] for row in values]
        supervised_tokens = [row["supervised_tokens"] for row in values]
        total_input = sum(input_tokens)
        total_supervised = sum(supervised_tokens)
        result[name] = {
            "records": len(values),
            "input_tokens_total": total_input,
            "supervised_tokens_total": total_supervised,
            "weighted_supervised_ratio": round(total_supervised / total_input, 6),
            "input_tokens_p50": percentile(input_tokens, 0.50),
            "input_tokens_p95": percentile(input_tokens, 0.95),
            "input_tokens_max": max(input_tokens),
        }
    return result


def benchmark_rank(row: dict[str, Any], target: int) -> tuple[int, int, float, int]:
    tier_rank = {
        "sql_result_verified": 0,
        "report_rule_only": 1,
        "kb_rule_only": 2,
        "judge_rule_only": 3,
    }.get(row["quality_tier"], 4)
    ratio_penalty = 0 if row["supervised_ratio"] >= 0.10 else 1
    return (
        abs(row["input_tokens"] - target),
        ratio_penalty,
        tier_rank,
        -row["supervised_tokens"],
    )


def parse_targets(value: str) -> list[int]:
    targets = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if any(target <= 0 for target in targets):
        raise ValueError("benchmark targets must be positive")
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--verdicts-dir", type=Path, required=True)
    parser.add_argument("--model", default="/models/Qwen3.5-9B")
    parser.add_argument("--catalog-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--benchmark-output-dir", type=Path)
    parser.add_argument("--benchmark-targets", default="4096,8192,16384,32768")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from swift import get_processor, get_template

    processor = get_processor(args.model)
    template = get_template(processor, loss_scale="default+ignore_empty_think")
    template.set_mode("train")

    args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    catalog_tmp = args.catalog_output.with_suffix(args.catalog_output.suffix + ".tmp")
    summary_tmp = args.summary_output.with_suffix(args.summary_output.suffix + ".tmp")
    targets = parse_targets(args.benchmark_targets)
    benchmark_choices: dict[int, tuple[tuple[int, int, float, int], dict[str, Any], dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    version_task_counts: Counter[tuple[str, str]] = Counter()
    started = time.time()

    manifest_cache: dict[tuple[Path, str], Any] = {}
    verdict_cache: dict[tuple[Path, str], Any] = {}
    with catalog_tmp.open("w", encoding="utf-8") as catalog_handle:
        for data_file in sorted(args.data_dir.glob("*.jsonl")):
            version = detect_version(data_file)
            manifest_file, verdict_file = source_mapping(
                data_file.name, args.manifests_dir, args.verdicts_dir
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

                training_row = make_training_row(converted)
                encoded = template.encode(training_row)
                input_ids = as_list(encoded["input_ids"])
                labels = as_list(encoded["labels"])
                input_tokens = len(input_ids)
                supervised_tokens = sum(label != -100 for label in labels)
                if supervised_tokens == 0:
                    stats["excluded_no_supervised_tokens"] += 1
                    continue
                supervised_ratio = supervised_tokens / input_tokens
                evidence = verdict_row.get("evidence") or {}
                metadata = {
                    "source_file": data_file.name,
                    "source_line": line_number,
                    "source_family": source_family(data_file.name),
                    "version": version,
                    "task_id": task_id,
                    "type": verdict_row.get("type"),
                    "quality_tier": quality_tier(verdict_row),
                    "agent_sql_ok": evidence.get("agent_sql_ok") is True,
                    "answer_type": evidence.get("answer_type"),
                    "input_tokens": input_tokens,
                    "supervised_tokens": supervised_tokens,
                    "supervised_ratio": round(supervised_ratio, 6),
                    "length_bucket": length_bucket(input_tokens),
                    "supervised_ratio_bucket": ratio_bucket(supervised_ratio),
                    "characters": conversion["characters"],
                    "messages": conversion["output_messages"],
                    "tool_calls": conversion["tool_calls"],
                    "assistant_messages": conversion.get("assistant_content_messages", 0),
                }
                catalog_handle.write(
                    json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                rows.append(metadata)
                task_counts[task_id] += 1
                version_task_counts[(version, task_id)] += 1
                stats["candidates_encoded"] += 1

                for target in targets:
                    rank = benchmark_rank(metadata, target)
                    current = benchmark_choices.get(target)
                    if current is None or rank < current[0]:
                        benchmark_choices[target] = (rank, training_row, metadata)

                if args.progress_every and stats["candidates_encoded"] % args.progress_every == 0:
                    elapsed = time.time() - started
                    print(
                        json.dumps(
                            {
                                "encoded": stats["candidates_encoded"],
                                "source_records": stats["source_records"],
                                "elapsed_seconds": round(elapsed, 1),
                                "records_per_second": round(stats["candidates_encoded"] / elapsed, 2),
                            },
                            separators=(",", ":"),
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

    if not rows:
        raise SystemExit("No candidates were encoded")
    os.replace(catalog_tmp, args.catalog_output)

    input_tokens = [row["input_tokens"] for row in rows]
    supervised_tokens = [row["supervised_tokens"] for row in rows]
    ratios = [row["supervised_ratio"] for row in rows]
    total_input = sum(input_tokens)
    total_supervised = sum(supervised_tokens)
    length_counts = Counter(row["length_bucket"] for row in rows)
    length_input_tokens = Counter()
    ratio_counts = Counter(row["supervised_ratio_bucket"] for row in rows)
    for row in rows:
        length_input_tokens[row["length_bucket"]] += row["input_tokens"]

    duplicate_version_task_groups = [
        count for count in version_task_counts.values() if count > 1
    ]
    duplicate_task_groups = [count for count in task_counts.values() if count > 1]
    summary = {
        "generated_at_epoch": int(time.time()),
        "model": args.model,
        "loss_scale": "default+ignore_empty_think",
        "source": {
            "data_dir": str(args.data_dir),
            "manifests_dir": str(args.manifests_dir),
            "verdicts_dir": str(args.verdicts_dir),
        },
        "stats": dict(sorted(stats.items())),
        "candidates": len(rows),
        "unique_task_ids": len(task_counts),
        "unique_version_task_ids": len(version_task_counts),
        "duplicate_task_id_groups": len(duplicate_task_groups),
        "duplicate_task_id_records": sum(duplicate_task_groups),
        "duplicate_version_task_groups": len(duplicate_version_task_groups),
        "duplicate_version_task_records": sum(duplicate_version_task_groups),
        "input_tokens": {
            "total": total_input,
            "min": min(input_tokens),
            "p50": percentile(input_tokens, 0.50),
            "p90": percentile(input_tokens, 0.90),
            "p95": percentile(input_tokens, 0.95),
            "p99": percentile(input_tokens, 0.99),
            "max": max(input_tokens),
        },
        "supervised_tokens": {
            "total": total_supervised,
            "min": min(supervised_tokens),
            "p50": percentile(supervised_tokens, 0.50),
            "p90": percentile(supervised_tokens, 0.90),
            "p95": percentile(supervised_tokens, 0.95),
            "p99": percentile(supervised_tokens, 0.99),
            "max": max(supervised_tokens),
        },
        "supervised_ratio": {
            "weighted": round(total_supervised / total_input, 6),
            "p01": percentile(ratios, 0.01),
            "p05": percentile(ratios, 0.05),
            "p50": percentile(ratios, 0.50),
            "p95": percentile(ratios, 0.95),
            "max": max(ratios),
        },
        "length_buckets": {
            key: {
                "records": length_counts[key],
                "record_share": round(length_counts[key] / len(rows), 6),
                "input_tokens": length_input_tokens[key],
                "input_token_share": round(length_input_tokens[key] / total_input, 6),
            }
            for key in [length_bucket(limit) for limit in LENGTH_LIMITS]
            + [f">{LENGTH_LIMITS[-1]}"]
            if length_counts[key]
        },
        "supervised_ratio_buckets": dict(sorted(ratio_counts.items())),
        "by_source_family": group_summary(rows, "source_family"),
        "by_type": group_summary(rows, "type"),
        "by_quality_tier": group_summary(rows, "quality_tier"),
        "benchmark_choices": {
            str(target): choice[2] for target, choice in sorted(benchmark_choices.items())
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    with summary_tmp.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(summary_tmp, args.summary_output)

    if args.benchmark_output_dir:
        args.benchmark_output_dir.mkdir(parents=True, exist_ok=True)
        for target, (_, training_row, metadata) in sorted(benchmark_choices.items()):
            stem = f"target_{target}_actual_{metadata['input_tokens']}"
            with (args.benchmark_output_dir / f"{stem}.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(training_row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            with (args.benchmark_output_dir / f"{stem}.metadata.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
