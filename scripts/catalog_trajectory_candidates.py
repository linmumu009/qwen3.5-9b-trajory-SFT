"""Catalog high-quality trajectory SFT candidates across all source models."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

from prepare_trajectory_sft import (
    conversion_error_category,
    convert_messages,
    detect_version,
    first_user_content,
    iter_jsonl,
    load_manifest_data,
    load_verdicts,
    normalize_text,
)


CANONICAL_TOOL_ARGUMENTS = {
    "bash": ({"command", "timeout"}, {"command"}),
    "read": ({"path", "offset", "limit"}, {"path"}),
    "write": ({"path", "content"}, {"path", "content"}),
    "edit": ({"path", "edits", "oldText", "newText"}, {"path"}),
}


def source_mapping(name: str, manifests: Path, verdicts: Path) -> tuple[Path, Path]:
    if name.startswith("qwen3.6-27B_"):
        return manifests / "manifest.jsonl", verdicts / "qwen3.6-27B_sft_all_fixed.jsonl"
    if name.startswith("deepseek_"):
        return (
            manifests / "trajectories_deepseek_v15_manifest.jsonl",
            verdicts / "aliyun-deepseek-v4-pro_v15_fixed.jsonl",
        )
    if name.startswith("glm52_"):
        return (
            manifests / "trajectories_glm52_v15_manifest.jsonl",
            verdicts / "aliyun-glm-5.2_v15_fixed.jsonl",
        )
    if name.startswith("qwen37max_"):
        version = detect_version(Path(name)).rsplit("_v", 1)[-1]
        manifest_name = (
            "trajectories_qwen37max_manifest.jsonl"
            if version == "15"
            else f"trajectories_qwen37max_v{version}_manifest.jsonl"
        )
        return (
            manifests / manifest_name,
            verdicts / f"aliyun-qwen3.7-max_v{version}_fixed.jsonl",
        )
    raise ValueError(f"No source mapping for {name}")


def canonical_tools(messages: list[dict]) -> tuple[bool, str | None]:
    for message in messages:
        if message["role"] != "tool_call":
            continue
        tool_call = json.loads(message["content"])
        name = tool_call.get("name")
        spec = CANONICAL_TOOL_ARGUMENTS.get(name)
        if spec is None:
            return False, "noncanonical_tool_name"
        arguments = tool_call.get("arguments")
        if not isinstance(arguments, dict):
            return False, "non_object_arguments"
        allowed, required = spec
        keys = set(arguments)
        if not required.issubset(keys):
            return False, "missing_required_argument"
        if not keys.issubset(allowed):
            return False, "noncanonical_argument_key"
    return True, None


def percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--verdicts-dir", type=Path, required=True)
    args = parser.parse_args()

    all_stats: Counter[str] = Counter()
    file_reports: dict[str, dict] = {}
    candidate_characters: list[int] = []
    candidate_messages: list[int] = []
    candidate_tool_calls: list[int] = []
    version_task_counts: Counter[tuple[str, str]] = Counter()
    task_counts: Counter[str] = Counter()

    for data_file in sorted(args.data_dir.glob("*.jsonl")):
        version = detect_version(data_file)
        manifest_file, verdict_file = source_mapping(data_file.name, args.manifests_dir, args.verdicts_dir)
        manifest_lookup, manifest_order, manifest_source_lookup = load_manifest_data(manifest_file, version)
        verdict_rows, conflicts = load_verdicts(verdict_file, version)
        if conflicts:
            raise RuntimeError(f"{data_file.name}: conflicting verdicts")

        stats: Counter[str] = Counter()
        error_categories: Counter[str] = Counter()
        for line_number, record in iter_jsonl(data_file):
            stats["records"] += 1
            normalized_user = normalize_text(first_user_content(record))
            task_id = None
            source = record.get("_source")
            source_candidates = manifest_source_lookup.get(source, set()) if isinstance(source, str) else set()
            if len(source_candidates) == 1:
                task_id = next(iter(source_candidates))
                stats["source_verified"] += 1
            elif line_number <= len(manifest_order) and manifest_order[line_number - 1][0] == normalized_user:
                task_id = manifest_order[line_number - 1][1]
                stats["position_verified"] += 1
            else:
                candidates = manifest_lookup.get(normalized_user, set())
                if len(candidates) == 1:
                    task_id = next(iter(candidates))
                    stats["unique_text_aligned"] += 1
            if task_id is None:
                stats["identity_unresolved"] += 1
                continue
            verdict = verdict_rows.get(task_id, {}).get("verdict")
            stats[f"verdict_{verdict}"] += 1
            if verdict != "correct":
                continue
            try:
                converted, conversion = convert_messages(record["messages"])
            except Exception as exc:
                stats["correct_conversion_error"] += 1
                error_categories[conversion_error_category(exc)] += 1
                continue
            stats["correct_structurally_valid"] += 1
            if converted[-1]["role"] != "assistant":
                stats["correct_without_final_assistant"] += 1
                continue
            if not conversion.get("tool_calls"):
                stats["correct_without_tool_roundtrip"] += 1
                continue
            tools_ok, tool_error = canonical_tools(converted)
            if not tools_ok:
                stats["correct_noncanonical_tools"] += 1
                error_categories[tool_error or "unknown_tool_error"] += 1
                continue

            stats["trajectory_sft_candidate"] += 1
            candidate_characters.append(conversion["characters"])
            candidate_messages.append(conversion["output_messages"])
            candidate_tool_calls.append(conversion["tool_calls"])
            version_task_counts[(version, task_id)] += 1
            task_counts[task_id] += 1

        file_reports[data_file.name] = {
            "stats": dict(sorted(stats.items())),
            "error_categories": dict(sorted(error_categories.items())),
        }
        all_stats.update(stats)

    duplicate_version_tasks = sum(1 for count in version_task_counts.values() if count > 1)
    duplicate_tasks = sum(1 for count in task_counts.values() if count > 1)
    report = {
        "files": file_reports,
        "total": dict(sorted(all_stats.items())),
        "candidate_unique_version_task": len(version_task_counts),
        "candidate_duplicate_version_task_groups": duplicate_version_tasks,
        "candidate_unique_task_id": len(task_counts),
        "candidate_duplicate_task_id_groups": duplicate_tasks,
        "candidate_character_distribution": {
            "min": min(candidate_characters, default=0),
            "p50": int(median(candidate_characters)) if candidate_characters else 0,
            "p90": percentile(candidate_characters, 0.90),
            "p95": percentile(candidate_characters, 0.95),
            "max": max(candidate_characters, default=0),
        },
        "candidate_message_distribution": {
            "min": min(candidate_messages, default=0),
            "p50": int(median(candidate_messages)) if candidate_messages else 0,
            "p95": percentile(candidate_messages, 0.95),
            "max": max(candidate_messages, default=0),
        },
        "candidate_tool_call_distribution": {
            "min": min(candidate_tool_calls, default=0),
            "p50": int(median(candidate_tool_calls)) if candidate_tool_calls else 0,
            "p95": percentile(candidate_tool_calls, 0.95),
            "max": max(candidate_tool_calls, default=0),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
