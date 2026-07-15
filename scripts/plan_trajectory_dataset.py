"""Create a metadata-only curriculum plan from token and content-quality catalogs.

This script does not materialize training text.  It applies safety/efficiency gates,
deduplicates at ``(version, task_id)``, caps repetitions per task, and assigns a
stable task-level train/validation/test split to prevent leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_trajectory_sft import iter_jsonl


PHASES = (
    ("core_8k", 0, 8192, 0.10),
    ("extension_16k", 8192, 16384, 0.10),
    ("long_32k_review", 16384, 32768, 0.10),
)
QUALITY_RANK = {
    "sql_result_verified": 0,
    "judge_rule_only": 1,
    "report_rule_only": 2,
    "kb_rule_only": 3,
}


def stable_split(task_id: str, train_share: float, validation_share: float) -> str:
    value = int.from_bytes(hashlib.sha256(task_id.encode("utf-8")).digest()[:8], "big")
    ratio = value / 2**64
    if ratio < train_share:
        return "train"
    if ratio < train_share + validation_share:
        return "validation"
    return "test"


def phase_definitions(production_max_tokens: int) -> tuple[tuple[str, int, int, float], ...]:
    if not 8192 < production_max_tokens < 32768:
        raise ValueError("production max tokens must be between 8192 and 32768")
    label = f"{production_max_tokens // 1024}k"
    return (
        ("core_8k", 0, 8192, 0.10),
        (f"extension_{label}", 8192, production_max_tokens, 0.10),
        ("long_32k_review", production_max_tokens, 32768, 0.10),
    )


def phase_for(
    row: dict[str, Any], phases: tuple[tuple[str, int, int, float], ...] = PHASES
) -> tuple[str, float] | None:
    tokens = int(row["input_tokens"])
    for name, lower, upper, min_ratio in phases:
        if lower < tokens <= upper:
            return name, min_ratio
    return None


def selection_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        QUALITY_RANK.get(row["quality_tier"], 9),
        int(row["network_commands"] > 0),
        min(int(row["tool_error_messages"]), 20),
        min(int(row["duplicate_tool_calls"]), 20),
        int(row["nonportable_path_occurrences"] > 0),
        int(row["input_tokens"]),
        -float(row["supervised_ratio"]),
        -int(row["final_assistant_chars"]),
        row["source_file"],
        int(row["source_line"]),
    )


def choose_with_source_diversity(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=selection_rank)
    chosen: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for row in ordered:
        if row["source_family"] in seen_sources:
            continue
        chosen.append(row)
        seen_sources.add(row["source_family"])
        if len(chosen) == limit:
            return chosen
    for row in ordered:
        if row in chosen:
            continue
        chosen.append(row)
        if len(chosen) == limit:
            break
    return chosen


def load_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for _, row in iter_jsonl(path):
        rows[(row["source_file"], int(row["source_line"]))] = row
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-catalog", type=Path, required=True)
    parser.add_argument("--content-quality-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--max-per-task-per-phase", type=int, default=3)
    parser.add_argument("--train-share", type=float, default=0.85)
    parser.add_argument("--validation-share", type=float, default=0.075)
    parser.add_argument("--min-final-answer-chars", type=int, default=50)
    parser.add_argument("--production-max-tokens", type=int, default=16384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.train_share < 1:
        raise ValueError("train share must be in (0, 1)")
    if not 0 <= args.validation_share < 1 - args.train_share:
        raise ValueError("validation share leaves no test share")
    if args.max_per_task_per_phase < 1:
        raise ValueError("max-per-task-per-phase must be positive")
    phases = phase_definitions(args.production_max_tokens)

    tokens = load_rows(args.token_catalog)
    quality = load_rows(args.content_quality_catalog)
    if set(tokens) != set(quality):
        raise RuntimeError(
            f"catalog key mismatch: token={len(tokens)} content_quality={len(quality)}"
        )

    exclusion_reasons: Counter[str] = Counter()
    phase_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, token_row in tokens.items():
        row = dict(quality[key])
        row.update(token_row)
        phase = phase_for(row, phases)
        if phase is None:
            exclusion_reasons["over_32k"] += 1
            continue
        phase_name, min_ratio = phase
        reasons = []
        if float(row["supervised_ratio"]) < min_ratio:
            reasons.append("low_supervised_ratio")
        if int(row["final_assistant_chars"]) < args.min_final_answer_chars:
            reasons.append("short_final_answer")
        if int(row["secret_pattern_hits"]) > 0:
            reasons.append("secret_pattern")
        if int(row["destructive_commands"]) > 0:
            reasons.append("destructive_command")
        if int(row["network_commands"]) > 0:
            reasons.append("network_command")
        if reasons:
            for reason in set(reasons):
                exclusion_reasons[reason] += 1
            continue
        row["phase"] = phase_name
        row["split"] = stable_split(
            row["task_id"], args.train_share, args.validation_share
        )
        row["requires_rejudge"] = row["quality_tier"] != "sql_result_verified"
        row["requires_path_normalization"] = row["nonportable_path_occurrences"] > 0
        phase_candidates[phase_name].append(row)

    selected: list[dict[str, Any]] = []
    dedupe_counts: Counter[str] = Counter()
    for phase_name, candidates in phase_candidates.items():
        version_task_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            version_task_groups[(row["version"], row["task_id"])].append(row)
        version_task_best = [
            min(group, key=selection_rank) for group in version_task_groups.values()
        ]
        dedupe_counts[f"{phase_name}_before_version_task_dedupe"] = len(candidates)
        dedupe_counts[f"{phase_name}_after_version_task_dedupe"] = len(version_task_best)

        task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in version_task_best:
            task_groups[row["task_id"]].append(row)
        for group in task_groups.values():
            selected.extend(
                choose_with_source_diversity(group, args.max_per_task_per_phase)
            )

    selected.sort(
        key=lambda row: (row["phase"], row["split"], row["task_id"], selection_rank(row))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary: dict[str, Any] = {
        "input_candidates": len(tokens),
        "selected": len(selected),
        "selected_unique_task_ids": len({row["task_id"] for row in selected}),
        "selected_input_tokens": sum(row["input_tokens"] for row in selected),
        "selected_supervised_tokens": sum(row["supervised_tokens"] for row in selected),
        "selected_weighted_supervised_ratio": round(
            sum(row["supervised_tokens"] for row in selected)
            / max(1, sum(row["input_tokens"] for row in selected)),
            6,
        ),
        "selected_requires_rejudge": sum(row["requires_rejudge"] for row in selected),
        "selected_requires_path_normalization": sum(
            row["requires_path_normalization"] for row in selected
        ),
        "max_per_task_per_phase": args.max_per_task_per_phase,
        "split_shares": {
            "train": args.train_share,
            "validation": args.validation_share,
            "test": round(1 - args.train_share - args.validation_share, 6),
        },
        "production_max_tokens": args.production_max_tokens,
        "phase_definitions": [
            {
                "name": name,
                "input_tokens_gt": lower,
                "input_tokens_lte": upper,
                "min_supervised_ratio": min_ratio,
            }
            for name, lower, upper, min_ratio in phases
        ],
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "dedupe": dict(sorted(dedupe_counts.items())),
        "selected_by_split": {},
        "selected_source_families": dict(
            sorted(Counter(row["source_family"] for row in selected).items())
        ),
        "selected_types": dict(sorted(Counter(row["type"] for row in selected).items())),
        "selected_quality_tiers": dict(
            sorted(Counter(row["quality_tier"] for row in selected).items())
        ),
        "by_phase": {},
    }
    for split_name in ("train", "validation", "test"):
        rows = [row for row in selected if row["split"] == split_name]
        summary["selected_by_split"][split_name] = {
            "records": len(rows),
            "unique_task_ids": len({row["task_id"] for row in rows}),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
        }
    for phase_name, _, _, _ in phases:
        rows = [row for row in selected if row["phase"] == phase_name]
        summary["by_phase"][phase_name] = {
            "records": len(rows),
            "unique_task_ids": len({row["task_id"] for row in rows}),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "supervised_tokens": sum(row["supervised_tokens"] for row in rows),
            "requires_rejudge": sum(row["requires_rejudge"] for row in rows),
            "requires_path_normalization": sum(
                row["requires_path_normalization"] for row in rows
            ),
            "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
            "source_families": dict(
                sorted(Counter(row["source_family"] for row in rows).items())
            ),
            "types": dict(sorted(Counter(row["type"] for row in rows).items())),
            "quality_tiers": dict(
                sorted(Counter(row["quality_tier"] for row in rows).items())
            ),
        }
    with args.summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
