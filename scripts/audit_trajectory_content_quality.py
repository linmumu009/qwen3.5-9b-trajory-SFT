"""Audit trajectory portability, safety and tool-output noise without printing content."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_trajectory_sft import iter_jsonl


TOOL_ERROR_RE = re.compile(
    r"traceback|\berror\b|no such file|command not found|permission denied|timed? out|"
    r"syntax error|exception|segmentation fault",
    re.IGNORECASE,
)
DESTRUCTIVE_RE = re.compile(
    r"\brm\s+-[^\n]*r[^\n]*f|\bmkfs\b|\bdd\s+if=|\bshutdown\b|\breboot\b|"
    r"\bchmod\s+777\b|\bchown\s+-R\b|(?:curl|wget)[^\n|]*\|\s*(?:ba)?sh\b",
    re.IGNORECASE,
)
NETWORK_RE = re.compile(
    r"\b(?:curl|wget|git\s+clone|pip\s+install|apt(?:-get)?\s+install|yum\s+install)\b",
    re.IGNORECASE,
)
NONPORTABLE_PATH_RE = re.compile(
    r"/(?:private/)?tmp/pi_sandbox_[^\s'\"`]+|/data/[^\s'\"`]+",
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
)


def percentile(values: list[int | float], ratio: float) -> int | float:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * ratio)]


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidate-catalog", type=Path, required=True)
    parser.add_argument("--record-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def load_catalog(path: Path) -> tuple[dict[str, set[int]], dict[tuple[str, int], dict[str, Any]]]:
    selected: dict[str, set[int]] = defaultdict(set)
    metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in iter_jsonl(path):
        filename = row["source_file"]
        line = int(row["source_line"])
        selected[filename].add(line)
        metadata[(filename, line)] = row
    return selected, metadata


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    role_chars: Counter[str] = Counter()
    tool_response_lengths: list[int] = []
    tool_error_messages = 0
    destructive_commands = 0
    network_commands = 0
    nonportable_paths = 0
    secret_hits = 0
    tool_calls: list[str] = []
    call_id_to_name: dict[str, str] = {}
    final_assistant_chars = 0
    reasoning_chars = 0
    system_chars = 0
    system_parts: list[str] = []

    for message in record.get("messages") or []:
        role = str(message.get("role"))
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        role_chars[role] += len(text)
        nonportable_paths += len(NONPORTABLE_PATH_RE.findall(text))
        if contains_secret(text):
            secret_hits += 1

        if role == "system":
            system_chars += len(text)
            system_parts.append(text)
        elif role == "assistant":
            final_assistant_chars = len(text)
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str):
                reasoning_chars += len(reasoning)
                nonportable_paths += len(NONPORTABLE_PATH_RE.findall(reasoning))
                if contains_secret(reasoning):
                    secret_hits += 1
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    argument_text = arguments
                    try:
                        argument_obj = json.loads(arguments)
                    except Exception:
                        argument_obj = arguments
                else:
                    argument_obj = arguments
                    argument_text = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
                signature = json.dumps(
                    {"name": name, "arguments": argument_obj},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                tool_calls.append(signature)
                if isinstance(call.get("id"), str):
                    call_id_to_name[call["id"]] = name
                if name == "bash":
                    destructive_commands += int(bool(DESTRUCTIVE_RE.search(argument_text)))
                    network_commands += int(bool(NETWORK_RE.search(argument_text)))
                nonportable_paths += len(NONPORTABLE_PATH_RE.findall(argument_text))
                if contains_secret(argument_text):
                    secret_hits += 1
        elif role == "tool":
            tool_response_lengths.append(len(text))
            if TOOL_ERROR_RE.search(text):
                tool_error_messages += 1

    duplicate_tool_calls = len(tool_calls) - len(set(tool_calls))
    consecutive_duplicate_tool_calls = sum(
        left == right for left, right in zip(tool_calls, tool_calls[1:])
    )
    total_chars = sum(role_chars.values()) + reasoning_chars
    tool_response_chars = role_chars["tool"]
    system_text = "\n".join(system_parts)
    return {
        "top_level_tools_present": isinstance(record.get("tools"), list),
        "system_chars": system_chars,
        "system_fingerprint": hashlib.sha256(system_text.encode("utf-8")).hexdigest(),
        "system_tool_contract_present": all(
            re.search(rf"\b{name}\b", system_text, re.IGNORECASE)
            for name in ("bash", "read", "write", "edit")
        ),
        "total_chars": total_chars,
        "assistant_content_chars": role_chars["assistant"],
        "reasoning_chars": reasoning_chars,
        "tool_response_chars": tool_response_chars,
        "tool_response_char_ratio": round(tool_response_chars / total_chars, 6) if total_chars else 0,
        "tool_response_messages": len(tool_response_lengths),
        "tool_response_max_chars": max(tool_response_lengths, default=0),
        "tool_response_over_4096": sum(value > 4096 for value in tool_response_lengths),
        "tool_response_over_16384": sum(value > 16384 for value in tool_response_lengths),
        "tool_response_over_65536": sum(value > 65536 for value in tool_response_lengths),
        "tool_error_messages": tool_error_messages,
        "duplicate_tool_calls": duplicate_tool_calls,
        "consecutive_duplicate_tool_calls": consecutive_duplicate_tool_calls,
        "destructive_commands": destructive_commands,
        "network_commands": network_commands,
        "nonportable_path_occurrences": nonportable_paths,
        "secret_pattern_hits": secret_hits,
        "final_assistant_chars": final_assistant_chars,
    }


def main() -> None:
    args = parse_args()
    selected, metadata = load_catalog(args.candidate_catalog)
    args.record_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    record_tmp = args.record_output.with_suffix(args.record_output.suffix + ".tmp")
    summary_tmp = args.summary_output.with_suffix(args.summary_output.suffix + ".tmp")
    rows: list[dict[str, Any]] = []

    with record_tmp.open("w", encoding="utf-8") as output:
        for data_file in sorted(args.data_dir.glob("*.jsonl")):
            wanted = selected.get(data_file.name, set())
            if not wanted:
                continue
            for line_number, record in iter_jsonl(data_file):
                if line_number not in wanted:
                    continue
                row = {
                    key: metadata[(data_file.name, line_number)][key]
                    for key in (
                        "source_file",
                        "source_line",
                        "source_family",
                        "version",
                        "task_id",
                        "type",
                        "quality_tier",
                        "input_tokens",
                        "supervised_tokens",
                        "supervised_ratio",
                    )
                }
                row.update(analyze_record(record))
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows.append(row)

    if len(rows) != len(metadata):
        raise RuntimeError(f"audited {len(rows)} rows but catalog contains {len(metadata)}")
    os.replace(record_tmp, args.record_output)

    fields = (
        "tool_response_char_ratio",
        "tool_response_max_chars",
        "tool_error_messages",
        "duplicate_tool_calls",
        "consecutive_duplicate_tool_calls",
        "destructive_commands",
        "network_commands",
        "nonportable_path_occurrences",
        "secret_pattern_hits",
        "final_assistant_chars",
    )
    distributions = {}
    for field in fields:
        values = [row[field] for row in rows]
        distributions[field] = {
            "records_nonzero": sum(value > 0 for value in values),
            "total": round(sum(values), 6),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values, default=0),
        }

    summary = {
        "records": len(rows),
        "top_level_tools_present": sum(row["top_level_tools_present"] for row in rows),
        "system_message_present": sum(row["system_chars"] > 0 for row in rows),
        "distinct_system_prompts": len({row["system_fingerprint"] for row in rows}),
        "system_tool_contract_present": sum(
            row["system_tool_contract_present"] for row in rows
        ),
        "tool_response_over_4096_messages": sum(row["tool_response_over_4096"] for row in rows),
        "tool_response_over_16384_messages": sum(row["tool_response_over_16384"] for row in rows),
        "tool_response_over_65536_messages": sum(row["tool_response_over_65536"] for row in rows),
        "distributions": distributions,
        "by_source_family": {},
        "by_type": {},
        "by_quality_tier": {},
    }
    for output_name, field in (
        ("by_source_family", "source_family"),
        ("by_type", "type"),
        ("by_quality_tier", "quality_tier"),
    ):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        summary[output_name] = {
            name: {
                "records": len(group),
                "tool_error_records": sum(row["tool_error_messages"] > 0 for row in group),
                "duplicate_call_records": sum(row["duplicate_tool_calls"] > 0 for row in group),
                "nonportable_path_records": sum(
                    row["nonportable_path_occurrences"] > 0 for row in group
                ),
                "destructive_command_records": sum(row["destructive_commands"] > 0 for row in group),
                "secret_pattern_records": sum(row["secret_pattern_hits"] > 0 for row in group),
                "tool_response_ratio_p50": percentile(
                    [row["tool_response_char_ratio"] for row in group], 0.50
                ),
                "tool_response_ratio_p95": percentile(
                    [row["tool_response_char_ratio"] for row in group], 0.95
                ),
            }
            for name, group in sorted(groups.items())
        }

    with summary_tmp.open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(summary_tmp, args.summary_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
