"""Convert OpenAI tool trajectories into strict ms-swift agent SFT JSONL."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION_RE = re.compile(r"_(\d{8}_v\d+)_openai\.jsonl$")

CANONICAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in the current sandbox workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds.",
                        "minimum": 1,
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read text from a file in the sandbox workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read."},
                    "offset": {
                        "type": "integer",
                        "description": "Optional zero-based line offset.",
                        "minimum": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum number of lines to return.",
                        "minimum": 1,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Create or overwrite a text file in the sandbox workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write."},
                    "content": {"type": "string", "description": "Complete file content."},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": (
                "Edit a text file using either an edits array or one oldText/newText replacement."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit."},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {"type": "string"},
                                "newText": {"type": "string"},
                            },
                            "required": ["oldText", "newText"],
                            "additionalProperties": False,
                        },
                    },
                    "oldText": {"type": "string"},
                    "newText": {"type": "string"},
                },
                "required": ["path"],
                "anyOf": [
                    {"required": ["edits"]},
                    {"required": ["oldText", "newText"]},
                ],
                "additionalProperties": False,
            },
        },
    },
]
CANONICAL_TOOLS_JSON = json.dumps(
    CANONICAL_TOOLS, ensure_ascii=False, separators=(",", ":")
)


def make_training_row(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the documented ms-swift agent row with an explicit tool contract."""
    return {"tools": CANONICAL_TOOLS_JSON, "messages": messages}


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            yield line_number, json.loads(line)


def jsonl_files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(path.glob("*.jsonl"))


def detect_version(path: Path) -> str:
    match = VERSION_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer version from input filename: {path.name}")
    return match.group(1)


def first_user_content(record: dict[str, Any]) -> str:
    for message in record.get("messages") or []:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    raise ValueError("record has no string user message")


def load_manifest_data(
    path: Path, version: str
) -> tuple[dict[str, set[str]], list[tuple[str, str]], dict[str, set[str]]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    source_lookup: dict[str, set[str]] = defaultdict(set)
    sortable: list[tuple[str, str, str]] = []
    for file in jsonl_files(path):
        for _, row in iter_jsonl(file):
            if row.get("v") != version:
                continue
            instruction = row.get("instruction")
            task_id = row.get("task_id")
            if isinstance(instruction, str) and isinstance(task_id, str):
                normalized = normalize_text(instruction)
                lookup[normalized].add(task_id)
                output = row.get("output")
                sort_key = Path(output).name if isinstance(output, str) else task_id
                sortable.append((sort_key, normalized, task_id))
                if isinstance(output, str):
                    source_lookup[Path(output).name].add(task_id)
    sortable.sort(key=lambda item: item[0])
    ordered = [(normalized, task_id) for _, normalized, task_id in sortable]
    return lookup, ordered, source_lookup


def load_verdicts(path: Path, version: str) -> tuple[dict[str, dict[str, Any]], set[str]]:
    rows: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for file in jsonl_files(path):
        for _, row in iter_jsonl(file):
            if row.get("v") != version:
                continue
            task_id = row.get("task_id")
            if not isinstance(task_id, str):
                continue
            previous = rows.get(task_id)
            if previous is not None and previous.get("verdict") != row.get("verdict"):
                conflicts.add(task_id)
            rows[task_id] = row
    return rows, conflicts


def parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"tool arguments must be a JSON object, got {type(value).__name__}")
    return value


def convert_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    converted: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    known_tool_call_ids: set[str] = set()
    answered_tool_call_ids: set[str] = set()

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"{role} content must be a string")

        if role in {"system", "user"}:
            converted.append({"role": role, "content": content})
        elif role == "assistant":
            reasoning = message.get("reasoning_content") or ""
            if not isinstance(reasoning, str):
                raise ValueError("assistant reasoning_content must be a string")
            assistant_content = content.strip()
            if reasoning.strip():
                assistant_content = f"<think>\n{reasoning.strip()}\n</think>\n\n{assistant_content}"
                stats["reasoning_messages"] += 1
            if assistant_content:
                converted.append({"role": "assistant", "content": assistant_content, "loss": True})
                stats["assistant_content_messages"] += 1

            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ValueError("assistant tool_calls must be a list")
            for call in tool_calls:
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    raise ValueError("tool call must contain a function object")
                function = call["function"]
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("tool call function.name must be a non-empty string")
                arguments = parse_arguments(function.get("arguments", {}))
                tool_call = {"name": name, "arguments": arguments}
                converted.append(
                    {
                        "role": "tool_call",
                        "content": json.dumps(tool_call, ensure_ascii=False, separators=(",", ":")),
                        "loss": True,
                    }
                )
                call_id = call.get("id")
                if isinstance(call_id, str):
                    if call_id in known_tool_call_ids:
                        raise ValueError(f"duplicate tool call id: {call_id}")
                    known_tool_call_ids.add(call_id)
                stats["tool_calls"] += 1
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in known_tool_call_ids:
                raise ValueError(f"tool response references unknown call id: {call_id!r}")
            if call_id in answered_tool_call_ids:
                raise ValueError(f"duplicate tool response for call id: {call_id}")
            answered_tool_call_ids.add(call_id)
            converted.append({"role": "tool_response", "content": content})
            stats["tool_responses"] += 1
        else:
            raise ValueError(f"unsupported role: {role!r}")

    unanswered = known_tool_call_ids - answered_tool_call_ids
    if unanswered:
        raise ValueError(f"unanswered tool calls: {len(unanswered)}")
    if not any(message["role"] in {"assistant", "tool_call"} for message in converted):
        raise ValueError("converted record has no trainable assistant output")
    stats["output_messages"] = len(converted)
    stats["characters"] = sum(len(message["content"]) for message in converted)
    return converted, dict(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--allowed-verdict", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--select-shortest", action="store_true")
    parser.add_argument("--require-tool-roundtrip", action="store_true")
    parser.add_argument(
        "--skip-conversion-errors",
        action="store_true",
        help="Exclude malformed trajectories after reporting them; identity alignment remains strict.",
    )
    return parser.parse_args()


def conversion_error_category(exc: Exception) -> str:
    message = str(exc)
    categories = (
        "unknown call id",
        "duplicate tool response",
        "duplicate tool call id",
        "unanswered tool calls",
        "tool arguments must be a JSON object",
        "assistant tool_calls must be a list",
        "tool call must contain a function object",
        "tool call function.name must be a non-empty string",
        "content must be a string",
        "unsupported role",
        "no trainable assistant output",
    )
    for category in categories:
        if category in message:
            return category
    if isinstance(exc, json.JSONDecodeError):
        return "invalid tool arguments JSON"
    return type(exc).__name__


def main() -> None:
    args = parse_args()
    version = detect_version(args.input)
    allowed_verdicts = set(args.allowed_verdict or ["correct"])
    manifest_lookup, manifest_order, manifest_source_lookup = load_manifest_data(args.manifests, version)
    verdicts, verdict_conflicts = load_verdicts(args.verdicts, version)
    if verdict_conflicts:
        raise SystemExit(f"Conflicting verdicts for {len(verdict_conflicts)} task IDs")

    report: dict[str, Any] = {
        "input": args.input.name,
        "version": version,
        "allowed_verdicts": sorted(allowed_verdicts),
        "records": 0,
        "manifest_rows": len(manifest_order),
        "aligned": 0,
        "source_verified": 0,
        "position_verified": 0,
        "position_mismatch": 0,
        "unmatched": 0,
        "ambiguous": 0,
        "missing_verdict": 0,
        "conversion_errors": 0,
        "conversion_error_categories": Counter(),
        "verdict_distribution": Counter(),
        "eligible": 0,
        "excluded_without_tool_roundtrip": 0,
        "written": 0,
    }
    eligible: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []

    for line_number, record in iter_jsonl(args.input):
        report["records"] += 1
        try:
            user_content = first_user_content(record)
        except ValueError:
            report["unmatched"] += 1
            continue
        normalized_user = normalize_text(user_content)
        task_id = None
        source = record.get("_source")
        source_candidates = manifest_source_lookup.get(source, set()) if isinstance(source, str) else set()
        if len(source_candidates) == 1:
            task_id = next(iter(source_candidates))
            report["source_verified"] += 1
        positional_task_id = None
        if task_id is None and line_number <= len(manifest_order):
            positional_instruction, positional_task_id_candidate = manifest_order[line_number - 1]
            if positional_instruction == normalized_user:
                positional_task_id = positional_task_id_candidate
                report["position_verified"] += 1
            else:
                report["position_mismatch"] += 1
        candidates = manifest_lookup.get(normalized_user, set())
        if not candidates:
            report["unmatched"] += 1
            continue
        if task_id is not None:
            pass
        elif positional_task_id is not None:
            task_id = positional_task_id
        elif len(candidates) != 1:
            report["ambiguous"] += 1
            continue
        else:
            task_id = next(iter(candidates))
        report["aligned"] += 1
        verdict_row = verdicts.get(task_id)
        if verdict_row is None:
            report["missing_verdict"] += 1
            continue
        verdict = str(verdict_row.get("verdict"))
        report["verdict_distribution"][verdict] += 1
        if verdict not in allowed_verdicts:
            continue
        try:
            converted, conversion_stats = convert_messages(record.get("messages") or [])
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            report["conversion_errors"] += 1
            report["conversion_error_categories"][conversion_error_category(exc)] += 1
            continue
        if args.require_tool_roundtrip and not (
            conversion_stats.get("tool_calls", 0) > 0
            and conversion_stats.get("tool_calls") == conversion_stats.get("tool_responses")
        ):
            report["excluded_without_tool_roundtrip"] += 1
            continue
        training_row = make_training_row(converted)
        metadata_row = {
            "source_file": args.input.name,
            "source_line": line_number,
            "version": version,
            "task_id": task_id,
            "type": verdict_row.get("type"),
            "verdict": verdict,
            "conversion": conversion_stats,
        }
        eligible.append((conversion_stats["characters"], line_number, training_row, metadata_row))

    report["eligible"] = len(eligible)
    fatal = report["unmatched"] + report["ambiguous"] + report["missing_verdict"]
    if not args.skip_conversion_errors:
        fatal += report["conversion_errors"]
    if fatal:
        report["verdict_distribution"] = dict(sorted(report["verdict_distribution"].items()))
        report["conversion_error_categories"] = dict(sorted(report["conversion_error_categories"].items()))
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit("Strict preparation failed; no output written")

    if args.select_shortest:
        eligible.sort(key=lambda item: (item[0], item[1]))
    else:
        eligible.sort(key=lambda item: item[1])
    if args.max_samples:
        eligible = eligible[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as data_out, args.metadata_output.open(
        "w", encoding="utf-8"
    ) as metadata_out:
        for _, _, training_row, metadata_row in eligible:
            data_out.write(json.dumps(training_row, ensure_ascii=False, separators=(",", ":")) + "\n")
            metadata_out.write(json.dumps(metadata_row, ensure_ascii=False, separators=(",", ":")) + "\n")

    report["written"] = len(eligible)
    if eligible:
        sizes = [item[0] for item in eligible]
        report["written_character_min"] = min(sizes)
        report["written_character_max"] = max(sizes)
    report["verdict_distribution"] = dict(sorted(report["verdict_distribution"].items()))
    report["conversion_error_categories"] = dict(sorted(report["conversion_error_categories"].items()))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
