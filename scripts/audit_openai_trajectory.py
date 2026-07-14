"""Stream-audit OpenAI-style trajectory JSONL without printing private content."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def type_name(value: Any) -> str:
    return type(value).__name__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL file or directory")
    parser.add_argument("--max-records-per-file", type=int, default=0)
    return parser.parse_args()


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.jsonl"))


def main() -> None:
    args = parse_args()
    files = iter_files(args.input)
    if not files:
        raise SystemExit(f"No JSONL files found: {args.input}")

    summary: Counter[str] = Counter()
    top_keys: Counter[str] = Counter()
    message_keys: dict[str, Counter[str]] = {}
    roles: Counter[str] = Counter()
    content_types: Counter[str] = Counter()
    tool_call_types: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    tool_argument_keys: Counter[str] = Counter()
    final_roles: Counter[str] = Counter()
    file_records: dict[str, int] = {}

    for path in files:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                if args.max_records_per_file and count >= args.max_records_per_file:
                    break
                count += 1
                summary["records"] += 1
                try:
                    record = json.loads(line)
                except Exception as exc:
                    summary["invalid_json"] += 1
                    print(f"INVALID_JSON {path.name}:{line_number} {exc}")
                    continue
                if not isinstance(record, dict):
                    summary["non_object_records"] += 1
                    continue
                top_keys.update(record.keys())
                messages = record.get("messages")
                if not isinstance(messages, list):
                    summary["records_without_messages_list"] += 1
                    continue
                summary["messages"] += len(messages)
                if record.get("tools") is not None:
                    summary["records_with_tools"] += 1
                if record.get("id") is not None:
                    summary["records_with_id"] += 1
                if record.get("task_id") is not None:
                    summary["records_with_task_id"] += 1
                assistant_turns = 0
                assistant_tool_turns = 0
                tool_messages = 0
                known_tool_call_ids: set[str] = set()
                answered_tool_call_ids: set[str] = set()
                for message in messages:
                    if not isinstance(message, dict):
                        summary["non_object_messages"] += 1
                        continue
                    role = str(message.get("role", "<missing>"))
                    roles[role] += 1
                    message_keys.setdefault(role, Counter()).update(message.keys())
                    content_types[f"{role}:{type_name(message.get('content'))}"] += 1
                    if role == "assistant":
                        assistant_turns += 1
                        if not message.get("content"):
                            summary["assistant_messages_with_empty_content"] += 1
                        if not message.get("reasoning_content"):
                            summary["assistant_messages_without_reasoning"] += 1
                        calls = message.get("tool_calls")
                        if calls:
                            assistant_tool_turns += 1
                            tool_call_types[type_name(calls)] += 1
                            for call in calls if isinstance(calls, list) else []:
                                if not isinstance(call, dict):
                                    summary["non_object_tool_calls"] += 1
                                    continue
                                call_id = call.get("id")
                                if isinstance(call_id, str):
                                    if call_id in known_tool_call_ids:
                                        summary["duplicate_tool_call_ids"] += 1
                                    known_tool_call_ids.add(call_id)
                                function = call.get("function")
                                if not isinstance(function, dict):
                                    summary["tool_calls_without_function_object"] += 1
                                    continue
                                name = str(function.get("name", "<missing>"))
                                tool_names[name] += 1
                                arguments = function.get("arguments", {})
                                if isinstance(arguments, str):
                                    try:
                                        arguments = json.loads(arguments)
                                    except Exception:
                                        summary["invalid_tool_arguments_json"] += 1
                                        continue
                                if not isinstance(arguments, dict):
                                    summary["non_object_tool_arguments"] += 1
                                    continue
                                for key in arguments:
                                    tool_argument_keys[f"{name}:{key}"] += 1
                    elif role == "tool":
                        tool_messages += 1
                        call_id = message.get("tool_call_id")
                        if not isinstance(call_id, str) or call_id not in known_tool_call_ids:
                            summary["tool_responses_with_unknown_id"] += 1
                        elif call_id in answered_tool_call_ids:
                            summary["duplicate_tool_responses"] += 1
                        else:
                            answered_tool_call_ids.add(call_id)
                final_role = str(messages[-1].get("role", "<missing>")) if messages else "<empty>"
                final_roles[final_role] += 1
                unanswered = known_tool_call_ids - answered_tool_call_ids
                if unanswered:
                    summary["records_with_unanswered_tool_calls"] += 1
                    summary["unanswered_tool_calls"] += len(unanswered)
                if assistant_turns:
                    summary["records_with_assistant"] += 1
                if assistant_tool_turns:
                    summary["records_with_assistant_tool_calls"] += 1
                if tool_messages:
                    summary["records_with_tool_messages"] += 1
                if assistant_tool_turns and tool_messages:
                    summary["records_with_tool_roundtrip"] += 1
        file_records[path.name] = count

    output = {
        "files": len(files),
        "file_records": file_records,
        "summary": dict(summary),
        "top_level_keys": dict(top_keys),
        "roles": dict(roles),
        "content_types": dict(content_types),
        "message_keys_by_role": {
            role: dict(counter) for role, counter in message_keys.items()
        },
        "tool_call_container_types": dict(tool_call_types),
        "tool_names": dict(tool_names),
        "tool_argument_keys": dict(tool_argument_keys),
        "final_roles": dict(final_roles),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
