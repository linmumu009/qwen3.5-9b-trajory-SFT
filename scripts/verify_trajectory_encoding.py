"""Verify Qwen3.5 trajectory tokenization and loss masking without printing content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swift import get_processor, get_template


def as_list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/Qwen3.5-9B")
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    processor = get_processor(args.model)
    template = get_template(processor, loss_scale="default+ignore_empty_think")
    template.set_mode("train")
    tokenizer = processor.tokenizer

    reports = []
    with args.dataset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            tools = row.get("tools")
            if not isinstance(tools, (str, list)):
                raise RuntimeError(f"record {line_number}: explicit tools contract is missing")
            parsed_tools = json.loads(tools) if isinstance(tools, str) else tools
            tool_names = [
                tool.get("function", {}).get("name")
                for tool in parsed_tools
                if isinstance(tool, dict)
            ]
            if tool_names != ["bash", "read", "write", "edit"]:
                raise RuntimeError(
                    f"record {line_number}: unexpected tools contract names: {tool_names!r}"
                )
            encoded = template.encode(row)
            input_ids = as_list(encoded["input_ids"])
            labels = as_list(encoded["labels"])
            supervised_ids = [label for label in labels if label != -100]
            input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
            system_prefix = input_text.split("<|im_start|>user", 1)[0]

            has_tool_call_input = "<tool_call>" in input_text
            has_tool_response_input = "<tool_response>" in input_text
            tool_call_supervised = "<tool_call>" in supervised_text
            tool_response_supervised = "<tool_response>" in supervised_text
            has_thinking_input = "<think>" in input_text and "</think>" in input_text
            tool_contract_in_system = (
                "<tools>" in system_prefix
                and all(name in system_prefix for name in ("bash", "read", "write", "edit"))
            )
            if not has_tool_call_input or not has_tool_response_input or not has_thinking_input:
                raise RuntimeError(
                    f"record {line_number}: trajectory markers missing "
                    f"tool_call={has_tool_call_input} "
                    f"tool_response={has_tool_response_input} "
                    f"thinking={has_thinking_input}"
                )
            if not tool_call_supervised:
                raise RuntimeError(f"record {line_number}: tool call is not supervised")
            if tool_response_supervised:
                raise RuntimeError(f"record {line_number}: tool response unexpectedly contributes to loss")
            if not supervised_ids:
                raise RuntimeError(f"record {line_number}: no supervised tokens")
            if not tool_contract_in_system:
                raise RuntimeError(f"record {line_number}: tools contract was not injected into system")

            reports.append(
                {
                    "record": line_number,
                    "input_tokens": len(input_ids),
                    "supervised_tokens": len(supervised_ids),
                    "supervised_ratio": round(len(supervised_ids) / len(input_ids), 6),
                    "tool_call_supervised": tool_call_supervised,
                    "tool_response_supervised": tool_response_supervised,
                    "thinking_present": has_thinking_input,
                    "tool_contract_in_system": tool_contract_in_system,
                }
            )

    if not reports:
        raise SystemExit("Dataset is empty")
    result = {
        "records": len(reports),
        "input_token_min": min(item["input_tokens"] for item in reports),
        "input_token_max": max(item["input_tokens"] for item in reports),
        "supervised_token_min": min(item["supervised_tokens"] for item in reports),
        "supervised_token_max": max(item["supervised_tokens"] for item in reports),
        "details": reports,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
