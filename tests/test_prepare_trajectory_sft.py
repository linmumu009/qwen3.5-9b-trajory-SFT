from __future__ import annotations

import json

from scripts.prepare_trajectory_sft import convert_messages, make_training_row


def test_convert_openai_tool_trajectory_to_swift_agent_messages() -> None:
    source = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "inspect data",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        {"role": "assistant", "content": "answer", "reasoning_content": "conclude"},
    ]

    converted, stats = convert_messages(source)

    assert [message["role"] for message in converted] == [
        "system",
        "user",
        "assistant",
        "tool_call",
        "tool_response",
        "assistant",
    ]
    assert converted[2]["content"] == "<think>\ninspect data\n</think>\n\n"
    assert json.loads(converted[3]["content"]) == {
        "name": "bash",
        "arguments": {"command": "ls"},
    }
    assert converted[3]["loss"] is True
    assert "conclude" in converted[-1]["content"]
    assert converted[-1]["content"].endswith("answer")
    assert stats["tool_calls"] == stats["tool_responses"] == 1


def test_convert_rejects_unanswered_tool_call() -> None:
    source = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
    ]

    try:
        convert_messages(source)
    except ValueError as exc:
        assert "unanswered tool calls" in str(exc)
    else:
        raise AssertionError("expected unanswered tool call to fail")


def test_training_row_includes_canonical_tool_contract() -> None:
    row = make_training_row([{"role": "user", "content": "question"}])
    tools = json.loads(row["tools"])

    assert [tool["function"]["name"] for tool in tools] == ["bash", "read", "write", "edit"]
    assert tools[0]["function"]["parameters"]["required"] == ["command"]
    assert tools[2]["function"]["parameters"]["required"] == ["path", "content"]
