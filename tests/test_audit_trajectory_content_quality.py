from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_trajectory_content_quality import analyze_record  # noqa: E402


def test_quality_audit_emits_counts_without_content():
    record = {
        "messages": [
            {"role": "system", "content": "Use the tools."},
            {"role": "user", "content": "Inspect the sandbox."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Check the file first.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"rm -rf /private/tmp/pi_sandbox_abc/out"}',
                        },
                    },
                    {
                        "id": "call-2",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"rm -rf /private/tmp/pi_sandbox_abc/out"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ERROR: failed"},
            {"role": "tool", "tool_call_id": "call-2", "content": "ERROR: failed"},
            {"role": "assistant", "content": "Done."},
        ]
    }
    result = analyze_record(record)

    assert result["tool_error_messages"] == 2
    assert result["system_tool_contract_present"] is False
    assert result["duplicate_tool_calls"] == 1
    assert result["consecutive_duplicate_tool_calls"] == 1
    assert result["destructive_commands"] == 2
    assert result["nonportable_path_occurrences"] == 2
    assert result["final_assistant_chars"] == 5
    serialized = str(result)
    assert "pi_sandbox_abc" not in serialized
    assert "rm -rf" not in serialized
