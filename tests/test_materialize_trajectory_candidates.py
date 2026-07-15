from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from materialize_trajectory_candidates import materialize  # noqa: E402


def test_materialize_authoritative_catalog_to_one_swift_jsonl(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    source = data_dir / "source.jsonl"
    record = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "inspect",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"ls"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        json.dumps({"source_file": source.name, "source_line": 1}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "candidates.jsonl"

    summary = materialize(catalog, data_dir, output)
    row = json.loads(output.read_text(encoding="utf-8"))

    assert summary["records"] == 1
    assert summary["source_files"] == 1
    assert len(summary["sha256"]) == 64
    assert set(row) == {"tools", "messages"}
    assert json.loads(row["tools"])[0]["function"]["name"] == "bash"
    assert [message["role"] for message in row["messages"]][-3:] == [
        "tool_call",
        "tool_response",
        "assistant",
    ]
