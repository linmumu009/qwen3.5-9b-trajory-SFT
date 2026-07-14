from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_audit_reports_structure_without_message_content(tmp_path: Path) -> None:
    marker = "PRIVATE_TRAJECTORY_MARKER"
    data = tmp_path / "sample.jsonl"
    record = {
        "_source": "sft_v1_dwh_task_000001.jsonl",
        "messages": [
            {"role": "user", "content": marker},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": marker,
                "tool_calls": [],
            },
        ],
    }
    data.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    script = Path(__file__).parents[1] / "scripts" / "audit_openai_trajectory.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(data)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert marker not in completed.stdout
    report = json.loads(completed.stdout)
    assert report["top_level_keys"] == {"_source": 1, "messages": 1}
    assert report["message_keys_by_role"]["assistant"] == {
        "content": 1,
        "reasoning_content": 1,
        "role": 1,
        "tool_calls": 1,
    }
