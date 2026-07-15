from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from materialize_correct_verdict_trajectories import (  # noqa: E402
    materialize_correct,
    verdict_filename,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_verdict_filename_maps_all_source_families() -> None:
    assert verdict_filename("qwen3.6-27B_20260628_v20_openai.jsonl", "20260628_v20") == (
        "qwen3.6-27B_20260628_v20_openai.jsonl"
    )
    assert verdict_filename("deepseek_20260628_v15_openai.jsonl", "20260628_v15") == (
        "aliyun-deepseek-v4-pro_20260628_v15_openai.jsonl"
    )


def test_materialize_correct_joins_verdict_to_full_raw_record(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    manifests_dir = tmp_path / "manifests"
    verdicts_dir = tmp_path / "verdicts"
    raw_file = data_dir / "qwen3.6-27B_20260628_v15_openai.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "question one"}]},
        {"messages": [{"role": "user", "content": "question two"}]},
    ]
    write_jsonl(raw_file, rows)
    write_jsonl(
        manifests_dir / "manifest.jsonl",
        [
            {"v": "20260628_v15", "task_id": "task-1", "instruction": "question one"},
            {"v": "20260628_v15", "task_id": "task-2", "instruction": "question two"},
        ],
    )
    write_jsonl(
        verdicts_dir / "qwen3.6-27B_20260628_v15_openai.jsonl",
        [
            {"v": "20260628_v15", "task_id": "task-1", "verdict": "correct"},
            {"v": "20260628_v15", "task_id": "task-2", "verdict": "incorrect"},
        ],
    )
    output = tmp_path / "correct.jsonl"

    summary = materialize_correct(data_dir, manifests_dir, verdicts_dir, output)
    selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["correct_records"] == 1
    assert summary["raw_records"] == 2
    assert summary["verdict_records"] == 2
    assert selected == [rows[0]]
