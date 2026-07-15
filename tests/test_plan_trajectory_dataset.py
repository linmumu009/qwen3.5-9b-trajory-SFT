from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plan_trajectory_dataset import (  # noqa: E402
    choose_with_source_diversity,
    phase_for,
    stable_split,
)


def row(source, tokens=4096):
    return {
        "source_family": source,
        "source_file": f"{source}.jsonl",
        "source_line": 1,
        "quality_tier": "kb_rule_only",
        "network_commands": 0,
        "tool_error_messages": 0,
        "duplicate_tool_calls": 0,
        "nonportable_path_occurrences": 0,
        "input_tokens": tokens,
        "supervised_ratio": 0.2,
        "final_assistant_chars": 100,
    }


def test_split_is_stable_at_task_level():
    first = stable_split("task-1", 0.85, 0.075)
    assert first == stable_split("task-1", 0.85, 0.075)
    assert first in {"train", "validation", "test"}


def test_phase_boundaries():
    assert phase_for({"input_tokens": 8192}) == ("core_8k", 0.10)
    assert phase_for({"input_tokens": 8193}) == ("extension_16k", 0.10)
    assert phase_for({"input_tokens": 32769}) is None


def test_cap_prefers_source_diversity():
    rows = [row("qwen"), row("qwen", 4100), row("glm", 4200), row("deepseek", 4300)]
    chosen = choose_with_source_diversity(rows, 3)
    assert len(chosen) == 3
    assert len({item["source_family"] for item in chosen}) == 3
