from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from materialize_trajectory_plan_sets import output_group  # noqa: E402


def row(phase: str, split: str, quality_tier: str = "kb_rule_only"):
    return {"phase": phase, "split": split, "quality_tier": quality_tier}


def test_output_group_keeps_training_review_and_heldout_isolated():
    assert output_group(row("core_8k", "train", "sql_result_verified")) == (
        "train_strong_verified"
    )
    assert output_group(row("extension_24k", "train")) == "train_review"
    assert output_group(row("core_8k", "validation")) == "heldout"
    assert output_group(row("extension_24k", "test")) == "heldout"
    assert output_group(row("long_32k_review", "train")) == "long_32k_review"
