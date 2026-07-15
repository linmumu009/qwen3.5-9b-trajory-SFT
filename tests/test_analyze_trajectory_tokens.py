from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_trajectory_tokens import (  # noqa: E402
    length_bucket,
    quality_tier,
    ratio_bucket,
    source_family,
)


def test_length_bucket_boundaries():
    assert length_bucket(1) == "1-2048"
    assert length_bucket(2048) == "1-2048"
    assert length_bucket(2049) == "2049-4096"
    assert length_bucket(262145) == ">262144"


def test_ratio_bucket_boundaries():
    assert ratio_bucket(0.049) == "0%-<5%"
    assert ratio_bucket(0.05) == "5%-<10%"
    assert ratio_bucket(0.10) == "10%-<20%"
    assert ratio_bucket(0.40) == ">=40%"


def test_quality_tier_prefers_sql_evidence():
    row = {
        "type": "kb",
        "evidence": {"agent_sql_ok": True, "is_report": True},
    }
    assert quality_tier(row) == "sql_result_verified"
    assert quality_tier({"type": "kb", "evidence": {}}) == "kb_rule_only"
    assert quality_tier({"type": "hybrid", "evidence": {"is_report": True}}) == (
        "report_rule_only"
    )


def test_source_family_mapping():
    assert source_family("qwen3.6-27B_20260628_v15_openai.jsonl") == "qwen3.6-27b"
    assert source_family("deepseek_20260628_v15_openai.jsonl") == "deepseek-v4-pro"
    assert source_family("glm52_20260628_v15_openai.jsonl") == "glm-5.2"
    assert source_family("qwen37max_20260628_v15_openai.jsonl") == "qwen3.7-max"
