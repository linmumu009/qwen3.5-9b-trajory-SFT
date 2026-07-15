from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rejudge_trajectory_candidates import (  # noqa: E402
    classify_candidate,
    coverage_score,
    normalized_numbers,
)


def plan(task_type="kb", quality_tier="kb_rule_only"):
    return {
        "source_file": "source.jsonl",
        "source_line": 1,
        "source_family": "qwen",
        "version": "20260628_v1",
        "task_id": "task-1",
        "type": task_type,
        "phase": "core_8k",
        "split": "train",
        "input_tokens": 4096,
        "quality_tier": quality_tier,
    }


def record(answer, tool_text, command="read /workspace/documents/doc_001.md"):
    return {
        "messages": [
            {"role": "user", "content": "问题"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"' + command + '"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": tool_text},
            {"role": "assistant", "content": answer},
        ]
    }


def kb_manifest(gold="处理时限为48小时"):
    return {
        "type": "kb",
        "instruction": "处理时限是什么？",
        "gold_answer": gold,
        "citation": {"doc_id": "doc_001"},
        "source_documents": ["doc_001"],
    }


def test_text_and_number_normalization():
    assert coverage_score("处理时限为48小时", "规范规定：处理时限为48小时。") == 1.0
    assert normalized_numbers("1,200 件，95%") == {"1200", "95%"}


def test_kb_grounded_answer_auto_passes():
    result = classify_candidate(
        plan(),
        record("规范规定处理时限为48小时。", "doc_001 规范规定处理时限为48小时。"),
        kb_manifest(),
        False,
    )
    assert result["status"] == "auto_pass"
    assert result["documents_accessed"] == 1


def test_kb_wrong_document_auto_rejects():
    result = classify_candidate(
        plan(),
        record("处理结果会另行通知。", "doc_999 内容无关", "read doc_999"),
        kb_manifest(),
        False,
    )
    assert result["status"] == "auto_reject"
    assert result["reason"] == "expected_document_not_used"


def test_kb_numeric_format_ambiguity_stays_manual():
    result = classify_candidate(
        plan(),
        record(
            "该版本于2026年1月1日生效。",
            "doc_001 显示 effective_date: 2026-01-01，但未说明其他信息。",
        ),
        kb_manifest("生效日期为2026-01-01"),
        False,
    )
    assert result["status"] == "manual_review"
    assert result["reason"] in {"unsupported_numeric_claim", "kb_semantic_scope_uncertain"}


def test_sql_verified_candidate_auto_passes():
    result = classify_candidate(
        plan("dwh", "sql_result_verified"),
        record("结果为10。", "查询结果为10。", "sqlite3 logistics.sqlite select 10"),
        {
            "type": "dwh",
            "instruction": "查询结果",
            "gold_answer": {"answer_type": "numeric", "value": 10},
            "expected_tables": [],
        },
        False,
    )
    assert result["status"] == "auto_pass"
    assert result["reason"] == "sql_result_verified"


def test_non_verified_dwh_stays_manual_when_semantics_need_review():
    result = classify_candidate(
        plan("dwh", "judge_rule_only"),
        record("结果需要进一步分析。", "fact_orders 查询完成", "select * from fact_orders"),
        {
            "type": "dwh",
            "instruction": "分析订单",
            "gold_answer": {"answer_type": "anomaly_report", "text": "需要归因"},
            "expected_tables": ["fact_orders"],
        },
        False,
    )
    assert result["status"] == "manual_review"
