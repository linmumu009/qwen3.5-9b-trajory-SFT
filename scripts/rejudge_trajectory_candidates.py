"""Conservatively rejudge selected trajectory candidates using recorded evidence.

The script never emits prompts, answers, tool outputs, or gold text.  It joins the
metadata-only dataset plan to the raw records and versioned manifests, then writes
only decision metadata and aggregate summaries.  Ambiguous semantic cases remain
``manual_review`` instead of being forced into positive SFT.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from prepare_trajectory_sft import iter_jsonl


# Permit numbers immediately after Chinese text (for example ``时限为48小时``),
# while still ignoring identifier suffixes such as ``doc_038``.
NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")
STRUCTURAL_PREFIX_RE = re.compile(r"(?m)^\s*(?:[-*]>#]+|\d+[.)、])\s*")
CLAIM_SPLIT_RE = re.compile(r"[。！？；\n]+")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def normalized_numbers(value: Any) -> set[str]:
    text = STRUCTURAL_PREFIX_RE.sub("", unicodedata.normalize("NFKC", str(value or "")))
    result = set()
    for item in NUMBER_RE.findall(text):
        item = item.replace(",", "").lstrip("+")
        if item.endswith("%"):
            number, suffix = item[:-1], "%"
        else:
            number, suffix = item, ""
        try:
            normalized = format(float(number), ".12g")
        except ValueError:
            normalized = number
        result.add(normalized + suffix)
    return result


def ngrams(text: str, size: int = 2) -> set[str]:
    if not text:
        return set()
    if len(text) < size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def coverage_score(claim: Any, evidence: Any) -> float:
    claim_norm = normalize_text(claim)
    evidence_norm = normalize_text(evidence)
    if not claim_norm or not evidence_norm:
        return 0.0
    if claim_norm in evidence_norm:
        return 1.0
    claim_grams = ngrams(claim_norm)
    evidence_grams = ngrams(evidence_norm)
    return round(len(claim_grams & evidence_grams) / max(1, len(claim_grams)), 6)


def claim_support_score(answer: str, evidence: str) -> float:
    claims = []
    for raw_claim in CLAIM_SPLIT_RE.split(STRUCTURAL_PREFIX_RE.sub("", answer)):
        normalized = normalize_text(raw_claim)
        if len(normalized) >= 8:
            claims.append((raw_claim, len(normalized)))
    if not claims:
        return coverage_score(answer, evidence)
    weighted = sum(coverage_score(claim, evidence) * size for claim, size in claims)
    return round(weighted / sum(size for _, size in claims), 6)


def final_assistant_content(record: dict[str, Any]) -> str:
    for message in reversed(record.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def recorded_tool_evidence(record: dict[str, Any]) -> tuple[str, str]:
    responses: list[str] = []
    calls: list[str] = []
    for message in record.get("messages") or []:
        if message.get("role") == "tool":
            responses.append(str(message.get("content") or ""))
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            calls.append(str(function.get("name") or ""))
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                calls.append(arguments)
            elif arguments is not None:
                calls.append(json.dumps(arguments, ensure_ascii=False, sort_keys=True))
    return "\n".join(responses), "\n".join(calls)


def expected_doc_ids(manifest: dict[str, Any]) -> set[str]:
    result = set()
    citation = manifest.get("citation")
    if isinstance(citation, dict) and citation.get("doc_id"):
        result.add(str(citation["doc_id"]))
    for item in manifest.get("source_documents") or []:
        if item:
            result.add(str(item))
    return result


def expected_tables(manifest: dict[str, Any]) -> set[str]:
    result = {str(item).lower() for item in manifest.get("expected_tables") or [] if item}
    criteria = manifest.get("verification_criteria")
    if isinstance(criteria, dict):
        result.update(str(item).lower() for item in criteria.get("must_query_tables") or [] if item)
    return result


def gold_text(manifest: dict[str, Any]) -> str:
    gold = manifest.get("gold_answer")
    if isinstance(gold, dict):
        parts = [gold.get("text") or ""]
        value = gold.get("value")
        if value is not None:
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return "\n".join(parts)
    return str(gold or "")


def manifest_fingerprint(row: dict[str, Any]) -> str:
    selected = {
        "type": row.get("type"),
        "instruction": row.get("instruction"),
        "gold_answer": row.get("gold_answer"),
        "expected_tables": row.get("expected_tables"),
        "citation": row.get("citation"),
        "source_documents": row.get("source_documents"),
    }
    return json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_manifests(directory: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], set[tuple[str, str]]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    fingerprints: dict[tuple[str, str], str] = {}
    conflicts: set[tuple[str, str]] = set()
    for path in sorted(directory.glob("*.jsonl")):
        for _, row in iter_jsonl(path):
            version = row.get("v")
            task_id = row.get("task_id")
            if not isinstance(version, str) or not isinstance(task_id, str):
                continue
            key = (version, task_id)
            fingerprint = manifest_fingerprint(row)
            if key in fingerprints and fingerprints[key] != fingerprint:
                conflicts.add(key)
                continue
            rows[key] = row
            fingerprints[key] = fingerprint
    return rows, conflicts


def classify_candidate(
    plan: dict[str, Any], record: dict[str, Any], manifest: dict[str, Any] | None, conflict: bool
) -> dict[str, Any]:
    base = {
        "source_file": plan["source_file"],
        "source_line": int(plan["source_line"]),
        "source_family": plan["source_family"],
        "version": plan["version"],
        "task_id": plan["task_id"],
        "type": plan["type"],
        "phase": plan["phase"],
        "split": plan["split"],
        "input_tokens": int(plan["input_tokens"]),
        "quality_tier": plan["quality_tier"],
    }
    if conflict:
        return {**base, "status": "manual_review", "reason": "manifest_conflict", "confidence": "high"}
    if manifest is None:
        return {**base, "status": "manual_review", "reason": "manifest_missing", "confidence": "high"}

    answer = final_assistant_content(record)
    tool_responses, tool_calls = recorded_tool_evidence(record)
    evidence = f"{tool_responses}\n{tool_calls}"
    gold = gold_text(manifest)
    gold_numbers = normalized_numbers(gold)
    answer_numbers = normalized_numbers(answer)
    evidence_numbers = normalized_numbers(evidence + "\n" + str(manifest.get("instruction") or ""))
    unsupported_numbers = sorted(answer_numbers - evidence_numbers)
    docs = expected_doc_ids(manifest)
    docs_accessed = sorted(doc for doc in docs if normalize_text(doc) in normalize_text(evidence))
    tables = expected_tables(manifest)
    call_text = tool_calls.lower()
    tables_accessed = sorted(table for table in tables if table in call_text)
    support = claim_support_score(answer, tool_responses)
    gold_overlap = coverage_score(gold, answer)
    gold_numbers_match = not gold_numbers or gold_numbers.issubset(answer_numbers)

    metrics = {
        "final_answer_chars": len(answer),
        "recorded_support_score": support,
        "gold_overlap_score": gold_overlap,
        "gold_numbers": len(gold_numbers),
        "gold_numbers_match": gold_numbers_match,
        "unsupported_answer_numbers": len(unsupported_numbers),
        "expected_documents": len(docs),
        "documents_accessed": len(docs_accessed),
        "expected_tables": len(tables),
        "tables_accessed": len(tables_accessed),
    }

    if plan.get("quality_tier") == "sql_result_verified":
        return {
            **base,
            **metrics,
            "status": "auto_pass",
            "reason": "sql_result_verified",
            "confidence": "high",
        }
    if not answer:
        return {**base, **metrics, "status": "auto_reject", "reason": "empty_final_answer", "confidence": "high"}

    task_type = str(plan.get("type") or manifest.get("type") or "")
    if task_type == "kb":
        doc_coverage = len(docs_accessed) / max(1, len(docs))
        if docs and doc_coverage == 0 and support < 0.5:
            return {
                **base,
                **metrics,
                "status": "auto_reject",
                "reason": "expected_document_not_used",
                "confidence": "high",
            }
        if unsupported_numbers and support < 0.7:
            return {
                **base,
                **metrics,
                "status": "manual_review",
                "reason": "unsupported_numeric_claim",
                "confidence": "medium",
            }
        if doc_coverage == 1 and support >= 0.9 and (
            gold_numbers_match or not gold_numbers
        ):
            return {
                **base,
                **metrics,
                "status": "auto_pass",
                "reason": "kb_grounded_in_expected_document",
                "confidence": "high",
            }
        if doc_coverage == 1 and support >= 0.8 and gold_overlap >= 0.65 and gold_numbers_match:
            return {
                **base,
                **metrics,
                "status": "auto_pass",
                "reason": "kb_gold_and_document_supported",
                "confidence": "medium",
            }
        return {
            **base,
            **metrics,
            "status": "manual_review",
            "reason": "kb_semantic_scope_uncertain",
            "confidence": "medium",
        }

    table_coverage = len(tables_accessed) / max(1, len(tables))
    if tables and table_coverage < 1:
        return {
            **base,
            **metrics,
            "status": "auto_reject",
            "reason": "expected_table_missing",
            "confidence": "high",
        }
    if unsupported_numbers and support < 0.5:
        return {
            **base,
            **metrics,
            "status": "manual_review",
            "reason": "unsupported_numeric_claim",
            "confidence": "medium",
        }
    answer_type = ""
    if isinstance(manifest.get("gold_answer"), dict):
        answer_type = str(manifest["gold_answer"].get("answer_type") or "")
    if (
        answer_type in {"numeric", "table"}
        and table_coverage == 1
        and support >= 0.85
        and gold_numbers_match
        and not unsupported_numbers
    ):
        return {
            **base,
            **metrics,
            "status": "auto_pass",
            "reason": "recorded_query_evidence_matches_gold",
            "confidence": "medium",
        }
    return {
        **base,
        **metrics,
        "status": "manual_review",
        "reason": "execution_or_semantic_review_required",
        "confidence": "medium",
    }


def write_summary(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "records": len(rows),
        "unique_task_ids": len({row["task_id"] for row in rows}),
        "status": dict(sorted(Counter(row["status"] for row in rows).items())),
        "reason": dict(sorted(Counter(row["reason"] for row in rows).items())),
        "by_status": {},
        "by_type": {},
        "by_split": {},
        "by_source_family": {},
    }
    status_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        status_groups[row["status"]].append(row)
    summary["by_status"] = {
        status: {
            "records": len(group),
            "unique_task_ids": len({row["task_id"] for row in group}),
            "input_tokens": sum(int(row["input_tokens"]) for row in group),
            "by_split": dict(sorted(Counter(row["split"] for row in group).items())),
            "by_type": dict(sorted(Counter(row["type"] for row in group).items())),
        }
        for status, group in sorted(status_groups.items())
    }
    for output_name, field in (
        ("by_type", "type"),
        ("by_split", "split"),
        ("by_source_family", "source_family"),
    ):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        summary[output_name] = {
            name: {
                "records": len(group),
                "unique_task_ids": len({row["task_id"] for row in group}),
                "status": dict(sorted(Counter(row["status"] for row in group).items())),
            }
            for name, group in sorted(groups.items())
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--phase", default="core_8k")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_rows = [row for _, row in iter_jsonl(args.plan) if row.get("phase") == args.phase]
    plan_by_source: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in plan_rows:
        line = int(row["source_line"])
        if line in plan_by_source[row["source_file"]]:
            raise RuntimeError(f"duplicate plan location: {row['source_file']}:{line}")
        plan_by_source[row["source_file"]][line] = row

    manifests, manifest_conflicts = load_manifests(args.manifests_dir)
    results: list[tuple[int, dict[str, Any]]] = []
    plan_order = {
        (row["source_file"], int(row["source_line"])): index for index, row in enumerate(plan_rows)
    }
    for source_file, selected_lines in sorted(plan_by_source.items()):
        path = args.data_dir / source_file
        if not path.is_file():
            raise FileNotFoundError(path)
        found: set[int] = set()
        for line_number, record in iter_jsonl(path):
            plan = selected_lines.get(line_number)
            if plan is None:
                continue
            found.add(line_number)
            key = (plan["version"], plan["task_id"])
            result = classify_candidate(plan, record, manifests.get(key), key in manifest_conflicts)
            results.append((plan_order[(source_file, line_number)], result))
        missing = set(selected_lines) - found
        if missing:
            raise RuntimeError(f"missing planned source lines in {source_file}: {sorted(missing)[:10]}")

    results.sort(key=lambda item: item[0])
    rows = [row for _, row in results]
    if len(rows) != len(plan_rows):
        raise RuntimeError(f"rejudge row mismatch: plan={len(plan_rows)} output={len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = write_summary(rows, args.summary_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
