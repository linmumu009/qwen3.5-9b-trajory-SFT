"""Summarize trajectory verdict JSONL files without printing task content."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Verdict JSONL file or directory")
    args = parser.parse_args()

    files = [args.input] if args.input.is_file() else sorted(args.input.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No verdict JSONL files found: {args.input}")

    result: dict[str, object] = {"files": {}, "total_records": 0}
    total_verdicts: Counter[str] = Counter()
    total_versions: Counter[str] = Counter()
    for path in files:
        verdicts: Counter[str] = Counter()
        versions: Counter[str] = Counter()
        keys: Counter[str] = Counter()
        invalid = 0
        records = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records += 1
                try:
                    row = json.loads(line)
                except Exception:
                    invalid += 1
                    continue
                if not isinstance(row, dict):
                    invalid += 1
                    continue
                keys.update(row.keys())
                verdicts[str(row.get("verdict", "<missing>"))] += 1
                versions[str(row.get("v", "<missing>"))] += 1
        result["files"][path.name] = {
            "records": records,
            "invalid": invalid,
            "verdicts": dict(sorted(verdicts.items())),
            "versions": dict(sorted(versions.items())),
            "keys": dict(sorted(keys.items())),
        }
        result["total_records"] += records
        total_verdicts.update(verdicts)
        total_versions.update(versions)
    result["total_verdicts"] = dict(sorted(total_verdicts.items()))
    result["total_versions"] = dict(sorted(total_versions.items()))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
