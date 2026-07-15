"""Materialize an authoritative candidate catalog into one ms-swift JSONL file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from catalog_trajectory_candidates import canonical_tools
from prepare_trajectory_sft import convert_messages, iter_jsonl, make_training_row


def load_catalog(path: Path) -> tuple[list[str], dict[str, dict[int, dict[str, Any]]]]:
    source_order: list[str] = []
    locations: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for _, row in iter_jsonl(path):
        source_file = row.get("source_file")
        source_line = row.get("source_line")
        if not isinstance(source_file, str) or not isinstance(source_line, int):
            raise ValueError("catalog rows require string source_file and integer source_line")
        if source_file not in locations:
            source_order.append(source_file)
        if source_line in locations[source_file]:
            raise ValueError(f"duplicate catalog location: {source_file}:{source_line}")
        locations[source_file][source_line] = row
    return source_order, locations


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(catalog: Path, data_dir: Path, output: Path) -> dict[str, Any]:
    source_order, locations = load_catalog(catalog)
    expected = sum(len(lines) for lines in locations.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    written = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for source_file in source_order:
                source_path = data_dir / source_file
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                selected = locations[source_file]
                found: set[int] = set()
                for line_number, record in iter_jsonl(source_path):
                    if line_number not in selected:
                        continue
                    found.add(line_number)
                    converted, conversion = convert_messages(record.get("messages") or [])
                    if not converted or converted[-1].get("role") != "assistant":
                        raise RuntimeError(f"candidate lost final answer: {source_file}:{line_number}")
                    if not conversion.get("tool_calls"):
                        raise RuntimeError(f"candidate lost tool roundtrip: {source_file}:{line_number}")
                    tools_ok, reason = canonical_tools(converted)
                    if not tools_ok:
                        raise RuntimeError(
                            f"candidate tool schema drift: {source_file}:{line_number}: {reason}"
                        )
                    handle.write(
                        json.dumps(
                            make_training_row(converted),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    written += 1
                missing = set(selected) - found
                if missing:
                    raise RuntimeError(
                        f"catalog locations missing in {source_file}: {sorted(missing)[:10]}"
                    )
        if written != expected:
            raise RuntimeError(f"materialized row mismatch: expected={expected}, written={written}")
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "catalog": str(catalog),
        "output": str(output),
        "records": written,
        "source_files": len(source_order),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "format": "ms-swift agent JSONL",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            materialize(args.catalog, args.data_dir, args.output),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
