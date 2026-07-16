#!/usr/bin/env python3
"""Validate clean Hugging Face exports produced from Megatron checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_FILES = {
    "chat_template.jinja",
    "config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
}
FORBIDDEN_MARKERS = (".distcp", "optimizer", "rng", "trainer_state")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model(root: Path, step: int, inspect_safetensors: bool) -> dict:
    model_dir = root / f"checkpoint-{step}-hf"
    errors: list[str] = []
    if not model_dir.is_dir():
        return {"step": step, "path": str(model_dir), "ok": False, "errors": ["missing directory"]}

    names = {path.name for path in model_dir.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_FILES - names)
    if missing:
        errors.append(f"missing files: {missing}")

    forbidden = sorted(
        str(path.relative_to(model_dir))
        for path in model_dir.rglob("*")
        if path.is_file() and any(marker in path.name.lower() for marker in FORBIDDEN_MARKERS)
    )
    if forbidden:
        errors.append(f"training artifacts present: {forbidden}")

    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if config.get("model_type") != "qwen3_5":
        errors.append(f"unexpected model_type: {config.get('model_type')!r}")
    architectures = config.get("architectures") or []
    if "Qwen3_5ForConditionalGeneration" not in architectures:
        errors.append(f"unexpected architectures: {architectures!r}")
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype != "bfloat16":
        errors.append(f"unexpected dtype: {dtype!r}")

    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    weight_map = index.get("weight_map") or {}
    shards = sorted(set(weight_map.values()))
    missing_shards = [name for name in shards if not (model_dir / name).is_file()]
    if not weight_map:
        errors.append("empty safetensors weight_map")
    if missing_shards:
        errors.append(f"missing safetensors shards: {missing_shards}")
    weight_bytes = sum((model_dir / name).stat().st_size for name in shards if (model_dir / name).is_file())
    index_total_size = (index.get("metadata") or {}).get("total_size")
    tensor_bytes = int(index_total_size) if index_total_size is not None else None
    file_overhead_bytes = weight_bytes - tensor_bytes if tensor_bytes is not None else None
    if tensor_bytes is not None:
        max_expected_overhead = max(16 * 1024 * 1024, tensor_bytes // 100)
        if file_overhead_bytes < 0 or file_overhead_bytes > max_expected_overhead:
            errors.append(
                "implausible safetensors file overhead: "
                f"tensor_bytes={tensor_bytes}, file_bytes={weight_bytes}, overhead={file_overhead_bytes}"
            )

    tensor_count = None
    if inspect_safetensors and not missing_shards:
        try:
            from safetensors import safe_open

            tensor_keys: set[str] = set()
            for shard in shards:
                with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
                    tensor_keys.update(handle.keys())
            tensor_count = len(tensor_keys)
            missing_tensors = sorted(set(weight_map) - tensor_keys)
            extra_tensors = sorted(tensor_keys - set(weight_map))
            if missing_tensors:
                errors.append(f"index tensors missing from shards: {missing_tensors[:10]}")
            if extra_tensors:
                errors.append(f"unindexed tensors in shards: {extra_tensors[:10]}")
        except ImportError:
            errors.append("safetensors package unavailable for header inspection")

    return {
        "step": step,
        "path": str(model_dir),
        "ok": not errors,
        "model_type": config.get("model_type"),
        "architectures": architectures,
        "dtype": dtype,
        "weight_shards": len(shards),
        "weight_bytes": weight_bytes,
        "tensor_bytes": tensor_bytes,
        "file_overhead_bytes": file_overhead_bytes,
        "indexed_tensors": len(weight_map),
        "inspected_tensors": tensor_count,
        "config_sha256": sha256(config_path) if config_path.is_file() else None,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", default="15,30,45,60,75,90,105,120,135,150")
    parser.add_argument("--inspect-safetensors", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    results = [validate_model(args.root.resolve(), step, args.inspect_safetensors) for step in steps]
    payload = {
        "root": str(args.root.resolve()),
        "expected_steps": steps,
        "valid_models": sum(result["ok"] for result in results),
        "total_models": len(results),
        "ok": all(result["ok"] for result in results),
        "models": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
