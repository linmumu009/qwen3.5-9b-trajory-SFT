"""Verify the pinned Qwen3.5 Atlas A3 training environment."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os

import torch
import torch_npu  # noqa: F401


EXPECTED_VERSIONS = {
    "ms_swift": "4.3.2",
    "numpy": "1.26.4",
    "torch": "2.9.0",
    "torch_npu": "2.9.0.post2",
    "transformers": "5.2.0",
    "flash-linear-attention": "0.4.2",
    "fla-core": "0.4.2",
    "mindspeed": "0.16.0",
    "megatron-core": "0.16.0",
    "triton_ascend": "3.2.1",
}

REQUIRED_MODULES = (
    "mindspeed.ops.triton.chunk_delta_h",
    "mindspeed.ops.triton.chunk_o",
    "mindspeed.ops.triton.chunk_scaled_dot_kkt",
    "mindspeed.ops.triton.wy_fast",
    "swift.model.chunk_gated_delta_rule",
)


def main() -> None:
    for package, expected in EXPECTED_VERSIONS.items():
        actual = metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package}: expected {expected}, got {actual}")
        print(f"VERSION_OK {package}={actual}")

    for module in REQUIRED_MODULES:
        imported = importlib.import_module(module)
        print(f"IMPORT_OK {module} -> {imported.__file__}")

    soc_version = os.environ.get("SOC_VERSION")
    if soc_version != "ascend910_9391":
        raise RuntimeError(f"Unexpected SOC_VERSION: {soc_version!r}")

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    if torch.npu.device_count() != 16:
        raise RuntimeError(f"Expected 16 NPUs, got {torch.npu.device_count()}")

    tensor = torch.arange(4, device="npu:0")
    values = tensor.cpu().tolist()
    if values != [0, 1, 2, 3]:
        raise RuntimeError(f"Unexpected NPU tensor result: {values}")
    print(f"NPU_OK device_count={torch.npu.device_count()} tensor={values}")


if __name__ == "__main__":
    main()
