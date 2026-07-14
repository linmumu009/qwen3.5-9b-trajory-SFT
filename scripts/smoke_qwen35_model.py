"""Run a short Qwen3.5 forward or LoRA train-step smoke test on one NPU."""

from __future__ import annotations

import argparse
import math

import torch
import torch_npu  # noqa: F401
import swift.model  # noqa: F401  # Apply ms-swift's Qwen3.5 NPU patch first.
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/Qwen3.5-9B")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--train-step", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_module = modeling_qwen3_5.chunk_gated_delta_rule.__module__
    if patch_module != "swift.model.chunk_gated_delta_rule":
        raise RuntimeError(f"Qwen3.5 NPU patch is inactive: {patch_module}")
    print(f"QWEN35_PATCH_OK module={patch_module}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = tokenizer(
        "请用一句话说明轨迹监督微调的目的。",
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    encoded = {key: value.to(args.device) for key, value in encoded.items()}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    torch.npu.reset_peak_memory_stats(args.device)

    if args.train_step:
        model = get_peft_model(
            model,
            LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
        )
        model.train()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**encoded, labels=encoded["input_ids"])
        loss = outputs.loss
        if not math.isfinite(float(loss.detach().cpu())):
            raise RuntimeError(f"Non-finite training loss: {loss}")
        loss.backward()
        optimizer.step()
        torch.npu.synchronize()
        print(
            "TRAIN_STEP_OK",
            f"loss={float(loss.detach().cpu()):.6f}",
            f"trainable_parameters={sum(p.numel() for p in trainable)}",
        )
    else:
        model.eval()
        with torch.no_grad():
            outputs = model(**encoded)
        torch.npu.synchronize()
        logits = outputs.logits
        if not torch.isfinite(logits).all():
            raise RuntimeError("Forward logits contain non-finite values")
        print("FORWARD_OK", f"logits_shape={tuple(logits.shape)}")

    peak_gib = torch.npu.max_memory_allocated(args.device) / 1024**3
    print(f"NPU_PEAK_MEMORY_GIB={peak_gib:.3f}")


if __name__ == "__main__":
    main()
