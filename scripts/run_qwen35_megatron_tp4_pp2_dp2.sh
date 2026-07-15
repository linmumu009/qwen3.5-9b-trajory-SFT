#!/usr/bin/env bash
# Qwen3.5-9B language-model full SFT on 16 Ascend A3 NPUs.
# Safe defaults run a two-step stress test. Override TRAIN_ITERS and checkpoint
# variables explicitly for the 150-step production run.

set -euo pipefail

MODEL=${MODEL:-/models/Qwen3.5-9B}
DATASET=${DATASET:?DATASET must point to an ms-swift agent JSONL file}
RUN_DIR=${RUN_DIR:?RUN_DIR must point to a new run directory}

TP=${TP:-4}
PP=${PP:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
EXPECTED_DP=${EXPECTED_DP:-2}

TRAIN_ITERS=${TRAIN_ITERS:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-16384}
LR=${LR:-1e-6}
MIN_LR=${MIN_LR:-1e-7}
DEFAULT_LR_WARMUP_ITERS=8
if (( TRAIN_ITERS <= DEFAULT_LR_WARMUP_ITERS )); then
    DEFAULT_LR_WARMUP_ITERS=$((TRAIN_ITERS > 1 ? TRAIN_ITERS - 1 : 0))
fi
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-${DEFAULT_LR_WARMUP_ITERS}}
SAVE_STEPS=${SAVE_STEPS:-1000000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-true}
NO_SAVE_RNG=${NO_SAVE_RNG:-true}
SAVE_SAFETENSORS=${SAVE_SAFETENSORS:-false}
MASTER_PORT=${MASTER_PORT:-29635}
NPU_MONITOR_INTERVAL=${NPU_MONITOR_INTERVAL:-15}

if (( NPROC_PER_NODE % (TP * PP) != 0 )); then
    echo "NPROC_PER_NODE must be divisible by TP * PP" >&2
    exit 2
fi
DP=$((NPROC_PER_NODE / (TP * PP)))
if (( DP != EXPECTED_DP )); then
    echo "parallel topology mismatch: TP=${TP}, PP=${PP}, DP=${DP}" >&2
    exit 2
fi
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP) != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE * DP" >&2
    exit 2
fi
if [[ ! -d "${MODEL}" ]]; then
    echo "missing model directory: ${MODEL}" >&2
    exit 2
fi
if [[ ! -f "${DATASET}" ]]; then
    echo "missing dataset file: ${DATASET}" >&2
    exit 2
fi
if [[ -e "${RUN_DIR}/started_at.txt" ]]; then
    echo "run directory has already been started: ${RUN_DIR}" >&2
    exit 2
fi

mkdir -p "${RUN_DIR}"
exec >> "${RUN_DIR}/train.log" 2>&1

echo $$ > "${RUN_DIR}/launcher.pid"
date --iso-8601=seconds > "${RUN_DIR}/started_at.txt"
sha256sum "${DATASET}" > "${RUN_DIR}/dataset.sha256"

export USE_MCORE_GDN=${USE_MCORE_GDN:-0}
export HCCL_OP_BASE_FFTS_MODE_ENABLE=${HCCL_OP_BASE_FFTS_MODE_ENABLE:-TRUE}
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export NPROC_PER_NODE
export MASTER_PORT

printf '%s\n' \
    "model=${MODEL}" \
    "dataset=${DATASET}" \
    "run_dir=${RUN_DIR}" \
    "tp=${TP}" \
    "pp=${PP}" \
    "dp=${DP}" \
    "nproc_per_node=${NPROC_PER_NODE}" \
    "micro_batch_size=${MICRO_BATCH_SIZE}" \
    "global_batch_size=${GLOBAL_BATCH_SIZE}" \
    "num_microbatches_per_dp_replica=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP)))" \
    "train_iters=${TRAIN_ITERS}" \
    "max_length=${MAX_LENGTH}" \
    "lr=${LR}" \
    "min_lr=${MIN_LR}" \
    "lr_warmup_iters=${LR_WARMUP_ITERS}" \
    "save_steps=${SAVE_STEPS}" \
    "save_total_limit=${SAVE_TOTAL_LIMIT}" \
    "no_save_optim=${NO_SAVE_OPTIM}" \
    "no_save_rng=${NO_SAVE_RNG}" \
    "save_safetensors=${SAVE_SAFETENSORS}" \
    "master_port=${MASTER_PORT}" \
    > "${RUN_DIR}/resolved_config.txt"
cat "${RUN_DIR}/resolved_config.txt"

(
    while kill -0 $$ 2>/dev/null; do
        date --iso-8601=seconds
        npu-smi info
        sleep "${NPU_MONITOR_INTERVAL}"
    done
) >> "${RUN_DIR}/npu_smi.log" 2>&1 &
MONITOR_PID=$!

cleanup() {
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

set +e
set -x
megatron sft \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --tuner_type full \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --bf16 true \
    --tensor_model_parallel_size "${TP}" \
    --pipeline_model_parallel_size "${PP}" \
    --sequence_parallel true \
    --use_distributed_optimizer true \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --loss_scale default+ignore_empty_think \
    --max_length "${MAX_LENGTH}" \
    --truncation_strategy delete \
    --padding_free false \
    --packing false \
    --group_by_length true \
    --attention_backend flash \
    --cross_entropy_loss_fusion true \
    --gradient_accumulation_fusion false \
    --masked_softmax_fusion false \
    --apply_wd_to_qk_layernorm true \
    --optimizer adam \
    --lr "${LR}" \
    --min_lr "${MIN_LR}" \
    --lr_decay_style cosine \
    --lr_warmup_iters "${LR_WARMUP_ITERS}" \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --clip_grad 1.0 \
    --train_iters "${TRAIN_ITERS}" \
    --logging_steps 1 \
    --eval_iters 0 \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --save_safetensors "${SAVE_SAFETENSORS}" \
    --no_save_optim "${NO_SAVE_OPTIM}" \
    --no_save_rng "${NO_SAVE_RNG}" \
    --output_dir "${RUN_DIR}" \
    --add_version false \
    --dataset_num_proc 8 \
    --dataloader_num_workers 4 \
    --seed 42 \
    --data_seed 42
status=$?
set +x
set -e

echo "${status}" > "${RUN_DIR}/exit_code"
date --iso-8601=seconds > "${RUN_DIR}/finished_at.txt"
exit "${status}"
