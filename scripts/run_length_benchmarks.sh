#!/usr/bin/env bash
set -uo pipefail

BENCHMARK_DIR=${1:?usage: run_length_benchmarks.sh BENCHMARK_DIR OUTPUT_DIR [TARGETS]}
OUTPUT_DIR=${2:?usage: run_length_benchmarks.sh BENCHMARK_DIR OUTPUT_DIR [TARGETS]}
TARGETS=${3:-"4096 8192 16384"}
MODEL=${MODEL:-/models/Qwen3.5-9B}
NPU_DEVICES=${NPU_DEVICES:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
SEQUENCE_PARALLEL_SIZE=${SEQUENCE_PARALLEL_SIZE:-1}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
DEEPSPEED=${DEEPSPEED:-}
FSDP=${FSDP:-}
STEP_TIMEOUT=${STEP_TIMEOUT:-1800}

if (( SEQUENCE_PARALLEL_SIZE > NPROC_PER_NODE )); then
    echo "SEQUENCE_PARALLEL_SIZE cannot exceed NPROC_PER_NODE" >&2
    exit 2
fi

extra_args=()
if (( SEQUENCE_PARALLEL_SIZE > 1 )); then
    extra_args+=(--sequence_parallel_size "${SEQUENCE_PARALLEL_SIZE}")
fi
if [[ -n "${DEEPSPEED}" ]]; then
    extra_args+=(--deepspeed "${DEEPSPEED}")
fi
if [[ -n "${FSDP}" ]]; then
    extra_args+=(--fsdp "${FSDP}")
fi

mkdir -p "${OUTPUT_DIR}"
summary_file="${OUTPUT_DIR}/benchmark_status.tsv"
printf 'target\tactual_tokens\tmax_length\tnproc_per_node\tsequence_parallel_size\tlora_rank\tdeepspeed\tfsdp\texit_code\tlog\n' > "${summary_file}"

for target in ${TARGETS}; do
    sample=$(find "${BENCHMARK_DIR}" -maxdepth 1 -type f -name "target_${target}_actual_*.jsonl" | head -n 1)
    if [[ -z "${sample}" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${target}" NA NA "${NPROC_PER_NODE}" "${SEQUENCE_PARALLEL_SIZE}" \
            "${LORA_RANK}" "${DEEPSPEED:-none}" "${FSDP:-none}" 2 missing \
            >> "${summary_file}"
        continue
    fi
    stem=$(basename "${sample}" .jsonl)
    actual=${stem##*_actual_}
    max_length=$(( (actual + 127) / 128 * 128 ))
    run_dir="${OUTPUT_DIR}/${stem}"
    log_file="${OUTPUT_DIR}/${stem}.log"
    mkdir -p "${run_dir}"
    training_sample="${sample}"
    if (( NPROC_PER_NODE > 1 )); then
        training_sample="${run_dir}/${stem}_replicated_${NPROC_PER_NODE}.jsonl"
        : > "${training_sample}"
        for ((rank = 0; rank < NPROC_PER_NODE; rank++)); do
            while IFS= read -r line; do
                printf '%s\n' "${line}" >> "${training_sample}"
            done < "${sample}"
        done
    fi

    set +e
    ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    timeout "${STEP_TIMEOUT}" \
    swift sft \
        --model "${MODEL}" \
        --dataset "${training_sample}" \
        --tuner_type lora \
        --target_modules all-linear \
        --lora_rank "${LORA_RANK}" \
        --lora_alpha "${LORA_ALPHA}" \
        --torch_dtype bfloat16 \
        --max_length "${max_length}" \
        --truncation_strategy delete \
        --loss_scale default+ignore_empty_think \
        --split_dataset_ratio 0 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --gradient_checkpointing true \
        --learning_rate 1e-4 \
        --warmup_steps 0 \
        --max_steps 1 \
        --logging_steps 1 \
        --save_strategy no \
        --report_to none \
        --dataset_num_proc 1 \
        --dataloader_num_workers 0 \
        --output_dir "${run_dir}" \
        --add_version false \
        "${extra_args[@]}" \
        > "${log_file}" 2>&1
    exit_code=$?
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${target}" "${actual}" "${max_length}" "${NPROC_PER_NODE}" \
        "${SEQUENCE_PARALLEL_SIZE}" "${LORA_RANK}" "${DEEPSPEED:-none}" \
        "${FSDP:-none}" "${exit_code}" "${log_file}" \
        >> "${summary_file}"
done

cat "${summary_file}"
