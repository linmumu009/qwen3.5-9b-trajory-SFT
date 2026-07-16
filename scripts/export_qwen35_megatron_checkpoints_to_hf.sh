#!/usr/bin/env bash
# Export Qwen3.5 Megatron torch_dist checkpoints to clean Hugging Face BF16
# safetensors directories. The source checkpoints are never modified.

set -euo pipefail

RUN_DIR=${RUN_DIR:?RUN_DIR must contain the completed Megatron training run}
EXPORT_ROOT=${EXPORT_ROOT:?EXPORT_ROOT must be a dedicated output directory}
STEPS=${STEPS:-"15 30 45 60 75 90 105 120 135 150"}

TP=${TP:-4}
PP=${PP:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29715}
TEST_CONVERT_PRECISION=${TEST_CONVERT_PRECISION:-false}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "missing run directory: ${RUN_DIR}" >&2
    exit 2
fi
if [[ ! -f "${RUN_DIR}/exit_code" ]] || [[ "$(tr -d '[:space:]' < "${RUN_DIR}/exit_code")" != "0" ]]; then
    echo "training run is not marked successful: ${RUN_DIR}" >&2
    exit 2
fi
if (( NPROC_PER_NODE != TP * PP )); then
    echo "export requires exactly one replica: NPROC_PER_NODE must equal TP * PP" >&2
    exit 2
fi
if [[ "${EXPORT_ROOT}" == "${RUN_DIR}" || "${EXPORT_ROOT}" == "${RUN_DIR}/"* ]]; then
    echo "EXPORT_ROOT must be outside RUN_DIR so checkpoints stay untouched" >&2
    exit 2
fi

mkdir -p "${EXPORT_ROOT}/logs"
STATE_FILE="${EXPORT_ROOT}/export_state.tsv"
MANIFEST_FILE="${EXPORT_ROOT}/manifest.tsv"
if [[ ! -f "${STATE_FILE}" ]]; then
    printf 'step\tstatus\tstarted_at\tfinished_at\tsource\toutput\n' > "${STATE_FILE}"
fi
if [[ ! -f "${MANIFEST_FILE}" ]]; then
    printf 'step\toutput\tweight_shards\tweight_bytes\tconfig_sha256\n' > "${MANIFEST_FILE}"
fi

validate_hf_dir() {
    local model_dir=$1
    [[ -f "${model_dir}/config.json" ]] || return 1
    [[ -f "${model_dir}/model.safetensors.index.json" ]] || return 1
    [[ -f "${model_dir}/tokenizer_config.json" ]] || return 1
    [[ -f "${model_dir}/tokenizer.json" ]] || return 1
    compgen -G "${model_dir}/model-*.safetensors" >/dev/null || return 1
    if find "${model_dir}" -type f \( -name '*.distcp' -o -iname '*optimizer*' -o -iname '*rng*' \) -print -quit | grep -q .; then
        return 1
    fi
}

export USE_MCORE_GDN=${USE_MCORE_GDN:-0}
export HCCL_OP_BASE_FFTS_MODE_ENABLE=${HCCL_OP_BASE_FFTS_MODE_ENABLE:-TRUE}
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export ASCEND_RT_VISIBLE_DEVICES
export NPROC_PER_NODE
export MASTER_PORT

for step in ${STEPS}; do
    source_dir="${RUN_DIR}/checkpoint-${step}"
    output_dir="${EXPORT_ROOT}/checkpoint-${step}-hf"
    log_file="${EXPORT_ROOT}/logs/checkpoint-${step}.log"
    started_at=$(date --iso-8601=seconds)

    if [[ ! -f "${source_dir}/latest_checkpointed_iteration.txt" ]]; then
        echo "checkpoint-${step}: missing latest_checkpointed_iteration.txt" >&2
        exit 3
    fi
    recorded_step=$(tr -d '[:space:]' < "${source_dir}/latest_checkpointed_iteration.txt")
    if [[ "${recorded_step}" != "${step}" ]]; then
        echo "checkpoint-${step}: iteration marker is ${recorded_step}" >&2
        exit 3
    fi

    if [[ -e "${output_dir}" ]]; then
        if validate_hf_dir "${output_dir}"; then
            echo "checkpoint-${step}: existing validated export, skipping"
            continue
        fi
        echo "checkpoint-${step}: refusing to overwrite incomplete output ${output_dir}" >&2
        exit 4
    fi

    temp_dir="${EXPORT_ROOT}/.checkpoint-${step}-hf.exporting-${BASHPID}"
    if [[ -e "${temp_dir}" ]]; then
        echo "checkpoint-${step}: temporary path already exists: ${temp_dir}" >&2
        exit 4
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${step}" started "${started_at}" - "${source_dir}" "${output_dir}" >> "${STATE_FILE}"
    echo "checkpoint-${step}: exporting to ${output_dir}"

    set +e
    megatron export \
        --mcore_model "${source_dir}" \
        --output_dir "${temp_dir}" \
        --to_hf true \
        --torch_dtype bfloat16 \
        --tensor_model_parallel_size "${TP}" \
        --pipeline_model_parallel_size "${PP}" \
        --test_convert_precision "${TEST_CONVERT_PRECISION}" \
        > "${log_file}" 2>&1
    status=$?
    set -e

    finished_at=$(date --iso-8601=seconds)
    if (( status != 0 )); then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${step}" "failed:${status}" "${started_at}" "${finished_at}" "${source_dir}" "${temp_dir}" >> "${STATE_FILE}"
        echo "checkpoint-${step}: export failed with status ${status}; see ${log_file}" >&2
        exit "${status}"
    fi
    if ! validate_hf_dir "${temp_dir}"; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${step}" validation_failed "${started_at}" "${finished_at}" "${source_dir}" "${temp_dir}" >> "${STATE_FILE}"
        echo "checkpoint-${step}: output validation failed; preserving ${temp_dir}" >&2
        exit 5
    fi

    mv "${temp_dir}" "${output_dir}"
    shard_count=$(find "${output_dir}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l)
    weight_bytes=$(find "${output_dir}" -maxdepth 1 -type f -name 'model-*.safetensors' -printf '%s\n' | awk '{s += $1} END {printf "%.0f", s}')
    config_sha256=$(sha256sum "${output_dir}/config.json" | awk '{print $1}')
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${step}" "${output_dir}" "${shard_count}" "${weight_bytes}" "${config_sha256}" >> "${MANIFEST_FILE}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${step}" completed "${started_at}" "${finished_at}" "${source_dir}" "${output_dir}" >> "${STATE_FILE}"
    echo "checkpoint-${step}: completed (${shard_count} shards, ${weight_bytes} bytes)"
done

echo "all requested checkpoints exported successfully"
