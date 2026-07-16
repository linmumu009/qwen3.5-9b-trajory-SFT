#!/usr/bin/env bash
# Start an OpenAI-compatible API for one exported Qwen3.5-9B checkpoint.

set -euo pipefail

MODEL_DIR=${MODEL_DIR:?MODEL_DIR must point to checkpoint-*-hf}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.5-9b-trajory-sft}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
MAX_LENGTH=${MAX_LENGTH:-16384}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
INFER_BACKEND=${INFER_BACKEND:-transformers}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

for required in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
    if [[ ! -f "${MODEL_DIR}/${required}" ]]; then
        echo "missing ${required} in ${MODEL_DIR}" >&2
        exit 2
    fi
done

export ASCEND_RT_VISIBLE_DEVICES
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

exec swift deploy \
    --model "${MODEL_DIR}" \
    --load_args false \
    --infer_backend "${INFER_BACKEND}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served_model_name "${SERVED_MODEL_NAME}" \
    --max_length "${MAX_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
