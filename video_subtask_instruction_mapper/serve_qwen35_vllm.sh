#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"

CMD=(
  vllm
  serve
  "${MODEL_ID}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --reasoning-parser qwen3
  --trust-remote-code
)

if [[ -n "${LIMIT_MM_PER_PROMPT}" ]]; then
  CMD+=(--limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}")
fi

"${CMD[@]}"
