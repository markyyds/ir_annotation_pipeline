#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

TEST_DATA="${TEST_DATA:-${PROJECT_ROOT}/test_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"
VLM_MODELS="${VLM_MODELS:-qwen3-max}"

for VLM_MODEL in ${VLM_MODELS}; do
  SAFE_MODEL="${VLM_MODEL//\//_}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/pipeline.py" \
    --test-data "${TEST_DATA}" \
    --vlm-model "${VLM_MODEL}" \
    --output-dir "${OUTPUT_ROOT}/${SAFE_MODEL}" \
    "$@"
done
