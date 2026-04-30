#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${SCRIPT_DIR}/visualize_subtask_instructions.py" \
  --test-data "${PROJECT_ROOT}/test_data" \
  --input-dir "${SCRIPT_DIR}/outputs" \
  --output-dir "${SCRIPT_DIR}/visualizations" \
  "$@"
