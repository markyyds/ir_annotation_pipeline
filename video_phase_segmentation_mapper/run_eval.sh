#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${SCRIPT_DIR}/evaluate_phase_segmentation.py" \
  --test-data "${PROJECT_ROOT}/test_data" \
  --output-dir "${SCRIPT_DIR}/eval_outputs" \
  "$@"
