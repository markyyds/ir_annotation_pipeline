#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${SCRIPT_DIR}/project_trace_and_visualize.py" \
  --test-data "${PROJECT_ROOT}/test_data" \
  --camera-dir "${SCRIPT_DIR}/da3_camera_outputs" \
  --output-dir "${SCRIPT_DIR}/trace_projection_outputs" \
  "$@"
