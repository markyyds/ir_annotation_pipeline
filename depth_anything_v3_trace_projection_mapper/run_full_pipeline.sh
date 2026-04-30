#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_da3_cameras.sh" "$@"
"${SCRIPT_DIR}/run_trace_projection.sh"
