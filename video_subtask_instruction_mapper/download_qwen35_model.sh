#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
LOCAL_DIR="${LOCAL_DIR:-/mnt/workspace/models/Qwen3.5-35B-A3B}"
SOURCE="${SOURCE:-hf}"

mkdir -p "${LOCAL_DIR}"

case "${SOURCE}" in
  hf)
    # Prefer the HF CLI download path over implicit vLLM worker downloads.
    # It can resume partial files and keeps all workers reading from local disk.
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
    huggingface-cli download "${MODEL_ID}" \
      --local-dir "${LOCAL_DIR}" \
      --local-dir-use-symlinks False \
      --resume-download
    ;;
  hf_git)
    # Git LFS path. Useful when huggingface-cli has trouble with signed CAS URLs.
    if ! command -v git-lfs >/dev/null 2>&1 && ! command -v git-lfs >/dev/null 2>&1; then
      echo "git-lfs is required for SOURCE=hf_git" >&2
      exit 1
    fi
    if [ ! -d "${LOCAL_DIR}/.git" ]; then
      GIT_LFS_SKIP_SMUDGE=1 git clone "https://huggingface.co/${MODEL_ID}" "${LOCAL_DIR}"
    fi
    git -C "${LOCAL_DIR}" lfs pull
    ;;
  modelscope)
    # Usually faster from CN networks if the model mirror exists.
    python - <<'PY'
import os
from modelscope import snapshot_download

snapshot_download(
    os.environ.get("MODEL_ID", "Qwen/Qwen3.5-35B-A3B"),
    local_dir=os.environ.get("LOCAL_DIR", "/mnt/workspace/models/Qwen3.5-35B-A3B"),
)
PY
    ;;
  *)
    echo "Unknown SOURCE=${SOURCE}. Use hf, hf_git, or modelscope." >&2
    exit 1
    ;;
esac

echo "Model downloaded to: ${LOCAL_DIR}"
find "${LOCAL_DIR}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'config.json' -o -name 'tokenizer*' \) | sort
