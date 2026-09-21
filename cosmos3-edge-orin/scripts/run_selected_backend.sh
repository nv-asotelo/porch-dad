#!/usr/bin/env bash
# Start the explicitly selected, already-built engine profile.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${COSMOS_PROFILE:-}" in
  fp16)
    : "${COSMOS_CACHE_DIR:?Selected cache is required}"
    exec bash "$project_dir/scripts/run_backend.sh" "${COSMOS_MODEL_DIR:?Selected model is required}"
    ;;
  rtn-v1)
    export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    rtn_args=(--max-input-len "${COSMOS_MAX_INPUT_LEN:-1024}" \
      --max-kv-capacity "${COSMOS_MAX_KV_CAPACITY:-2048}" --host 127.0.0.1 --port 8000)
    if [[ -n "${COSMOS_MAX_IMAGE_TOKENS:-}" ]]; then
      rtn_args+=(--max-image-tokens "$COSMOS_MAX_IMAGE_TOKENS")
    fi
    if [[ -n "${COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE-${COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE:-}}" ]]; then
      rtn_args+=(--max-image-tokens-per-image "${COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE-$COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE}")
    fi
    rtn_args+=(--encoder-embedding-cache-budget-bytes "${COSMOS_ENCODER_CACHE_BYTES:-0}")
    exec "$project_dir/external/TensorRT-Edge-LLM/.venv/bin/python" \
      "$project_dir/scripts/rtn_backend.py" serve \
      --model "${COSMOS_MODEL_DIR:?Selected model is required}" \
      --cache-dir "${COSMOS_CACHE_DIR:?Selected cache is required}" \
      "${rtn_args[@]}"
    ;;
  *) printf 'COSMOS_PROFILE must identify a validated fp16 or rtn-v1 profile.\n' >&2; exit 2 ;;
esac
