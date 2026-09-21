#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: scripts/run_backend.sh [--preflight-only] /absolute/path/to/reasoner-checkpoint

Starts the pinned TensorRT Edge-LLM 0.10.1 server on a Jetson Orin.
Build engines first with scripts/build_model_cache.py; startup requires a ready,
normalized cache bundle matching the requested input/KV profile.
The model directory must contain only the reasoner component configs.
--preflight-only permits known repairable sidecars for the build helper to normalize.

Environment overrides:
  EDGE_LLM_DIR             Pinned upstream checkout (default external/TensorRT-Edge-LLM)
  EDGE_LLM_PYTHON          Its Python executable (default checkout/.venv/bin/python)
  COSMOS_CACHE_DIR         Compiled-engine cache (default data/engine-cache-chw)
  COSMOS_BACKEND_HOST      Bind address (default 127.0.0.1)
  COSMOS_BACKEND_PORT      Port (default 8000)
  COSMOS_MAX_INPUT_LEN     Input token limit (default 1024)
  COSMOS_MAX_KV_CAPACITY   KV token capacity (default 2048)
  COSMOS_MAX_IMAGE_TOKENS  Built aggregate override (selected compact MLP: 512).
                          Unset or blank preserves the original builder default.
  COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE  Built per-image override (selected
                          compact MLP: 512). An explicit blank preserves the
                          original FP16 cache's None builder override instead.
  COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE  Runtime image cap (default 512; UI can override)
                          Match the engine override to the selected cache. An absent
                          override can use this runtime value as a legacy build
                          override; blank and absent differ.
  COSMOS_ENCODER_CACHE_BYTES  Encoder embedding cache budget in bytes (default 0)
  COSMOS_STATIC_CLOCKS    Selected 0; dynamic clocks within the existing power mode.
                          This standalone launcher does not apply clocks itself.
  COSMOS_TOP_P            Runtime top_p default (default 1; requests can override)
Image tokens are independent of max output tokens (UI default 64, editable to 512).
Apply both runtime-control patches and rebuild the native binding before launch.
HELP
  exit 0
fi

preflight_only=0
if [[ "${1:-}" == --preflight-only ]]; then
  preflight_only=1
  shift
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="${EDGE_LLM_DIR:-$project_dir/external/TensorRT-Edge-LLM}"
python_bin="${EDGE_LLM_PYTHON:-$upstream_dir/.venv/bin/python}"
cache_dir="${COSMOS_CACHE_DIR:-$project_dir/data/engine-cache-chw}"
checkpoint="${1:-}"
expected_revision=e8b29522938901f6df19ebeedd4b69bc8edbcd97

if [[ $# != 1 || "$checkpoint" != /* || ! -d "$checkpoint" ]]; then
  printf 'Provide one absolute local reasoner checkpoint directory. See --help.\n' >&2
  exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != aarch64 ]]; then
  printf 'This backend requires Linux aarch64 on the Jetson Orin.\n' >&2
  exit 2
fi
if [[ ! -r /proc/device-tree/model ]] || ! tr '\000' '\n' < /proc/device-tree/model | grep -qi 'Jetson.*Orin'; then
  printf 'An NVIDIA Jetson Orin device-tree identity is required.\n' >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  printf 'Missing backend Python: %s. Follow docs/backend-build.md.\n' "$python_bin" >&2
  exit 2
fi
if [[ "$(git -C "$upstream_dir" rev-parse HEAD)" != "$expected_revision" ]]; then
  printf 'Backend checkout differs from the inspected v0.10.1 pin.\n' >&2
  exit 2
fi

"$python_bin" - "$checkpoint" "$upstream_dir" "$cache_dir" "$project_dir" "$preflight_only" <<'PY'
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
config = json.loads((root / 'config.json').read_text())
if config.get('model_type') != 'cosmos3_edge' or not config.get('vision_config'):
    raise SystemExit('Expected a Cosmos3-Edge multimodal reasoner checkpoint.')
for relative in ('transformer/config.json', 'vae/config.json'):
    if (root / relative).exists():
        raise SystemExit(f'Reasoner-only launch refuses generator config: {relative}')
# Match the pinned builder's precedence: hf_quant_config.json wins globally;
# otherwise each component uses its own quantization_config or the root's.
quant_file = root / 'hf_quant_config.json'
if quant_file.exists():
    quant_sources = [('hf_quant_config.json', json.loads(quant_file.read_text()).get('quantization', {}))]
else:
    quant_sources = [
        (f'config.json:{name}', (config.get(name) or {}).get('quantization_config')
         or config.get('quantization_config') or {})
        for name in ('text_config', 'vision_config')
    ]

def validate_orin_quantization(value, source):
    if not isinstance(value, dict):
        raise SystemExit(f'Expected quantization metadata object in {source}.')
    if any(kind in json.dumps(value).upper() for kind in ('FP8', 'FP4')):
        raise SystemExit(f'Orin-incompatible quantization metadata in {source}.')
    for key, item in value.items():
        # Any nonempty embedded KV scheme selects FP8 in the pinned builder,
        # even when its contents do not contain the literal string "FP8".
        if key == 'kv_cache_scheme' and item:
            raise SystemExit(f'Orin requires FP16 KV cache; found {source}.{key}.')
        if key == 'kv_cache_quant_algo' and item:
            if str(item).strip().lower() not in ('', 'fp16'):
                raise SystemExit(f'Orin requires FP16 KV cache; found {source}.{key}.')
        if isinstance(item, dict):
            validate_orin_quantization(item, f'{source}.{key}')
        elif isinstance(item, list):
            for index, child in enumerate(item):
                if isinstance(child, dict):
                    validate_orin_quantization(child, f'{source}.{key}[{index}]')

for source, quant in quant_sources:
    validate_orin_quantization(quant, source)
index_file = root / 'model.safetensors.index.json'
if index_file.exists():
    index = json.loads(index_file.read_text())
    for relative in set(index['weight_map'].values()):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SystemExit(f'Missing or out-of-directory checkpoint shard: {relative}')
elif not (root / 'model.safetensors').is_file():
    raise SystemExit('Missing safetensors checkpoint or index.')

sys.path.insert(0, str(pathlib.Path(sys.argv[4]) / 'scripts'))
from preflight_cosmos_artifacts import validate
try:
    sys.path.insert(0, sys.argv[2])
    from experimental.server.runtime.engine_build import BuildOptions
    options = BuildOptions(max_input_len=int(os.environ.get('COSMOS_MAX_INPUT_LEN', '1024')),
        max_kv_cache_capacity=int(os.environ.get('COSMOS_MAX_KV_CAPACITY', '2048')), max_batch_size=1,
        max_image_tokens=int(os.environ['COSMOS_MAX_IMAGE_TOKENS']) if os.environ.get('COSMOS_MAX_IMAGE_TOKENS') else None,
        max_image_tokens_per_image=int(os.environ.get('COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE', os.environ.get('COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE'))) if os.environ.get('COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE', os.environ.get('COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE')) else None)
    validate(root, sys.argv[2], sys.argv[3],
             max_input_len=int(os.environ.get('COSMOS_MAX_INPUT_LEN', '1024')),
             max_kv_capacity=int(os.environ.get('COSMOS_MAX_KV_CAPACITY', '2048')),
             preflight_only=sys.argv[5] == '1', build_options=options)
except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as error:
    raise SystemExit('Cosmos3 artifact preflight refused: ' + str(error)) from error
import tensorrt
if tensorrt.__version__.split('.')[0] != '10':
    raise SystemExit(f'Expected JetPack TensorRT 10; found {tensorrt.__version__}.')
print(f'Checkpoint validated structurally; TensorRT {tensorrt.__version__}.', flush=True)
PY

if [[ "$preflight_only" == 1 ]]; then
  exit 0
fi

export TRT_PACKAGE_DIR=/usr
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
cd "$upstream_dir"
visual_args=()
if [[ -n "${COSMOS_MAX_IMAGE_TOKENS:-}" ]]; then
  visual_args+=(--max-image-tokens "$COSMOS_MAX_IMAGE_TOKENS")
fi
if [[ -n "${COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE-${COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE:-}}" ]]; then
  visual_args+=(--max-image-tokens-per-image "${COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE-$COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE}")
fi
printf 'Starting actual TensorRT Edge-LLM; health remains unavailable until engine load succeeds.\n'
exec "$python_bin" "$project_dir/scripts/serve_backend.py" "$checkpoint" \
  --encoder-embedding-cache-budget-bytes "${COSMOS_ENCODER_CACHE_BYTES:-0}" \
  --host "${COSMOS_BACKEND_HOST:-127.0.0.1}" \
  --port "${COSMOS_BACKEND_PORT:-8000}" \
  --served-model-name Cosmos3-Edge \
  --cache-dir "$cache_dir" \
  --engine-cache-max-size-gb 12 \
  --max-input-len "${COSMOS_MAX_INPUT_LEN:-1024}" \
  --max-kv-cache-capacity "${COSMOS_MAX_KV_CAPACITY:-2048}" \
  --max-batch-size 1 \
  --max-queued-requests 1 \
  --queue-timeout 10 \
  --reasoning-parser none "${visual_args[@]}"
