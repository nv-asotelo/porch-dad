#!/bin/bash
# Run the benchmark on NVIDIA LocateAnything-3B (INT4, PyTorch) on the Orin.
#
#   sudo bash run_la_bench.sh [bench args...]
#
# LocateAnything has no TensorRT path, so it runs in PyTorch from the TensorRT-Edge-LLM venv (its
# CUDA build of torch) with transformers 4.57.1 from a --target overlay, and INT4 weights from
# locateanything/la_int4.py. Only one model fits beside the NVR, so the shim is stopped for the
# window and everything that was running is started again on the way out, as in build_engines.sh.
# The run is capped (LA_MEMORY_MAX) in its own scope, which on Jetson also counts the GPU's nvmap
# buffers, so a model that outgrows the board fails on its own.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${LA_MODEL_DIR:-/home/orin/tensorrt-edgellm-workspace/LocateAnything-3B-INT4/model}
WEIGHTS=${LA_WEIGHTS:-/home/orin/tensorrt-edgellm-workspace/LocateAnything-3B-INT4/LocateAnything-3B-int4.safetensors}
OVERLAY=${LA_OVERLAY:-/home/orin/tensorrt-edgellm-workspace/LocateAnything-3B-INT4/overlay}
PY=/home/orin/TensorRT-Edge-LLM/.venv/bin/python
DATA=${VLMBENCH_DATA:-/home/orin/nvr/vlmbench/data}
RESULTS=${VLMBENCH_RESULTS:-/home/orin/nvr/vlmbench/results}
CAP=${LA_MEMORY_MAX:-3300M}

[ "$(id -u)" -eq 0 ] || { echo "run with sudo: it stops and starts the shim" >&2; exit 2; }
was_active=()
for unit in porch-dad cosmos-edge-ui; do
  systemctl is-active -q "$unit" && was_active+=("$unit")
done
restore() {
  systemctl start cosmos3-edge-shim
  for unit in "${was_active[@]}"; do systemctl start "$unit"; done
  for _ in $(seq 1 60); do
    curl -sf -m 2 http://127.0.0.1:8000/health/ready >/dev/null && break
    sleep 3
  done
  echo "restored: shim $(systemctl is-active cosmos3-edge-shim), ready=$(curl -s -m 2 http://127.0.0.1:8000/health/ready);" \
       "started again: ${was_active[*]:-none}"
}
trap restore EXIT

systemctl stop cosmos3-edge-shim
echo "shim stopped; MemAvailable $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MiB"
mkdir -p "$RESULTS" && chown orin:orin "$RESULTS"
labels=()
[ -f "$DATA/labels.json" ] && labels=(--labels "$DATA/labels.json")
systemd-run --scope -q -p MemoryMax="$CAP" -p MemorySwapMax=0 \
  runuser -u orin -- env PYTHONPATH="$OVERLAY:$MODEL" HF_HUB_OFFLINE=1 \
  "$PY" "$HERE/locateanything/la_bench.py" --model "$MODEL" --weights "$WEIGHTS" --data "$DATA" "${labels[@]}" \
  --out "$RESULTS/locateanything-3b-int4.jsonl" "$@"
echo "exit=$?"
