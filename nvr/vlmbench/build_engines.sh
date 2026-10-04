#!/bin/bash
# Build TensorRT-Edge-LLM engines for benchmark candidates, on the Orin, in one window.
#
#   sudo bash build_engines.sh <name>[:<minImageTokens>:<maxImageTokens>:<maxImageTokensPerImage>] ...
#
# <name> is a directory under ~orin/tensorrt-edgellm-workspace holding onnx/llm and, for a VLM,
# onnx/visual. Engines land in <name>/engines/reasoning (the visual engine in its visual/), and
# <name>/model links to onnx/llm, which is where the shim looks for an engine's own checkpoint.
#
# A ~2B INT4 build peaks around 3.9 GB, and with the NVR up the board only has that much once
# Cosmos3-Edge is unloaded, so the shim is stopped for the window. systemd's Requires= takes
# porch-dad and Live Vision down with it; whatever was running is started again on the way
# out - success, failure or Ctrl-C - with the shim back on its default engine. Every build runs
# as orin inside a scope capped at BUILD_MEMORY_MAX (swap allowed), so a build that outgrows
# the board is killed on its own instead of the OOM killer choosing among Frigate's processes.
#
# Limits match the deployed Cosmos v3 engine, so every candidate is measured in the same
# envelope: batch 1, 1024 input tokens, 1024 KV, 64-640 image tokens with 320 per image. InternVL
# encodes 448 px tiles of 256 tokens and its builder wants both bounds in multiples of 256, so it
# is built 256:512:320 - one tile, 256 tokens, the nearest it gets to Cosmos's 320.
set -u
W=/home/orin/tensorrt-edgellm-workspace
B=/home/orin/TensorRT-Edge-LLM/build/examples
PLUGIN=/home/orin/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
CAP=${BUILD_MEMORY_MAX:-3400M}
SWAP=${BUILD_SWAP_MAX:-1536M}

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

# Runs "$@" as orin in its own capped scope; the last log line is the scope's memory.peak.
capped() {
  systemd-run --scope -q -p MemoryMax="$CAP" -p MemorySwapMax="$SWAP" \
    runuser -u orin -- env EDGELLM_PLUGIN_PATH="$PLUGIN" bash -c \
      '"$@"; rc=$?; echo "peak_mib=$(( $(cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.peak) / 1048576 ))"; exit $rc' \
      capped "$@"
}
peak() { sed -n 's/^peak_mib=//p' "$1" | tail -1; }

systemctl stop cosmos3-edge-shim
echo "shim stopped; MemAvailable $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MiB"

status=0
for spec in "$@"; do
  IFS=: read -r name min_tokens max_tokens per_image <<< "$spec"
  min_tokens=${min_tokens:-64}; max_tokens=${max_tokens:-640}; per_image=${per_image:-320}
  d=$W/$name
  echo "== $name"
  t0=$(date +%s)
  if ! capped "$B/llm/llm_build" --onnxDir "$d/onnx/llm" --engineDir "$d/engines/reasoning" \
      --maxBatchSize 1 --maxInputLen 1024 --maxKVCacheCapacity 1024 > "$d/llm_build.log" 2>&1; then
    echo "   llm_build FAILED after $(( $(date +%s) - t0 ))s:"; tail -5 "$d/llm_build.log"; status=1; continue
  fi
  echo "   llm  $(( $(date +%s) - t0 ))s, peak $(peak "$d/llm_build.log") MiB," \
       "$(du -h "$d/engines/reasoning/llm.engine" | cut -f1)"
  if [ -d "$d/onnx/visual" ]; then
    t0=$(date +%s)
    if ! capped "$B/multimodal/visual_build" --onnxDir "$d/onnx/visual" --engineDir "$d/engines/reasoning" \
        --minImageTokens "$min_tokens" --maxImageTokens "$max_tokens" --maxImageTokensPerImage "$per_image" \
        > "$d/visual_build.log" 2>&1; then
      echo "   visual_build FAILED after $(( $(date +%s) - t0 ))s:"; tail -5 "$d/visual_build.log"; status=1; continue
    fi
    echo "   visual $(( $(date +%s) - t0 ))s, peak $(peak "$d/visual_build.log") MiB," \
         "$(du -h "$d/engines/reasoning/visual/visual.engine" | cut -f1)"
  fi
  ln -sfn onnx/llm "$d/model"
  chown -h orin:orin "$d/model"
done
exit $status
