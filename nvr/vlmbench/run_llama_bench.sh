#!/bin/bash
# Bring a llama.cpp model up in NVIDIA's Jetson container and run the benchmark on it.
#
#   sudo bash run_llama_bench.sh <label> <model.gguf> [<mmproj.gguf>|-] [bench.py args...]
#
# Gemma 4 E2B and Nemotron 3 Nano 4B have no TensorRT-Edge-LLM path that fits this board (Gemma's
# 4.7 GB FP16 per-layer embedding table is loaded whole; Nemotron's 3136-wide projections do not
# divide into the INT4 kernel's 128-wide groups), so they run the way Jetson AI Lab runs them on
# Orin Nano: llama-server from ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin. GGUFs are read
# from $GGUF_DIR. Only one model fits beside the NVR, so the shim is stopped for the window;
# whatever was running is started again on the way out, as in build_engines.sh. The container gets
# a hard memory cap (LLAMA_MEMORY_MAX), which on Jetson also covers the GPU's nvmap buffers, so a
# model that does not fit fails inside its container. JetPack 7.2's NvMap error-12 workaround from
# Jetson AI Lab is applied: unified memory, no auto-fit, explicit context, no prompt cache.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${LLAMA_CPP_IMAGE:-ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin}
GGUF_DIR=${GGUF_DIR:-/home/orin/tensorrt-edgellm-workspace/gguf}
DATA=${VLMBENCH_DATA:-/home/orin/nvr/vlmbench/data}
RESULTS=${VLMBENCH_RESULTS:-/home/orin/nvr/vlmbench/results}
CAP=${LLAMA_MEMORY_MAX:-3300m}
PORT=${LLAMA_PORT:-8091}
CTX=${LLAMA_CTX:-2048}
NAME=vlmbench-llama

[ "$(id -u)" -eq 0 ] || { echo "run with sudo: it stops and starts the shim" >&2; exit 2; }
label=$1 model=$2 mmproj=${3:--}
shift 3 2>/dev/null || shift $#

was_active=()
for unit in porch-dad cosmos-edge-ui; do
  systemctl is-active -q "$unit" && was_active+=("$unit")
done

restore() {
  docker rm -f "$NAME" >/dev/null 2>&1
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
args=(-m "/models/$model" --host 127.0.0.1 --port "$PORT" -ngl 999 -c "$CTX" -b 512 -ub 512 -np 1
      --fit off --cache-ram 0 --jinja)
[ "$mmproj" != "-" ] && args+=(--mmproj "/models/$mmproj")
t0=$(date +%s)
cid=$(docker run -d --name "$NAME" --runtime nvidia --network host --memory "$CAP" --memory-swap "$CAP" \
  -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 -v "$GGUF_DIR:/models:ro" "$IMAGE" llama-server "${args[@]}") || exit 1
up=0
for _ in $(seq 1 120); do
  curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null && { up=1; break; }
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ] || break
  sleep 2
done
cgroup=/sys/fs/cgroup/system.slice/docker-$cid.scope
if [ $up -ne 1 ]; then
  echo "== $label did not come up in $(( $(date +%s) - t0 ))s (oom_kill=$(awk '/oom_kill / {print $2}' "$cgroup/memory.events" 2>/dev/null)):"
  docker logs --tail 15 "$NAME" 2>&1 | sed 's/^/   /'
  exit 1
fi
echo "== $label up in $(( $(date +%s) - t0 ))s; memory.current $(( $(cat "$cgroup/memory.current") / 1048576 )) MiB"
mkdir -p "$RESULTS" && chown orin:orin "$RESULTS"
runuser -u orin -- python3 "$HERE/bench.py" --url "http://127.0.0.1:$PORT/v1/chat/completions" --cgroup "$cgroup" \
  --data "$DATA" $( [ -f "$DATA/labels.json" ] && echo --labels "$DATA/labels.json" ) ${BENCH_ARGS:-} --label "$label" --out "$RESULTS/$label.jsonl" "$@"
echo "   peak $(( $(cat "$cgroup/memory.peak") / 1048576 )) MiB, oom_kill=$(awk '/oom_kill / {print $2}' "$cgroup/memory.events")"
docker logs "$NAME" 2>&1 | grep -E "load time|model buffer size|KV self size|compute buffer size|CUDA0 model buffer|mmproj|clip" | tail -12 | sed 's/^/   /'
