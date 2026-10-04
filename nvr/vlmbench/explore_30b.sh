#!/bin/bash
# Nemotron 3 Nano 30B-A3B on the 8 GB Orin Nano, contained: does it run at all, and how fast?
#
#   sudo bash explore_30b.sh
#
# Jetson AI Lab's catalogue ticks "Orin Nano 8GB" for this model, but its own page says 32 GB RAM
# and AGX Orin, and the smallest GGUF is 17 GB. The only way it can run here is llama.cpp
# demand-paging the mmapped file from the SD card. This measures that, inside fences that keep the
# NVR safe: CPU only (CUDA_VISIBLE_DEVICES empty, so the GPU's NvMap allocator is never touched),
# a hard memory cap (EXPLORE_MEMORY_MAX - page cache for the mapped file counts against it, so the
# weights stream instead of crowding Frigate out), a read-bandwidth cap on the SD card so recording
# writes are not starved, three CPU cores, and a deadline. The shim is stopped for the window and
# everything that was running is started again on the way out.
set -u
IMAGE=${LLAMA_CPP_IMAGE:-ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin}
GGUF_DIR=${GGUF_30B_DIR:-/home/orin/vlm-sweep/gguf-30b}
MODEL=${GGUF_30B:-Nemotron-3-Nano-30B-A3B-Q4_0.gguf}
CAP=${EXPLORE_MEMORY_MAX:-2500m}
READ_BPS=${EXPLORE_READ_BPS:-40mb}
DEADLINE=${EXPLORE_DEADLINE_S:-480}
PORT=8093
NAME=vlmbench-30b

[ "$(id -u)" -eq 0 ] || { echo "run with sudo: it stops and starts the shim" >&2; exit 2; }
was_active=()
for unit in porch-dad cosmos-edge-ui; do
  systemctl is-active -q "$unit" && was_active+=("$unit")
done
restore() {
  docker rm -f "$NAME" >/dev/null 2>&1
  systemctl start cosmos3-edge-shim
  for unit in "${was_active[@]}"; do systemctl start "$unit"; done
  for _ in $(seq 1 60); do curl -sf -m 2 http://127.0.0.1:8000/health/ready >/dev/null && break; sleep 3; done
  echo "restored: shim $(systemctl is-active cosmos3-edge-shim); started again: ${was_active[*]:-none}"
}
trap restore EXIT

systemctl stop cosmos3-edge-shim
echo "shim stopped; MemAvailable $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MiB; model $(du -h "$GGUF_DIR/$MODEL" | cut -f1)"
dev=$(findmnt -n -o SOURCE --target "$GGUF_DIR" | sed 's/p[0-9]*$//')
t0=$(date +%s.%N)
cid=$(docker run -d --name "$NAME" --runtime nvidia --network host -e CUDA_VISIBLE_DEVICES= \
  --memory "$CAP" --memory-swap "$CAP" --cpus 3 --device-read-bps "$dev:$READ_BPS" \
  -v "$GGUF_DIR:/models:ro" "$IMAGE" \
  llama-server -m "/models/$MODEL" --host 127.0.0.1 --port "$PORT" -ngl 0 -c 512 -t 3 -np 1 --no-warmup) || exit 1
cg=/sys/fs/cgroup/system.slice/docker-$cid.scope
up=0
while [ "$(echo "$(date +%s.%N) - $t0 < $DEADLINE" | bc)" = 1 ]; do
  curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null && { up=1; break; }
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] || break
  sleep 2
done
load=$(echo "$(date +%s.%N) - $t0" | bc)
echo "server up=$up after ${load}s; container memory $(( $(cat $cg/memory.current 2>/dev/null || echo 0) / 1048576 )) MiB"
if [ $up -eq 1 ]; then
  left=$(echo "$DEADLINE - $load" | bc | cut -d. -f1)
  python3 - "$PORT" "$left" <<'EOF'
import json, sys, time, urllib.request
port, budget = sys.argv[1], max(30, int(sys.argv[2]))
body = {"messages": [{"role": "user", "content": "In one sentence, what is a Jetson Orin Nano?"}],
        "max_tokens": 24, "temperature": 0, "stream": True}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0, stamps, text = time.time(), [], ""
try:
    with urllib.request.urlopen(req, timeout=budget) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            for ch in json.loads(line[5:]).get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("reasoning_content")
                if piece:
                    stamps.append(time.time() - t0)
                    text += piece
except Exception as e:
    print(f"request ended: {type(e).__name__}: {e}")
n = len(stamps)
print(f"tokens={n} ttft_s={stamps[0]:.1f}" if n else "tokens=0 (no token inside the deadline)")
if n > 1:
    print(f"decode {(n - 1) / (stamps[-1] - stamps[0]):.3f} tok/s; text so far: {text[:120]!r}")
EOF
fi
echo "peak $(( $(cat $cg/memory.peak 2>/dev/null || echo 0) / 1048576 )) MiB, oom_kill=$(awk '/oom_kill / {print $2}' $cg/memory.events 2>/dev/null)," \
     "read $(awk '/rbytes=/ {for (i=1;i<=NF;i++) if ($i ~ /^rbytes=/) {split($i,a,"="); s+=a[2]}} END {print int(s/1048576)}' $cg/io.stat 2>/dev/null) MiB from storage"
docker logs "$NAME" 2>&1 | grep -E "model size|n_params|mmap|error|CPU_Mapped|load time" | tail -6 | sed 's/^/   /'
