#!/bin/bash
# Bring each TensorRT-Edge-LLM candidate up in the shim, one at a time, and run the benchmark on it.
#
#   sudo bash run_trt_bench.sh <label>=<engine dir>[=<family>] ...
#
# The shim serves whatever /opt/tensorrt-edgellm/models/default points at, so each candidate is
# brought up exactly the way Live Vision switches models: repoint the link, restart the shim. For
# the window the shim gets a runtime-only MemoryMax (SHIM_MEMORY_MAX, default 3300M, under what the
# board has free with the NVR up): a candidate that outgrows it is OOM-killed inside its own cgroup
# and recorded as not fitting, instead of the kernel picking a victim among Frigate's processes.
# Extra bench.py flags can be passed in BENCH_ARGS (e.g. "--limit 2" for a smoke test).
# On exit - success, failure or Ctrl-C - the link goes back to what it was, the cap is lifted and
# the shim is restarted on its usual engine. porch-dad and Live Vision restart with the shim
# (Requires=), and while a candidate is up it also answers Frigate's GenAI requests.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LINK=/opt/tensorrt-edgellm/models/default
DATA=${VLMBENCH_DATA:-/home/orin/nvr/vlmbench/data}
RESULTS=${VLMBENCH_RESULTS:-/home/orin/nvr/vlmbench/results}
CAP=${SHIM_MEMORY_MAX:-3300M}

[ "$(id -u)" -eq 0 ] || { echo "run with sudo: it repoints the engine link and restarts the shim" >&2; exit 2; }
original=$(readlink "$LINK")

wait_ready() {
  for _ in $(seq 1 100); do
    curl -sf -m 2 http://127.0.0.1:8000/health/ready >/dev/null && return 0
    systemctl is-failed -q cosmos3-edge-shim && return 1
    sleep 3
  done
  return 1
}

restore() {
  ln -sfn "$original" "$LINK"
  systemctl set-property --runtime cosmos3-edge-shim MemoryMax=infinity
  systemctl reset-failed cosmos3-edge-shim 2>/dev/null
  systemctl restart cosmos3-edge-shim
  wait_ready
  echo "restored: $LINK -> $(readlink "$LINK"), shim $(systemctl is-active cosmos3-edge-shim)," \
       "ready=$(curl -s -m 2 http://127.0.0.1:8000/health/ready)"
}
trap restore EXIT

systemctl set-property --runtime cosmos3-edge-shim MemoryMax="$CAP"
mkdir -p "$RESULTS" && chown orin:orin "$RESULTS"
for spec in "$@"; do
  IFS== read -r label engine family <<< "$spec"
  echo "== $label ($engine)"
  ln -sfn "$engine" "$LINK"
  systemctl reset-failed cosmos3-edge-shim 2>/dev/null
  t0=$(date +%s)
  systemctl restart cosmos3-edge-shim
  if ! wait_ready; then
    echo "   did not come up in $(( $(date +%s) - t0 ))s:"
    journalctl -u cosmos3-edge-shim -n 8 --no-pager | sed 's/^/   /'
    grep -h oom /sys/fs/cgroup/system.slice/cosmos3-edge-shim.service/memory.events 2>/dev/null | sed 's/^/   /'
    continue
  fi
  echo "   ready in $(( $(date +%s) - t0 ))s; memory.current $(( $(cat /sys/fs/cgroup/system.slice/cosmos3-edge-shim.service/memory.current) / 1048576 )) MiB"
  runuser -u orin -- python3 "$HERE/bench.py" --data "$DATA" $( [ -f "$DATA/labels.json" ] && echo --labels "$DATA/labels.json" ) ${BENCH_ARGS:-} \
    --label "$label" ${family:+--family "$family"} --out "$RESULTS/$label.jsonl"
  echo "   peak $(( $(cat /sys/fs/cgroup/system.slice/cosmos3-edge-shim.service/memory.peak) / 1048576 )) MiB," \
       "oom_kill=$(awk '/oom_kill / {print $2}' /sys/fs/cgroup/system.slice/cosmos3-edge-shim.service/memory.events)"
done
