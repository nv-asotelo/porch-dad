#!/usr/bin/env bash
# Phase 0 gate. Refuse to start on a board this runbook was not measured on.
#
# Every check here maps to a failure that actually happened, and all of them surface LATE - a
# version mismatch shows up as an engine build error or a load-time OOM three phases downstream,
# where it looks like a quantization problem. Fail here instead.
set -uo pipefail

fail=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=1; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }

echo "== Phase 0 preflight =="

# --- board -------------------------------------------------------------------------------------
model=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)
case "$model" in
  *"Orin Nano"*) ok "board: $model" ;;
  *) bad "board: $model — this runbook is measured on Jetson Orin Nano Super 8 GB only" ;;
esac

# 8 GB board reports ~7.4 GiB MemTotal. A 4 GB Nano cannot hold this model at all; a 16 GB
# Orin NX will work but every RAM figure in the runbook will be wrong.
mem_gb=$(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo)
if awk "BEGIN{exit !($mem_gb > 6.5 && $mem_gb < 8.5)}"; then
  ok "memory: ${mem_gb} GiB total (8 GB board)"
else
  bad "memory: ${mem_gb} GiB — expected ~7.4 GiB. Engine sizing in this runbook assumes 8 GB"
fi

# --- no NVENC, no FP8: both are sm_89+, this board is sm_87 ------------------------------------
cap=$(python3 - <<'PY' 2>/dev/null || echo unknown
try:
    import torch; print("".join(str(x) for x in torch.cuda.get_device_capability(0)))
except Exception: print("unknown")
PY
)
[ "$cap" = "87" ] && ok "compute capability: sm_87 (no FP8, no NVENC — expected)" \
                  || warn "compute capability: $cap (expected 87; could not confirm)"

# --- software stack ----------------------------------------------------------------------------
jp=$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null | cut -d- -f1)
case "${jp:-}" in
  7.*) ok "JetPack: $jp" ;;
  "")  warn "JetPack: could not read nvidia-jetpack package version" ;;
  *)   bad "JetPack: $jp — measured on 7.2.1; TensorRT majors differ across JetPack lines" ;;
esac

cuda=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
case "${cuda:-}" in
  13.*) ok "CUDA: $cuda" ;;
  "")   bad "CUDA: nvcc not on PATH — add /usr/local/cuda/bin" ;;
  *)    bad "CUDA: $cuda — measured on 13.2" ;;
esac

trt=$(python3 -c 'import tensorrt;print(tensorrt.__version__)' 2>/dev/null)
case "${trt:-}" in
  10.*) ok "TensorRT: $trt" ;;
  11.*) warn "TensorRT: $trt — newer than the measured 10.16.2.10. USE_TRT_NATIVE_ATTN may now work" ;;
  "")   bad "TensorRT: python module not importable" ;;
  *)    bad "TensorRT: $trt — measured on 10.16.2.10" ;;
esac

# --- disk: checkpoint + INT4 copy + ONNX + engines all coexist during the build -----------------
free_gb=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ "${free_gb:-0}" -ge 40 ]; then
  ok "disk: ${free_gb} GiB free in \$HOME"
else
  bad "disk: ${free_gb} GiB free — need ~40 GiB; checkpoint, INT4 copy, ONNX and engines coexist"
fi

# --- the relative-path trap --------------------------------------------------------------------
if [ -n "${EDGELLM_PLUGIN_PATH:-}" ] && [ -f "${EDGELLM_PLUGIN_PATH}" ]; then
  ok "EDGELLM_PLUGIN_PATH set and resolves: $EDGELLM_PLUGIN_PATH"
elif [ -f "$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so" ]; then
  warn "EDGELLM_PLUGIN_PATH unset — it DEFAULTS TO A RELATIVE PATH. Export it before any build:
         export EDGELLM_PLUGIN_PATH=\$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
else
  warn "plugin library not built yet — expected until Phase 2 completes"
fi

echo
[ "$fail" -eq 0 ] && echo "preflight PASSED — proceed to Phase 1" \
                  || echo "preflight FAILED — do not proceed; fix the FAIL lines above"
exit "$fail"
