#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: scripts/build_backend.sh [fp16|int4]

Build the pinned TensorRT-Edge-LLM runtime on a normally booted Orin.
Uses one compiler worker and the FMHA CuTe kernel group. The int4 mode adds
INT4 V2 kernels without deleting existing FMHA artifacts. This does not build
model engines or claim successful model inference. Logs go to results/build-*.
HELP
  exit 0
fi
mode="${1:-fp16}"
[[ $# -le 1 && "$mode" =~ ^(fp16|int4)$ ]] || { printf 'Expected fp16 or int4.\n' >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || { printf 'Run on the Jetson Orin.\n' >&2; exit 2; }
[[ -r /proc/device-tree/model ]] && tr '\000' '\n' < /proc/device-tree/model | grep -qi 'Jetson.*Orin'
[[ "$(findmnt -n -o FSTYPE /)" == ext4 ]] || { printf 'Boot the installed OS before building.\n' >&2; exit 2; }
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream="$project/external/TensorRT-Edge-LLM"
[[ "$(git -C "$upstream" rev-parse HEAD)" == e8b29522938901f6df19ebeedd4b69bc8edbcd97 ]]
python_bin="$upstream/.venv/bin/python"
[[ -x "$python_bin" ]]
receipt="$(mktemp -d "$project/results/build-$mode.XXXXXX")"
export PATH="$upstream/.venv/bin:/usr/local/cuda-13.2/bin:$PATH"
export TRT_PACKAGE_DIR=/usr
export CUDA_PATH=/usr/local/cuda-13.2
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="$project/data/tmp"
export XDG_CACHE_HOME="$project/data/cache"
export CUDA_CACHE_PATH="$project/data/cache/cuda"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
exec > >(tee "$receipt/build.log") 2>&1
monitor_pid=
finish_build() {
  rc=$?
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$receipt/ended-at.txt"
  printf '%s\n' "$rc" > "$receipt/exit-code"
}
trap finish_build EXIT
if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 500 > "$receipt/tegrastats.log" 2>&1 &
  monitor_pid=$!
fi
printf 'Build receipt: %s\n' "$receipt"
date -u '+%Y-%m-%dT%H:%M:%SZ'
"$python_bin" -m pip check
"$python_bin" - "$receipt" <<'PY'
import datetime,json,pathlib,sys
import cupy as cp
import tensorrt
assert tensorrt.__version__.split('.')[0]=='10', tensorrt.__version__
assert cp.cuda.runtime.getDeviceCount()==1
props=cp.cuda.runtime.getDeviceProperties(0)
assert (props['major'],props['minor'])==(8,7), props
values=cp.arange(1024,dtype=cp.int32)
observed=int(values.sum().get())
assert observed==523776, observed
cp.cuda.runtime.deviceSynchronize()
name=props['name']; name=name.decode() if isinstance(name,bytes) else name
record={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':name,'compute_capability':[8,7],'cuda_runtime':cp.cuda.runtime.runtimeGetVersion(),'tensorrt':tensorrt.__version__,'cuda_array_sum_passed':True,'model_inference_verified':False}
pathlib.Path(sys.argv[1],'cuda-preflight.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
cd "$upstream"
groups=fmha
[[ "$mode" != int4 ]] || groups=fmha,int4_fp16_gemm
"$python_bin" kernelSrcs/build_cutedsl.py --gpu_arch sm_87 --arch aarch64 \
  --cuda-version 13.2 --kernels "$groups" --jobs 1
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=13.2 \
  "-DENABLE_CUTE_DSL=${groups//,/;}" -DBUILD_PYTHON_BINDINGS=ON \
  "-DPython_EXECUTABLE=$python_bin" \
  "-Dpybind11_DIR=$("$python_bin" -m pybind11 --cmakedir)"
cmake --build build --parallel 1 --target _edgellm_runtime NvInfer_edgellm_plugin
"$python_bin" - <<'PY'
from experimental.server.runtime.engine import _import_runtime
runtime=_import_runtime()
assert hasattr(runtime,'LLMRuntime')
print('Native runtime import passed:',runtime.__file__)
PY
printf 'Native backend built. Model engine build and inference validation remain.\n'
