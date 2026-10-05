#!/bin/bash
# Cross-compile upstream llama.cpp for the Orin Nano (aarch64, CUDA 13.0, sm_87) on an x86 machine
# with Docker, and copy llama-server to the Orin. Run on the workstation, not the Orin:
#
#   bash build_llama_upstream.sh [git ref, default origin/master] [orin ssh target, default orin@orin]
#
# Why: NVIDIA's ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin (b10373, built 2026-08-12)
# predates llama.cpp's --lazy-mode (2026-08-27/30), which reads Gemma 4's 1.5 GiB per-layer
# embedding table from disk on demand instead of holding it resident. With it, Gemma 4 E2B fits
# beside the NVR; without it, it was OOM-killed at 4.1 GB. The binary is static apart from the
# CUDA runtime, so it runs inside that same image: run_llama_bench.sh LLAMA_BIN_DIR=... and the
# Gemma proxy.json mount it at /opt/llama-upstream. Building on the Orin itself would take hours
# beside Frigate and its compilers would not fit in the memory a loaded model leaves.
set -euo pipefail
REF=${1:-origin/master}
ORIN=${2:-orin@orin}
WORK=${LLAMA_CROSS_DIR:-/tmp/llama-cross}
DEST=/home/orin/tensorrt-edgellm-workspace/llama-upstream
mkdir -p "$WORK"
cat > "$WORK/build.sh" <<'EOS'
#!/bin/bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
if [ ! -f /work/.deps-done ]; then
  # The aarch64 (sbsa) CUDA libraries come from NVIDIA's cross-compile repo, same signing key.
  echo 'deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/cross-linux-sbsa /' \
    > /etc/apt/sources.list.d/cuda-cross.list
  apt-get update -q
  apt-get install -y -q --no-install-recommends cuda-cross-sbsa-13-0 g++-aarch64-linux-gnu \
    cmake ninja-build git ca-certificates
  touch /work/.deps-done
fi
# CMake's FindCUDAToolkit looks for targets/aarch64-linux when cross-compiling to aarch64.
[ -e /usr/local/cuda/targets/aarch64-linux ] || ln -s sbsa-linux /usr/local/cuda/targets/aarch64-linux
cd /work
[ -d llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git fetch -q origin && git checkout -q "$LLAMA_REF"
git log -1 --format='%h %cd %s' > /work/VERSION
cat > /work/toolchain.cmake <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu /usr/local/cuda/targets/sbsa-linux)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
EOF
cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=/work/toolchain.cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_FLAGS="-target-dir sbsa-linux" \
  -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build -j "$(nproc)" --target llama-server
mkdir -p /work/out && cp build/bin/llama-server /work/out/
EOS
docker run --rm -e LLAMA_REF="$REF" -v "$WORK:/work" nvidia/cuda:13.0.1-devel-ubuntu24.04 bash /work/build.sh
ssh "$ORIN" "mkdir -p $DEST/bin"
scp "$WORK/out/llama-server" "$ORIN:$DEST/bin/llama-server"
scp "$WORK/VERSION" "$ORIN:$DEST/VERSION"
echo "llama-server $(cat "$WORK/VERSION") -> $ORIN:$DEST/bin"
