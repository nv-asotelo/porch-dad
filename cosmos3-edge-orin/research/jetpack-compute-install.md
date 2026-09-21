# Minimal conventional compute install for the 8 GB Orin

**Status: source-verified recipe; not installed or tested on the device.** The actual board is P3767-0005, 8 GB. This recipe targets its prepared Ubuntu 24.04 / Jetson Linux R39.2.1 rootfs and public TensorRT-Edge-LLM v0.10.1 (`e8b29522938901f6df19ebeedd4b69bc8edbcd97`). It installs ordinary NVIDIA development packages without modifying upstream CMake. No package installation or hardware operation was performed for this research.

## Repository and rootfs verification

The correct NVIDIA repositories are `jetson/common` and `jetson/som`, suite **`r39.2`**, architecture **`arm64`**. The official `nvidia-l4t-apt-source_39.2.1-20260806224157_arm64.deb` contains these entries, also enables `jetson/ffmpeg r39.2`, installs `/etc/apt/trusted.gpg.d/jetson-ota-public.asc`, and supplies NVIDIA's priority-600 repository preference. No separate CUDA desktop/SBSA repository or replacement signing-key setup is needed. [R39.2.1 package documentation](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html), [official source package](https://repo.download.nvidia.com/jetson/som/pool/main/n/nvidia-l4t-apt-source/nvidia-l4t-apt-source_39.2.1-20260806224157_arm64.deb)

`apply_binaries.sh` prepares the NVIDIA rootfs packages, and the official documentation says the apt-source package and key are preinstalled. Therefore the prepared rootfs is expected to be sufficient. **This research did not inspect the actual prepared rootfs's apt files.** Verify after boot, before adding or changing repositories:

```bash
uname -m
cat /etc/os-release
head -n 1 /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core nvidia-l4t-apt-source
cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
test -s /etc/apt/trusted.gpg.d/jetson-ota-public.asc
cat /etc/apt/preferences.d/nvidia-repo-pin
python3 --version
df -h / /usr/local
```

Expected: `aarch64`, Ubuntu `24.04`, R39 revision `2.1`, Python `3.12`, and the two compute sources below. If they are absent or disagree, diagnose the BSP installation first; do not bypass signature checks.

```text
deb https://repo.download.nvidia.com/jetson/common r39.2 main
deb https://repo.download.nvidia.com/jetson/som r39.2 main
```

We verified both repositories' `InRelease` signatures with the public key shipped in the apt-source package, and matched the uncompressed `Packages` SHA-256 values against those signed manifests. Signing subkey: `13804AEEB181616F3B4964270D296FFB880FB004`; primary key: `3C6D1FF3100C8C3ABB0869C0E6543461A9996195`. Package records, download URLs, published hashes, and signature output are preserved in [metadata evidence](../results/jetpack-compute-metadata.json). [Common signed metadata](https://repo.download.nvidia.com/jetson/common/dists/r39.2/InRelease), [SoM signed metadata](https://repo.download.nvidia.com/jetson/som/dists/r39.2/InRelease)

## Install the pinned components

The published **JetPack `7.2.1-b49` apt metadata** selects CUDA toolkit meta-version `13.2.2-1`; its compiler/runtime component build is **`13.2.86-1`**. TensorRT is **`10.16.2.10-1+cuda13.2`**. These are the actual repository versions, despite the JetPack download page's broader CUDA 13.2.1 description. Do not substitute older `13.2.75`/`13.2.78` package builds also present in the repository. [Official package index](https://repo.download.nvidia.com/jetson/common/dists/r39.2/main/binary-arm64/Packages.gz)

Run in Bash on the booted Orin. Review the simulation's dependency versions, download size, and installed size before executing the final install command. No full-system upgrade is part of this recipe.

```bash
set -euo pipefail
test "$(dpkg --print-architecture)" = arm64
. /etc/os-release
test "$ID" = ubuntu && test "$VERSION_ID" = 24.04
dpkg-query -W -f='${Version}\n' nvidia-l4t-core | grep -q '^39\.2\.1-'

compute_packages=(
  cuda-nvcc-13-2=13.2.86-1
  cuda-cudart-dev-13-2=13.2.86-1
  cuda-driver-dev-13-2=13.2.86-1
  cuda-nvrtc-dev-13-2=13.2.86-1
  libcurand-dev-13-2=10.4.2.66-1
  libcublas-13-2=13.4.1.3-1
  libcufft-13-2=12.2.0.57-1
  libcusolver-13-2=12.2.0.11-1
  libcusparse-13-2=12.7.10.12-1
  libnvjitlink-13-2=13.2.86-1
  libnvinfer-dev=10.16.2.10-1+cuda13.2
  libnvinfer-plugin-dev=10.16.2.10-1+cuda13.2
  libnvonnxparsers-dev=10.16.2.10-1+cuda13.2
  python3-libnvinfer=10.16.2.10-1+cuda13.2
)
build_packages=(build-essential binutils cmake git ca-certificates python3-venv python3-dev)

sudo apt-get update
apt-cache policy nvidia-l4t-core cuda-nvcc-13-2 libnvinfer10 python3-libnvinfer
apt-get --simulate install --no-install-recommends \
  "${build_packages[@]}" "${compute_packages[@]}" | tee compute-apt-plan.txt

# Execute after checking that the plan resolves the pinned NVIDIA versions.
sudo apt-get install -y --no-install-recommends \
  "${build_packages[@]}" "${compute_packages[@]}"
dpkg-query -W -f='${binary:Package}\t${Version}\n' > compute-installed-packages.tsv
```

The explicit pins cover the top-level NVIDIA packages; apt resolves their dependencies. Save the simulation and installed-package list to capture the actual Ubuntu and transitive versions. This is a reproducible component selection, not a claim that a moving apt repository is an immutable OS image.

The CUDA compiler pulls CRT, NVVM, PTX compiler, CCCL, runtime, and configuration packages. `libcurand-dev` supplies `curand_kernel.h`. NVRTC's development package supplies the JIT compiler headers/library. The core cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, and nvJitLink runtime libraries provide the conventional CUDA library environment for the pinned CuPy wheel and kernel builders; we retain these runtime libraries rather than depending on partial-import/lazy-loading behavior. Their large unrelated development archives are omitted. [CuPy 13.6 installation requirements](https://docs.cupy.dev/en/v13.6.0/install.html), [CuPy import graph](https://github.com/cupy/cupy/blob/v13.6.0/cupy/__init__.py)

TensorRT development packages provide the ordinary include paths and unversioned linker libraries, including ONNX parser support required by the project's CMake even when serving directly from a checkpoint. `python3-libnvinfer` pulls the matching TensorRT runtime, plugin, version-compatible plugin, and ONNX runtime packages. Avoid `python3-libnvinfer-dev`: its metapackage adds lean/dispatch development packages that this build does not need. Avoid `nvidia-jetpack`, `nvidia-cuda-dev`, and `nvidia-tensorrt-dev` metapackages for this narrow install. [Pinned CMake requirements](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/CMakeLists.txt#L134), [TensorRT finder](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cmake/FindTensorRT.cmake)

## Python environment, validation, and build

Use system Python 3.12 with `--system-site-packages` so the virtual environment can import NVIDIA's apt-installed TensorRT binding. Do not install a second TensorRT wheel over it.

```bash
cd external/TensorRT-Edge-LLM
test "$(git rev-parse HEAD)" = e8b29522938901f6df19ebeedd4b69bc8edbcd97
git submodule update --init --recursive
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'pybind11==3.0.4' \
  'nvidia-cutlass-dsl[cu13]==4.7.0' 'cupy-cuda13x==13.6.0' \
  'cuda-python>=12.6,<14'
export CUDA_PATH=/usr/local/cuda-13.2
export PATH="$CUDA_PATH/bin:$PATH"
export TRT_PACKAGE_DIR=/usr
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvcc --version
python - <<'PY'
import tensorrt as trt
import cupy as cp
print('TensorRT', trt.__version__, trt.__file__)
assert trt.__version__.startswith('10.16.2')
assert cp.cuda.Device(0).compute_capability == '87'
cp.show_config()
x = cp.arange(16, dtype=cp.float32)
assert float(x.sum().get()) == 120.0
cp.cuda.get_current_stream().synchronize()
print('CUDA allocation, compilation, and reduction passed')
PY

python kernelSrcs/build_cutedsl.py --gpu_arch sm_87 --arch aarch64 \
  --cuda-version 13.2 --kernels fmha --jobs 1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=13.2 \
  -DENABLE_CUTE_DSL=fmha -DBUILD_PYTHON_BINDINGS=ON \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build --parallel 1
python -m pip install -e '.[server,server-tools]'
python -m pip check
python -m pip freeze > compute-python-packages.txt
python -m experimental.server --help
```

The GPU probe allocates only a tiny array but still requires a booted, functioning NVIDIA driver. Missing compatible arm64 wheels, a failed import, failed SM87 kernel generation, or a native linker error remain real gates; retain logs and diagnose before launching the model. The pinned upstream script itself specifies CuPy 13.6.0 and CUTLASS DSL 4.7.0. The `server,server-tools` extras avoid the PyTorch/export/quantization stack. [Pinned kernel builder](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/kernelSrcs/build_cutedsl.py), [pinned Python extras](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/pyproject.toml)

Transformers 5.14.1 and Jinja2 3.1.6 from `server-tools` are needed to derive this checkpoint's processed chat template. A CPU-only test loaded the exact tokenizer and verified that the pinned helper reproduces the current UI's single-user prompt for thinking on/off, including model image delimiters, without Torch. See [tokenizer evidence](../results/cosmos3-tokenizer-preflight.json). Without these dependencies, upstream silently writes a generic template. `scripts/run_backend.sh` now rejects that fallback and stale generic Cosmos3 cache artifacts; use a fresh `COSMOS_CACHE_DIR` if an old cache is rejected. This is a tokenizer/template check, not model inference.

Then follow [backend launch and actual signs-of-life checks](../docs/backend-build.md#4-establish-actual-signs-of-life), using the already verified reasoner snapshot and `scripts/run_backend.sh`. CUDA compilation or `--help` success does not establish working model inference.

## Memory and storage limits

Package installed size measures **disk**, not resident RAM or VRAM. The signed metadata lists approximately 2.86 GiB for `libnvinfer10` and 3.25 GiB for `libnvinfer-dev` alone; their compressed downloads total approximately 4.12 GiB. Ordinary development archives make the initial build large on disk but do not consume that much RAM merely by being installed. A single compile worker helps bound transient build pressure. Prefer sufficient NVMe free space and preserve standard packages instead of patching upstream linker discovery to save static-archive disk space.

No cuDNN, TensorRT samples, VPI, DLA tools, Nsight desktop tools, camera desktop stack, NCCL, or quantization environment is added by the explicit recipe. Existing BSP/desktop services still require separate measured assessment; this package selection alone cannot promise a RAM saving. Record idle available memory after boot, then actual runtime peaks and latency. FP16 model fit on 8 GB shared memory and final performance remain unverified.
