#!/usr/bin/env bash
# Install the public R39.2.1 compute components recorded in research/jetpack-compute-install.md.
set -euo pipefail
mode="${1:-plan}"
receipt_dir="${2:-/home/jetson/cosmos-edge/results/compute-install}"
case "$mode" in plan|apply) ;; *) echo 'Usage: install_jetpack_compute.sh [plan|apply] [absolute-receipt-directory]' >&2; exit 2;; esac
[[ $EUID == 0 ]] || { echo 'Run as root through sudo.' >&2; exit 2; }
[[ "$receipt_dir" == /* ]] || { echo 'Receipt directory must be absolute.' >&2; exit 2; }
[[ "$(dpkg --print-architecture)" == arm64 ]]
. /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]]
[[ "$(dpkg-query -W -f='${Version}' nvidia-l4t-core)" == 39.2.1-* ]]
tr '\000' '\n' < /proc/device-tree/model | grep -qi 'Jetson.*Orin'
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
apt_options=(-o Acquire::Retries=2)
if [[ -n "${COSMOS_PACKAGE_PROXY:-}" ]]; then
  # Recovery can have only IPv6 outside loopback. APT's AI_ADDRCONFIG
  # otherwise rejects a usable IPv4 loopback proxy before connecting.
  apt_options+=(-o "Acquire::http::Proxy=$COSMOS_PACKAGE_PROXY" -o "Acquire::https::Proxy=$COSMOS_PACKAGE_PROXY"
    -o Acquire::Connect::AddrConfig=false)
fi
mkdir -p "$receipt_dir"
if [[ "$mode" == plan ]]; then
  rm -f "$receipt_dir/plan.txt"
  apt-get "${apt_options[@]}" -o APT::Update::Error-Mode=any update 2>&1 | tee "$receipt_dir/update.log"
  apt-cache policy nvidia-l4t-core cuda-nvcc-13-2 libnvinfer10 python3-libnvinfer | tee "$receipt_dir/policy.txt"
  apt-get "${apt_options[@]}" --simulate install --no-install-recommends "${build_packages[@]}" "${compute_packages[@]}" 2>&1 | tee "$receipt_dir/plan.pending.txt"
  mv "$receipt_dir/plan.pending.txt" "$receipt_dir/plan.txt"
else
  [[ -s "$receipt_dir/plan.txt" ]] || { echo 'Run and review plan first.' >&2; exit 2; }
  DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" install -y --no-install-recommends "${build_packages[@]}" "${compute_packages[@]}" 2>&1 | tee "$receipt_dir/install.log"
  dpkg-query -W -f='${binary:Package}\t${Version}\n' > "$receipt_dir/installed-packages.tsv"
  dpkg --audit | tee "$receipt_dir/dpkg-audit.txt"
  for specification in "${compute_packages[@]}"; do
    package="${specification%%=*}"
    expected="${specification#*=}"
    actual="$(dpkg-query -W -f='${Version}' "$package")"
    [[ "$actual" == "$expected" ]] || { printf '%s version mismatch: %s\n' "$package" "$actual" >&2; exit 1; }
  done
  /usr/local/cuda-13.2/bin/nvcc --version | tee "$receipt_dir/nvcc.txt"
  python3 -c 'import tensorrt; assert tensorrt.__version__.startswith("10.16.2"); print(tensorrt.__version__, tensorrt.__file__)' | tee "$receipt_dir/tensorrt.txt"
fi
