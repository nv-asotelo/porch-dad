#!/usr/bin/env bash
set -euo pipefail

printf 'Observed at: '
date -u '+%Y-%m-%dT%H:%M:%SZ'
printf 'Host: '
uname -srm

if [[ "$(uname -s)" != Linux ]]; then
  printf 'This inventory must run on the Jetson; current host is not Linux.\n' >&2
  exit 2
fi

for file in /proc/device-tree/model /proc/device-tree/compatible /etc/nv_tegra_release /etc/os-release; do
  if [[ -r "$file" ]]; then
    printf '\n%s\n' "$file"
    tr '\000' '\n' < "$file"
    printf '\n'
  fi
done

printf '\nJetPack, CUDA, and TensorRT packages\n'
if command -v dpkg-query >/dev/null 2>&1; then
  dpkg-query -W -f='${Package}\t${Version}\n' \
    'nvidia-jetpack*' 'nvidia-l4t-core' 'cuda-toolkit*' \
    'libnvinfer*' 'python3-libnvinfer*' 2>/dev/null || true
fi

printf '\nCUDA compiler\n'
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  /usr/local/cuda/bin/nvcc --version
else
  printf 'nvcc is unavailable\n'
fi

printf '\nMemory\n'
free -h
printf '\nStorage\n'
df -h / "${COSMOS_DATA_DIR:-.}"
printf '\nNetwork addresses\n'
if command -v ip >/dev/null 2>&1; then
  ip -brief address
fi
printf '\nPower mode (query only)\n'
if command -v nvpmodel >/dev/null 2>&1; then
  nvpmodel -q 2>&1 || true
fi

if command -v timeout >/dev/null 2>&1 && command -v tegrastats >/dev/null 2>&1; then
  printf '\nTwo-second tegrastats sample\n'
  timeout 2s tegrastats --interval 500 || [[ "$?" == 124 ]]
fi
