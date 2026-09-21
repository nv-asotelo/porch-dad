#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$project_dir/external"

checkout() {
    local name=$1 url=$2 commit=$3
    local target="$project_dir/external/$name"
    if [[ ! -e "$target" ]]; then
        git -c credential.helper= clone --depth 1 "$url" "$target"
    fi
    [[ "$(git -C "$target" remote get-url origin)" == "$url" ]] || {
        printf 'Unexpected upstream for %s\n' "$name" >&2
        exit 1
    }
    [[ -z "$(git -C "$target" status --porcelain)" ]] || {
        printf 'Preserving local changes in %s; resolve before fetching.\n' "$target" >&2
        exit 1
    }
    git -c credential.helper= -C "$target" fetch --depth 1 origin "$commit"
    git -C "$target" checkout --detach "$commit"
}

checkout TensorRT-Edge-LLM https://github.com/NVIDIA/TensorRT-Edge-LLM.git e8b29522938901f6df19ebeedd4b69bc8edbcd97
checkout live-vlm-webui https://github.com/NVIDIA-AI-IOT/live-vlm-webui.git 2fd5ba0b334c334d24bf0f9439d8742b243d22be
git -c credential.helper= -C "$project_dir/external/TensorRT-Edge-LLM" submodule update --init
