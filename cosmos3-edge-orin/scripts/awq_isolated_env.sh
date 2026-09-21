#!/usr/bin/env bash
# Source this file with the absolute, newly created auxiliary task directory.
# This configures library paths; it is not an OS filesystem sandbox.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  echo 'Source this file: source scripts/awq_isolated_env.sh /absolute/task-directory' >&2
  exit 2
fi
if [[ $# != 1 || $1 != /* || ! -d $1 || -L $1 ]]; then
  echo 'An existing absolute task directory, not a symlink, is required.' >&2
  return 2
fi
export COSMOS_AUX_DIR
COSMOS_AUX_DIR=$(cd -- "$1" && pwd -P) || return 2
if [[ $COSMOS_AUX_DIR == / || $COSMOS_AUX_DIR == "${HOME:-}" ]]; then
  echo 'Use the newly created task directory, never a filesystem or home root.' >&2
  return 2
fi
# Never change HOME. Redirect supported package caches explicitly.
export TMPDIR="$COSMOS_AUX_DIR/tmp" TMP="$COSMOS_AUX_DIR/tmp" TEMP="$COSMOS_AUX_DIR/tmp"
export XDG_CACHE_HOME="$COSMOS_AUX_DIR/cache"
export XDG_CONFIG_HOME="$COSMOS_AUX_DIR/config"
export XDG_DATA_HOME="$COSMOS_AUX_DIR/data"
export XDG_STATE_HOME="$COSMOS_AUX_DIR/state"
export XDG_RUNTIME_DIR="$COSMOS_AUX_DIR/run"
export PIP_CACHE_DIR="$COSMOS_AUX_DIR/cache/pip"
export PIP_CONFIG_FILE=/dev/null
export HF_HOME="$COSMOS_AUX_DIR/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_DATASETS_CACHE="$HF_HOME/datasets" HF_MODULES_CACHE="$HF_HOME/modules"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE" TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_TOKEN_PATH="$HF_HOME/token"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$COSMOS_AUX_DIR/cache/torch"
export TORCH_EXTENSIONS_DIR="$COSMOS_AUX_DIR/cache/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$COSMOS_AUX_DIR/cache/torchinductor"
export TRITON_CACHE_DIR="$COSMOS_AUX_DIR/cache/triton"
export CUDA_CACHE_PATH="$COSMOS_AUX_DIR/cache/cuda"
export NUMBA_CACHE_DIR="$COSMOS_AUX_DIR/cache/numba"
export MPLCONFIGDIR="$COSMOS_AUX_DIR/config/matplotlib"
export PYTHONPYCACHEPREFIX="$COSMOS_AUX_DIR/cache/pycache"
export PYTHONNOUSERSITE=1 PYTHONUSERBASE="$COSMOS_AUX_DIR/python-user"
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL="$COSMOS_AUX_DIR/config/gitconfig"
export GIT_TERMINAL_PROMPT=0
export WANDB_DISABLED=true WANDB_DIR="$COSMOS_AUX_DIR/cache/wandb"
export WANDB_CONFIG_DIR="$COSMOS_AUX_DIR/config/wandb"
export WANDB_CACHE_DIR="$COSMOS_AUX_DIR/cache/wandb"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN PYTHONPATH
mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" \
  "$XDG_STATE_HOME" "$XDG_RUNTIME_DIR" "$HF_HOME" "$TORCH_HOME" \
  "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX" \
  "$PIP_CACHE_DIR" "$PYTHONUSERBASE" || return 2
touch "$GIT_CONFIG_GLOBAL" || return 2
chmod 700 "$XDG_RUNTIME_DIR" || return 2
cd -- "$COSMOS_AUX_DIR" || return 2
