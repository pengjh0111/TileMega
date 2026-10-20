#!/usr/bin/env bash
# Source explicitly; this file never changes shared shell or package configuration.
export AS1_ROOT=/home/d50064638/junhuip
export REPO="$AS1_ROOT/TileMega"
export AS1_WORK="$AS1_ROOT/tilemega-as1"
if [[ ! -f /usr/local/Ascend/cann-9.1.0/set_env.sh ]]; then
    printf '%s\n' 'Expected inventoried CANN 9.1.0 environment is missing' >&2
    return 1 2>/dev/null || exit 1
fi
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH="/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin:$PATH"
export TMPDIR="$AS1_WORK/tmp"
export XDG_CACHE_HOME="$AS1_WORK/cache/xdg"
export CCACHE_DIR="$AS1_WORK/cache/ccache"
export PIP_CACHE_DIR="$AS1_WORK/cache/pip"
export TORCH_EXTENSIONS_DIR="$AS1_WORK/cache/torch_extensions"
export TRITON_CACHE_DIR="$AS1_WORK/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$AS1_WORK/cache/torchinductor"
export HF_HOME="$AS1_WORK/cache/huggingface"
export PYTHONPYCACHEPREFIX="$AS1_WORK/cache/pycache"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# CANN environment-variable reference: envref_07_0006/0007, 9.1.
export ASCEND_CACHE_PATH="$AS1_WORK/cache/cann"
export ASCEND_WORK_PATH="$AS1_WORK/logs/cann"
export ASCEND_PROCESS_LOG_PATH="$AS1_WORK/logs/cann/plog"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CCACHE_DIR" "$PIP_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$HF_HOME" "$PYTHONPYCACHEPREFIX" "$ASCEND_CACHE_PATH" \
    "$ASCEND_WORK_PATH" "$ASCEND_PROCESS_LOG_PATH"
