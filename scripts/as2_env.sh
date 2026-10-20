#!/usr/bin/env bash
export AS2_ROOT=/home/d50064638/junhuip
export REPO="$AS2_ROOT/TileMega"
export AS1_WORK="$AS2_ROOT/tilemega-as1"
export AS2_WORK="$AS2_ROOT/tilemega-as2"
if [[ ! -f /usr/local/Ascend/cann-9.1.0/set_env.sh || ! -x "$AS1_WORK/envs/as1-npu/bin/python" ]]; then
    printf '%s\n' 'The verified AS-1 CANN/private Python environment is required' >&2
    return 1 2>/dev/null || exit 1
fi
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export VIRTUAL_ENV="$AS1_WORK/envs/as1-npu"
if [[ -x "$AS2_WORK/envs/as2/bin/python" ]]; then
    export VIRTUAL_ENV="$AS2_WORK/envs/as2"
fi
export PATH="$VIRTUAL_ENV/bin:$AS1_WORK/envs/host/bin:/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin:$PATH"
export TMPDIR="$AS2_WORK/tmp"
export XDG_CACHE_HOME="$AS2_WORK/cache/xdg"
export CCACHE_DIR="$AS2_WORK/cache/ccache"
export PIP_CACHE_DIR="$AS2_WORK/cache/pip"
export TORCH_EXTENSIONS_DIR="$AS2_WORK/cache/torch_extensions"
export TRITON_CACHE_DIR="$AS2_WORK/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$AS2_WORK/cache/torchinductor"
export HF_HOME="$AS2_WORK/cache/huggingface"
export PYTHONPYCACHEPREFIX="$AS2_WORK/cache/pycache"
export ASCEND_CACHE_PATH="$AS2_WORK/cache/cann"
export ASCEND_WORK_PATH="$AS2_WORK/logs/cann"
export ASCEND_PROCESS_LOG_PATH="$AS2_WORK/logs/cann/plog"
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CCACHE_DIR" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$PYTHONPYCACHEPREFIX" \
  "$ASCEND_CACHE_PATH" "$ASCEND_WORK_PATH" "$ASCEND_PROCESS_LOG_PATH"
