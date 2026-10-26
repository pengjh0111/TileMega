#!/usr/bin/env bash
_as2_baseline="${1:-}"
case "$_as2_baseline" in
    vllm-ascend|sglang-npu) ;;
    *) printf '%s\n' 'Use vllm-ascend or sglang-npu' >&2; return 1 2>/dev/null || exit 1 ;;
esac
_as2_baseline_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$_as2_baseline_scripts/as2_cpu_env.sh" || return 1
export VIRTUAL_ENV="$AS2_WORK/envs/$_as2_baseline"
if [[ ! -x "$VIRTUAL_ENV/bin/python" ]]; then
    printf '%s\n' 'The private baseline environment has not been created' >&2
    return 1 2>/dev/null || exit 1
fi
export PATH="$VIRTUAL_ENV/bin:$PATH"
export TORCHAIR_CACHE_HOME="$AS2_WORK/cache/torchair/$_as2_baseline"
export VLLM_CACHE_ROOT="$AS2_WORK/cache/vllm"
export VLLM_CONFIG_ROOT="$AS2_WORK/cache/vllm_config"
export SGLANG_CACHE_DIR="$AS2_WORK/cache/sglang"
export SGLANG_TORCH_PROFILER_DIR="$AS2_WORK/logs/sglang_profiler"
export UV_CACHE_DIR="$AS2_WORK/cache/uv"
export CARGO_HOME="$AS2_WORK/cache/cargo"
_as2_python_headers="$AS2_WORK/third_party/python312-dev-3.12.3-1ubuntu0.17/usr/include"
if [[ -f "$_as2_python_headers/python3.12/Python.h" ]]; then
    export CPLUS_INCLUDE_PATH="$_as2_python_headers/python3.12:$_as2_python_headers${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export C_INCLUDE_PATH="$_as2_python_headers/python3.12:$_as2_python_headers${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
fi
if [[ "$_as2_baseline" == sglang-npu ]]; then
    _as2_vendor="$AS2_WORK/third_party/sgl-kernel-npu-2026.9.0.post10-private"
    _as2_custom="$_as2_vendor/CANN-custom_ops-none-linux.x86_64/packages/vendors/customize"
    _as2_transformer="$_as2_vendor/cann-ops-transformer-custom_linux-x86_64/packages/vendors/custom_transformer"
    if [[ -d "$_as2_custom" && -d "$_as2_transformer" ]]; then
        export ASCEND_CUSTOM_OPP_PATH="$_as2_custom:$_as2_transformer"
        export LD_LIBRARY_PATH="$_as2_custom/op_api/lib:$_as2_transformer/op_api/lib:$LD_LIBRARY_PATH"
    fi
    _as2_atb="$AS2_WORK/third_party/nnal-9.1.0-private/atb/set_env.sh"
    if [[ -f "$_as2_atb" ]]; then
        source "$_as2_atb" --cxx_abi=1
    fi
fi
mkdir -p "$TORCHAIR_CACHE_HOME" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" \
    "$SGLANG_CACHE_DIR" "$SGLANG_TORCH_PROFILER_DIR" "$UV_CACHE_DIR" "$CARGO_HOME"
unset _as2_baseline _as2_baseline_scripts _as2_python_headers _as2_vendor _as2_custom _as2_transformer _as2_atb
