#!/usr/bin/env bash
# CPU preparation must not auto-load the installed NPU device backend.
_as2_cpu_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$_as2_cpu_script_dir/as2_env.sh" || return 1
unset _as2_cpu_script_dir
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
