#!/usr/bin/env bash
set -euo pipefail

# Run on an 80 GB device. Dry construction and the memory report require no GPU.
# User scope for DM-1 excludes performance measurements.
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_root"
export PYTHONPATH="$task_root/python${PYTHONPATH:+:$PYTHONPATH}"
export TILEMEGA_GPU_LOCK="${TILEMEGA_GPU_LOCK:-/root/r14_work/gpu.lock}"
export TILEMEGA_CACHE_DIR="${TILEMEGA_CACHE_DIR:-$HOME/.cache/tilemega-dm}"
task_python="${TILEMEGA_PYTHON:-python3}"
task_models="${TILEMEGA_MODEL_ROOT:-/root/models}"
task_checkpoint="${TILEMEGA_MOE_CHECKPOINT:-$task_models/qwen3_30b_a3b}"
task_output="${TILEMEGA_MOE_OUT:-runs/dm1-moe-full}"

if [[ ! -f "$task_checkpoint/model.safetensors.index.json" ]]; then
    "$task_python" -m tilemega.serving.download_models --model qwen3_moe \
        --model-root "$task_models" --report "$task_output/download.json"
    if [[ ! -f "$task_checkpoint/model.safetensors.index.json" ]]; then
        printf '%s\n' "Checkpoint index missing: $task_checkpoint" >&2
        exit 1
    fi
fi
task_target="${TILEMEGA_TARGET:-auto}"
if [[ "${TILEMEGA_DRY_ONLY:-0}" == 1 && -z "${TILEMEGA_TARGET:-}" ]]; then
    task_target="$task_root/configs/targets/sm_89.json"
fi
task_args=(--checkpoint "$task_checkpoint" --out "$task_output" --target "$task_target" "$@")
"$task_python" -m tilemega.moe.full dry-build "${task_args[@]}"
if [[ "${TILEMEGA_DRY_ONLY:-0}" == 1 ]]; then
    exit 0
fi
"$task_python" -m tilemega.moe.full preflight "${task_args[@]}"
"$task_python" -m tilemega.moe.full build "${task_args[@]}"
"$task_python" -m tilemega.moe.full check "${task_args[@]}" --hf-check
