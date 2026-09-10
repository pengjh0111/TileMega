#!/usr/bin/env bash
set -euo pipefail
[[ "${TILEMEGA_TARGET_ARCH:-sm_120}" == sm_120 ]] || { echo 'requires sm_120'; exit 2; }
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-"$repo/docs/experiments/FUSION/sm120"}; mkdir -p "$out"
printf 'status=NOT_RUN\nreason=manual sm_120 execution required\n' > "$out/status.txt"
for edge in gemm_elementwise gemm_rmsnorm; do
  for arm in unfused fused; do
    echo "${edge} ${arm} model_prediction=not_run measured=not_run ctas_per_sm=not_run" >> "$out/manifest.tsv"
  done
done
echo 'script_only=1; outputs require a 5090 host' >> "$out/status.txt"
