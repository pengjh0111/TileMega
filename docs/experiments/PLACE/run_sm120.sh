#!/usr/bin/env bash
set -euo pipefail
[[ "${TILEMEGA_TARGET_ARCH:-sm_120}" == sm_120 ]] || { echo 'requires sm_120'; exit 2; }
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-"$repo/docs/experiments/PLACE/sm120"}; mkdir -p "$out"
printf 'status=NOT_RUN\nreason=manual sm_120 execution required\n' > "$out/status.txt"
for mapping in stage_major affine_balanced; do
  echo "${mapping} model_prediction=not_run measured=not_run max_worker_span=not_run" >> "$out/manifest.tsv"
done
echo 'script_only=1; outputs require a 5090 host' >> "$out/status.txt"
