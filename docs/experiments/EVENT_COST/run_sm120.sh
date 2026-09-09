#!/usr/bin/env bash
# A9 manual-only target run. No sm_120 measurement is claimed by this script.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${OUT_DIR:-${here}/calibration_sm120}"
mkdir -p "${out}"
trap 'printf "FAIL\n" > "${out}/status.txt"' ERR
command -v nvidia-smi >/dev/null
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${GPU_INDEX:-0}")"
if [[ "${cap}" != "12.0" ]]; then
  printf 'FAIL: requires sm_120, got %s\n' "${cap}" > "${out}/status.txt"
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}"
python3 "${here}/run_calibration.py" --out "${out}" --arch sm_120
# The reused runner rotates all arms/states within each round, records ptxas
# resources and exact queue counts, and requires 50 full-arm fresh processes
# per cell. An unsafe arm is never counted as correctness evidence.
