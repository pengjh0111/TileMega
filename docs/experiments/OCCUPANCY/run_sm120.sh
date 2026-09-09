#!/usr/bin/env bash
# T0 manual-only experiment. Run on the target; never treat compilation as a result.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
out="${OUT_DIR:-${here}/raw_sm120}"
mkdir -p "${out}"
trap 'printf "FAIL\n" > "${out}/status.txt"' ERR
command -v nvidia-smi >/dev/null
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${GPU_INDEX:-0}")"
[[ "${cap}" == "12.0" ]] || { echo "requires sm_120, got ${cap}" >&2; printf 'FAIL: wrong device\n' > "${out}/status.txt"; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}"
export TILEMEGA_WARMUP="${TILEMEGA_WARMUP:-5}"
export TILEMEGA_REPEAT="${TILEMEGA_REPEAT:-11}"
python3 "${repo}/docs/experiments/L2_ATTRIB/run_t1.py" \
  --out "${out}" --arch sm_120 --variants occ1,occ2 \
  --seqs 4,128 --correctness-runs 50 --runs 25 --phases build,correctness,attrib
python3 "${repo}/docs/experiments/L2_ATTRIB/summarize_t1.py" \
  "${out}/attrib.tsv" --baseline occ1 --out "${out}/report"
# run_t1 evaluates F-40 using runtime block/warp and TargetSpec budgets. No
# 4090 register threshold is copied into this script. Each row carries spill
# bytes, registers, occupancy-smem, task-smem, grid and the binding resource.
printf 'PASS\n' > "${out}/status.txt"
