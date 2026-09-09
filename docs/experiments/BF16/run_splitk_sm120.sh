#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${OUT_DIR:-${here}/raw_splitk_sm120}"
mkdir -p "${out}"
trap 'printf "FAIL\n" > "${out}/status.txt"' ERR
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${GPU_INDEX:-0}")"
[[ "${cap}" == "12.0" ]] || { echo "requires sm_120, got ${cap}" >&2; printf 'FAIL: wrong device\n' > "${out}/status.txt"; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}"
python3 "${here}/run_splitk.py" --out "${out}" --arch sm_120 --runs 50 --jobs "${JOBS:-4}"
