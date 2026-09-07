#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Queue-driven L2 vs L1: both reference models, seq={4,128,512}, 25 fresh
# processes. L1 and L2 are timed by the same process, so observations pair.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw_taskqueue"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-25}"
# The final queue-era κ sweep selects 1 on both reference models. Override for a
# diagnostic, but make the primary L2/L1 result use the measured winner.
kappa="${KAPPA:-1}"
mkdir -p "${raw}/bin" "${raw}/log" "${raw}/final"

common=(-std=c++17 -O2 -arch=native -lineinfo
  "-DTILEMEGA_EVENT_KAPPA=${kappa}"
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)

for model in gqa2 mha4; do
  src="${repo}/docs/experiments/SEQSCAN/raw/src/${model}.cu"
  [[ -s "${src}" ]] || { echo "missing ${src}; run SEQSCAN/run.sh" >&2; exit 77; }
  "${nvcc}" "${common[@]}" "${src}" "${link[@]}" \
    -o "${raw}/bin/${model}" 2> "${raw}/log/${model}.ptxas"
  for seq in 4 128 512; do
    fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
    [[ -s "${fixture}/manifest.json" ]] || { echo "missing ${fixture}" >&2; exit 77; }
    out="${raw}/final/${model}_s${seq}"
    "${repo}/scripts/gpu_stat_run.sh" -n "${runs}" -t 120 \
      -l "e2e_l2_${model}_s${seq}" -k -o "${out}" -- \
      "${raw}/bin/${model}" "${fixture}" > "${raw}/log/${model}_s${seq}.runner"
    pass="$(grep -h -c '^RESULT status=PASS' "${out}"/run_*.log | awk '{n+=$1} END{print n+0}')"
    [[ "${pass}" == "${runs}" ]] || { echo "${model} seq=${seq}: ${pass}/${runs}" >&2; exit 1; }
  done
done

python3 "${here}/summarize.py" "${raw}" > "${raw}/summary.txt"
cat "${raw}/summary.txt"
echo PASS > "${raw}/status.txt"
