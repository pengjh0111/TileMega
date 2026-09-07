#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Fresh queue-era epoch control. This deliberately measures 32 serialized
# launches at seq=1; it does not claim to be growing-cache token generation.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw_taskqueue"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-50}"
mkdir -p "${raw}/bin" "${raw}/log" "${raw}/final"
common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)

printf 'model\tarm\tprocesses\tpasses\ttotal_mismatch\ttimeouts\n' > "${raw}/iterations.tsv"
for model in gqa2 mha4; do
  src="${repo}/docs/experiments/SEQSCAN/raw/src/${model}.cu"
  fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s1_p0"
  "${nvcc}" "${common[@]}" "${src}" "${link[@]}" \
    -o "${raw}/bin/${model}_monotone" 2> "${raw}/log/${model}_monotone.ptxas"
  "${nvcc}" "${common[@]}" -DTILEMEGA_NEGATIVE_RESET_EVENTS=1 \
    "${src}" "${link[@]}" -o "${raw}/bin/${model}_reset" \
    2> "${raw}/log/${model}_reset.ptxas"
  for arm in monotone reset; do
    out="${raw}/final/${model}_${arm}"
    TILEMEGA_ITERATIONS=32 "${repo}/scripts/gpu_stat_run.sh" -n "${runs}" \
      -t 120 -l "epoch_${model}_${arm}" -k -o "${out}" -- \
      "${raw}/bin/${model}_${arm}" "${fixture}" > "${raw}/log/${model}_${arm}.runner"
    pass="$(grep -h -c '^RESULT status=PASS' "${out}"/run_*.log | awk '{n+=$1} END{print n+0}')"
    mismatch="$(sed -n 's/^E2E_ITER_SWEEP .*total_mismatch=\([0-9]*\).*/\1/p' "${out}"/run_*.log | awk '{n+=$1} END{print n+0}')"
    timeout_count="$(find "${out}" -name '*.timeout' | wc -l)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${model}" "${arm}" "${runs}" \
      "${pass}" "${mismatch}" "${timeout_count}" | tee -a "${raw}/iterations.tsv"
  done
done
python3 "${here}/summarize.py" "${raw}" > "${raw}/timing.tsv"
cat "${raw}/timing.tsv"
reset_passes="$(awk -F '\t' '$2 == "reset" {n += $4} END {print n + 0}' \
  "${raw}/iterations.tsv")"
if [[ "${reset_passes}" -eq 0 ]]; then
  echo PASS_NEGATIVE_FAILED > "${raw}/status.txt"
else
  echo FAIL_NEGATIVE_DID_NOT_FAIL > "${raw}/status.txt"
fi
cat "${raw}/status.txt"
