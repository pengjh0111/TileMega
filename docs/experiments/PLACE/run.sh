#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Queue-era Place comparison: generated critical-path order versus the numeric
# round-robin/topological baseline. Both arms are one executable with a
# host-side policy switch, eliminating code-generation differences.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw_taskqueue"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-25}"
mkdir -p "${raw}/bin" "${raw}/log" "${raw}/final"

ctest --test-dir "${build}" -R '^list_scheduler$' -V > "${raw}/oracle.txt"
grep 'PLACE_ORACLE' "${raw}/oracle.txt"

common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)
arms=(critical_path round_robin)

for model in gqa2 mha4; do
  src="${repo}/docs/experiments/SEQSCAN/raw/src/${model}.cu"
  fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s128_p3"
  "${nvcc}" "${common[@]}" "${src}" "${link[@]}" \
    -o "${raw}/bin/${model}" 2> "${raw}/log/${model}.ptxas"
  for ((round=0; round<runs; ++round)); do
    for ((slot=0; slot<2; ++slot)); do
      arm="${arms[$(((round + slot) % 2))]}"
      TILEMEGA_SCHEDULE_POLICY="${arm}" \
        "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 \
        -l "place_${model}_${arm}" -k \
        -o "${raw}/final/${model}_${arm}/r${round}" -- \
        "${raw}/bin/${model}" "${fixture}" \
        >> "${raw}/log/${model}_${arm}.runner" 2>&1
    done
  done
done

python3 "${here}/summarize_place.py" "${raw}" > "${raw}/summary.txt"
cat "${raw}/summary.txt"
echo PASS > "${raw}/status.txt"
