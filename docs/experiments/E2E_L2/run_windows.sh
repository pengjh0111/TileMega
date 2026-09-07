#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Value of exact task dependency windows versus forcing every incoming edge to
# wait for every producer task. The switch is host-side, so both arms use the
# exact same executable.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw_taskqueue_windows"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-25}"
mkdir -p "${raw}/bin" "${raw}/log" "${raw}/final"
common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)
arms=(exact all)

for model in gqa2 mha4; do
  src="${repo}/docs/experiments/SEQSCAN/raw/src/${model}.cu"
  "${nvcc}" "${common[@]}" "${src}" "${link[@]}" \
    -o "${raw}/bin/${model}" 2> "${raw}/log/${model}.ptxas"
  for seq in 4 128 512; do
    fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
    for ((round=0; round<runs; ++round)); do
      for ((slot=0; slot<2; ++slot)); do
        arm="${arms[$(((round + slot) % 2))]}"
        env_args=()
        [[ "${arm}" == all ]] && env_args=(TILEMEGA_FORCE_ALL_DEPENDENCIES=1)
        env "${env_args[@]}" "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 \
          -l "windows_${model}_${seq}_${arm}" -k \
          -o "${raw}/final/${model}_s${seq}_${arm}/r${round}" -- \
          "${raw}/bin/${model}" "${fixture}" \
          >> "${raw}/log/${model}_s${seq}_${arm}.runner" 2>&1
      done
    done
    for arm in "${arms[@]}"; do
      logs=("${raw}/final/${model}_s${seq}_${arm}"/r*/run_*.log)
      pass="$(grep -h -c '^RESULT status=PASS' "${logs[@]}" | awk '{n+=$1} END{print n+0}')"
      [[ "${pass}" == "${runs}" ]] || {
        echo "${model} seq=${seq} ${arm}: ${pass}/${runs}" >&2
        exit 1
      }
    done
  done
done

python3 "${here}/summarize_windows.py" "${raw}" > "${raw}/summary.txt"
cat "${raw}/summary.txt"
echo PASS > "${raw}/status.txt"
