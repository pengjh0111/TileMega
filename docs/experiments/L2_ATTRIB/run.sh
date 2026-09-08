#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Queue-era four-arm attribution. Unsafe arms are timing probes and their
# numerical result is deliberately not an acceptance condition.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
# T1 controls regenerate sources: archived generated files contain their own
# WAIT macro and therefore cannot be used to test the new polling switch.
if [[ -n "${T1_VARIANTS:-}" ]]; then
  t1_out="${OUT_DIR:-${here}/raw_t1_repro}"
  python3 "${here}/run_t1.py" --out "${t1_out}" \
    --variants "${T1_VARIANTS}" --runs "${RUNS:-25}" \
    --correctness-runs "${CORRECTNESS_RUNS:-50}" \
    --seqs "${SEQS:-4,128}" --kappa "${KAPPA:-1}" \
    --phases "${PHASES:-build,correctness,attrib}"
  python3 "${here}/summarize_t1.py" "${t1_out}/attrib.tsv" \
    --baseline "${T1_BASELINE:-base}" --expected-rounds "${RUNS:-25}" \
    --out "${t1_out}/report"
  exit 0
fi
raw="${here}/raw_taskqueue"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-25}"
mkdir -p "${raw}/bin" "${raw}/log" "${raw}/final"

common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)
arms=(neither nowait full l1nosync)
flags() {
  case "$1" in
    neither) echo '-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1 -DTILEMEGA_UNSAFE_NO_EVENT_NOTIFY=1' ;;
    nowait) echo '-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1' ;;
    full) echo '' ;;
    l1nosync) echo '-DTILEMEGA_UNSAFE_NO_GRID_SYNC=1' ;;
  esac
}

for model in gqa2 mha4; do
  src="${repo}/docs/experiments/SEQSCAN/raw/src/${model}.cu"
  for arm in "${arms[@]}"; do
    # shellcheck disable=SC2206
    extra=($(flags "${arm}"))
    "${nvcc}" "${common[@]}" "${extra[@]}" "${src}" "${link[@]}" \
      -o "${raw}/bin/${model}_${arm}" 2> "${raw}/log/${model}_${arm}.ptxas"
  done
  for seq in 4 128; do
    fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
    for ((round=0; round<runs; ++round)); do
      for ((slot=0; slot<${#arms[@]}; ++slot)); do
        arm="${arms[$(((round + slot) % ${#arms[@]}))]}"
        "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 \
          -l "attrib_${model}_${seq}_${arm}" -k \
          -o "${raw}/final/${model}_s${seq}_${arm}/r${round}" -- \
          "${raw}/bin/${model}_${arm}" "${fixture}" \
          >> "${raw}/log/${model}_s${seq}_${arm}.runner" 2>&1 \
          || [[ "${arm}" != full ]]
      done
    done
    for arm in "${arms[@]}"; do
      logs=("${raw}/final/${model}_s${seq}_${arm}"/r*/run_*.log)
      samples="$(grep -h -c '^E2E_TIME' "${logs[@]}" | awk '{n+=$1} END{print n+0}')"
      [[ "${samples}" == "${runs}" ]] || {
        echo "${model} seq=${seq} ${arm}: ${samples}/${runs} timing samples" >&2
        exit 1
      }
      if [[ "${arm}" == full ]]; then
        pass="$(grep -h -c '^RESULT status=PASS' "${logs[@]}" | awk '{n+=$1} END{print n+0}')"
        [[ "${pass}" == "${runs}" ]] || {
          echo "${model} seq=${seq} full: ${pass}/${runs} correct" >&2
          exit 1
        }
      fi
    done
  done
done

python3 "${here}/summarize.py" "${raw}" > "${raw}/summary.txt"
cat "${raw}/summary.txt"
echo PASS > "${raw}/status.txt"
