#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-D2 online arm: the same four-arm decomposition as L2_ATTRIB, run twice over
# the stage-major placement (mode 0) and the cross-stage rotated one (mode 5).
# The eight arm x placement combinations rotate by (round + slot) % 8 so the two
# placements interleave in time; running one placement to completion and then
# the other would confound the comparison with whatever the machine was doing.
# Unsafe arms stay timing probes and their numerical result is deliberately not
# an acceptance condition.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-25}"
correctness="${CORRECTNESS_RUNS:-50}"
mkdir -p "${raw}/bin" "${raw}/log" "${raw}/final"

common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)
arms=(neither nowait full l1nosync)
places=(0 5)
flags() {
  case "$1" in
    neither) echo '-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1 -DTILEMEGA_UNSAFE_NO_EVENT_NOTIFY=1' ;;
    nowait) echo '-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1' ;;
    full) echo '' ;;
    l1nosync) echo '-DTILEMEGA_UNSAFE_NO_GRID_SYNC=1' ;;
  esac
}

combos=()
for arm in "${arms[@]}"; do
  for place in "${places[@]}"; do combos+=("${arm}:${place}"); done
done

for model in gqa2 mha4; do
  src="${repo}/docs/experiments/SEQSCAN/raw/src/${model}.cu"
  for combo in "${combos[@]}"; do
    arm="${combo%%:*}"; place="${combo##*:}"
    # shellcheck disable=SC2206
    extra=($(flags "${arm}"))
    "${nvcc}" "${common[@]}" -DTILEMEGA_PLACEMENT="${place}" "${extra[@]}" \
      "${src}" "${link[@]}" -o "${raw}/bin/${model}_${arm}_p${place}" \
      2> "${raw}/log/${model}_${arm}_p${place}.ptxas"
  done
done

# --- the placement statistics line, once per binary that is numerically valid
: > "${raw}/place_stats.txt"
for model in gqa2 mha4; do
  for place in "${places[@]}"; do
    for seq in 4 128; do
      fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
      TILEMEGA_PLACEMENT_BASE_DUMP=1 "${raw}/bin/${model}_full_p${place}" "${fixture}" \
        > "${raw}/log/stats_${model}_s${seq}_p${place}.out" 2>&1
      printf '%s\tseq=%s\t' "${model}" "${seq}" >> "${raw}/place_stats.txt"
      grep -h '^E2E_PLACE_STATS' "${raw}/log/stats_${model}_s${seq}_p${place}.out" \
        >> "${raw}/place_stats.txt"
    done
  done
done
cat "${raw}/place_stats.txt"

# --- correctness, per placement -----------------------------------------
for place in "${places[@]}"; do
  for model in gqa2 mha4; do
    for seq in 4 128; do
      fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
      "${repo}/scripts/gpu_stat_run.sh" -n "${correctness}" -t 120 \
        -l "rotate_p${place}_${model}_s${seq}" -k \
        -o "${raw}/final/correct_p${place}_${model}_s${seq}" -- \
        "${raw}/bin/${model}_full_p${place}" "${fixture}" \
        | tee -a "${raw}/correctness.tsv"
    done
  done
done

# --- the paired timing rounds -------------------------------------------
for model in gqa2 mha4; do
  for seq in 4 128; do
    fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
    for ((round=0; round<runs; ++round)); do
      for ((slot=0; slot<${#combos[@]}; ++slot)); do
        combo="${combos[$(((round + slot) % ${#combos[@]}))]}"
        arm="${combo%%:*}"; place="${combo##*:}"
        "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 \
          -l "rotate_${model}_${seq}_${arm}_p${place}" -k \
          -o "${raw}/final/${model}_s${seq}_${arm}_p${place}/r${round}" -- \
          "${raw}/bin/${model}_${arm}_p${place}" "${fixture}" \
          >> "${raw}/log/${model}_s${seq}_${arm}_p${place}.runner" 2>&1 \
          || [[ "${arm}" != full ]]
      done
    done
    for combo in "${combos[@]}"; do
      arm="${combo%%:*}"; place="${combo##*:}"
      logs=("${raw}/final/${model}_s${seq}_${arm}_p${place}"/r*/run_*.log)
      samples="$(grep -h -c '^E2E_TIME' "${logs[@]}" | awk '{n+=$1} END{print n+0}')"
      [[ "${samples}" == "${runs}" ]] || {
        echo "${model} seq=${seq} ${arm} p${place}: ${samples}/${runs} timing samples" >&2
        exit 1
      }
      if [[ "${arm}" == full ]]; then
        pass="$(grep -h -c '^RESULT status=PASS' "${logs[@]}" | awk '{n+=$1} END{print n+0}')"
        [[ "${pass}" == "${runs}" ]] || {
          echo "${model} seq=${seq} full p${place}: ${pass}/${runs} correct" >&2
          exit 1
        }
      fi
    done
  done
done

python3 "${here}/summarize.py" "${raw}" > "${raw}/summary.tsv"
cat "${raw}/summary.tsv"
python3 "${here}/fork.py" "${raw}/summary.tsv" | tee "${raw}/fork.txt"
echo PASS > "${raw}/status.txt"
