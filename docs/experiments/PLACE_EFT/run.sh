#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-S2 gates S2-a..S2-e.
#
# Six arms per cell: the three closed forms the host already had (modes 0, 4
# and 5, selected by TILEMEGA_PLACEMENT on the committed control sources) and
# the three this round adds (eft, band, wavefront, carried in the generated
# source by `tilemega-place-eft` and compiled with TILEMEGA_PLACEMENT=0 so that
# the CTA-to-SM map is the legacy one and the Plan is the only difference).
#
# Two properties this script exists to protect:
#
#   * the control arms are built from the committed sources round one measured,
#     not from anything regenerated here, so H6's "modes 0 and 5 re-measured as
#     control arms" is a re-measurement of the same binary recipe;
#   * the six arms rotate by (round + slot) % 6, so no arm owns a fixed position
#     in time.  Running one arm to completion and then the next would confound
#     the comparison with clocks, thermals and whatever else drifts.
#
# The provenance diff is the first thing that runs: `tilemega-place-eft` emits
# `legacy_grid_stride` through its own import and codegen path, and if that
# source is byte-identical to the committed control then the eft/band/wavefront
# sources differ from the control only in the Plan.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${RAW_DIR:-${here}/raw}"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-25}"
correctness="${CORRECTNESS_RUNS:-50}"
mkdir -p "${raw}/bin" "${raw}/log" "${raw}/final" "${raw}/plan"

common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)
control="${repo}/docs/experiments/PLAN_CONTRACT/legacy_identity/plan"
arms=(legacy_grid_stride balanced rotate eft band wavefront)

# arm -> (source, TILEMEGA_PLACEMENT) for one cell.
arm_source() {
  local arm="$1" model="$2" seq="$3"
  case "${arm}" in
    legacy_grid_stride|rotate) echo "${control}/${model}.cu" ;;
    balanced) echo "${control}/${model}_balanced.cu" ;;
    *) echo "${raw}/plan/${model}_s${seq}_${arm}.cu" ;;
  esac
}
arm_macro() {
  case "$1" in legacy_grid_stride) echo 0 ;; balanced) echo 4 ;; rotate) echo 5 ;;
    *) echo 0 ;; esac
}

if [[ "${SKIP_GENERATE:-0}" != 1 ]]; then
  "${build}/tools/tilemega-place-eft" "${repo}" "${raw}/manifest.tsv" "${raw}" \
    2> "${raw}/log/place_eft.log"
fi

# --- provenance: the driver's legacy source against the committed control
: > "${raw}/provenance.tsv"
printf 'model\tseq\tbytes\tidentical\n' >> "${raw}/provenance.tsv"
for seq in 4 128; do
  for model in gqa2 mha4; do
    emitted="${raw}/plan/${model}_s${seq}_legacy_grid_stride.cu"
    if diff -q "${control}/${model}.cu" "${emitted}" > /dev/null; then same=1; else same=0; fi
    printf '%s\t%s\t%s\t%s\n' "${model}" "${seq}" "$(stat -c%s "${emitted}")" "${same}" \
      >> "${raw}/provenance.tsv"
    [[ "${same}" == 1 ]] || {
      echo "FAIL: ${model} s${seq} legacy source differs from the control" >&2; exit 1; }
  done
done
cat "${raw}/provenance.tsv"

# --- compile every arm of every reference cell
for seq in 4 128; do
  for model in gqa2 mha4; do
    for arm in "${arms[@]}"; do
      src="$(arm_source "${arm}" "${model}" "${seq}")"
      "${nvcc}" "${common[@]}" -DTILEMEGA_PLACEMENT="$(arm_macro "${arm}")" \
        "${src}" "${link[@]}" -o "${raw}/bin/${model}_s${seq}_${arm}" \
        2> "${raw}/log/nvcc_${model}_s${seq}_${arm}.log"
    done
  done
done

# --- the placement statistics line each arm reports about its own queues
: > "${raw}/place_stats.txt"
for seq in 4 128; do
  for model in gqa2 mha4; do
    fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
    for arm in "${arms[@]}"; do
      "${raw}/bin/${model}_s${seq}_${arm}" "${fixture}" \
        > "${raw}/log/stats_${model}_s${seq}_${arm}.out" 2>&1
      printf '%s\tseq=%s\t%s\t' "${model}" "${seq}" "${arm}" >> "${raw}/place_stats.txt"
      grep -h '^E2E_PLACE_STATS' "${raw}/log/stats_${model}_s${seq}_${arm}.out" \
        >> "${raw}/place_stats.txt"
    done
  done
done
cat "${raw}/place_stats.txt"

# --- S2-a: 50 fresh processes per arm
printf 'model\tseq\tarm\tpasses\tprocesses\n' > "${raw}/correctness.tsv"
for seq in 4 128; do
  for model in gqa2 mha4; do
    fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
    for arm in "${arms[@]}"; do
      "${repo}/scripts/gpu_stat_run.sh" -n "${correctness}" -t 120 \
        -l "eft_${model}_s${seq}_${arm}" -k \
        -o "${raw}/final/correct_${model}_s${seq}_${arm}" -- \
        "${raw}/bin/${model}_s${seq}_${arm}" "${fixture}" \
        > "${raw}/log/correct_${model}_s${seq}_${arm}.runner" 2>&1
      pass="$(grep -rh -c '^RESULT status=PASS' \
        "${raw}/final/correct_${model}_s${seq}_${arm}" | awk '{n+=$1} END{print n+0}')"
      printf '%s\t%s\t%s\t%s\t%s\n' "${model}" "${seq}" "${arm}" "${pass}" \
        "${correctness}" >> "${raw}/correctness.tsv"
    done
  done
done
cat "${raw}/correctness.tsv"

# --- S2-b: paired rounds, rotating arm order
for seq in 4 128; do
  for model in gqa2 mha4; do
    fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
    for ((round=0; round<runs; ++round)); do
      for ((slot=0; slot<${#arms[@]}; ++slot)); do
        arm="${arms[$(((round + slot) % ${#arms[@]}))]}"
        "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 \
          -l "eft_${model}_${seq}_${arm}" -k \
          -o "${raw}/final/${model}_s${seq}_${arm}/r${round}" -- \
          "${raw}/bin/${model}_s${seq}_${arm}" "${fixture}" \
          >> "${raw}/log/${model}_s${seq}_${arm}.runner" 2>&1
      done
    done
    for arm in "${arms[@]}"; do
      logs=("${raw}/final/${model}_s${seq}_${arm}"/r*/run_*.log)
      samples="$(grep -h -c '^E2E_TIME' "${logs[@]}" | awk '{n+=$1} END{print n+0}')"
      pass="$(grep -h -c '^RESULT status=PASS' "${logs[@]}" | awk '{n+=$1} END{print n+0}')"
      [[ "${samples}" == "${runs}" && "${pass}" == "${runs}" ]] || {
        echo "${model} s${seq} ${arm}: ${samples}/${runs} timed, ${pass}/${runs} correct" >&2
        exit 1; }
    done
  done
done

# --- S2-d: the four-arm sync decomposition, on mode 5 and on the chosen plan.
# The unsafe arms are timing probes; their numerical answer is not an acceptance
# condition, which is why only `full` is checked for PASS above.
decomp=(neither nowait full l1nosync)
flags() {
  case "$1" in
    neither) echo '-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1 -DTILEMEGA_UNSAFE_NO_EVENT_NOTIFY=1' ;;
    nowait) echo '-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1' ;;
    full) echo '' ;;
    l1nosync) echo '-DTILEMEGA_UNSAFE_NO_GRID_SYNC=1' ;;
  esac
}
combos=()
for probe in "${decomp[@]}"; do
  for arm in rotate eft; do combos+=("${probe}:${arm}"); done
done
for seq in 4 128; do
  for model in gqa2 mha4; do
    for combo in "${combos[@]}"; do
      probe="${combo%%:*}"; arm="${combo##*:}"
      # shellcheck disable=SC2206
      extra=($(flags "${probe}"))
      "${nvcc}" "${common[@]}" -DTILEMEGA_PLACEMENT="$(arm_macro "${arm}")" "${extra[@]}" \
        "$(arm_source "${arm}" "${model}" "${seq}")" "${link[@]}" \
        -o "${raw}/bin/${model}_s${seq}_${arm}_${probe}" \
        2> "${raw}/log/nvcc_${model}_s${seq}_${arm}_${probe}.log"
    done
    fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
    for ((round=0; round<runs; ++round)); do
      for ((slot=0; slot<${#combos[@]}; ++slot)); do
        combo="${combos[$(((round + slot) % ${#combos[@]}))]}"
        probe="${combo%%:*}"; arm="${combo##*:}"
        "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 \
          -l "eftdec_${model}_${seq}_${arm}_${probe}" -k \
          -o "${raw}/final/dec_${model}_s${seq}_${arm}_${probe}/r${round}" -- \
          "${raw}/bin/${model}_s${seq}_${arm}_${probe}" "${fixture}" \
          >> "${raw}/log/dec_${model}_s${seq}_${arm}_${probe}.runner" 2>&1 \
          || [[ "${probe}" != full ]]
      done
    done
  done
done

# --- S2-c: the real-width cell, same rotation, its own fixture
realwidth="${REALWIDTH:-1}"
if [[ "${realwidth}" == 1 ]]; then
  for seq in 4 128; do
    work="${repo}/docs/experiments/REALMODEL/raw/work/r2sim_s${seq}"
    fixture="${work}/export/fixture"
    [[ -d "${fixture}" ]] || { echo "FAIL: no real-width fixture at ${fixture}" >&2; exit 1; }
    for arm in "${arms[@]}"; do
      case "${arm}" in
        legacy_grid_stride|rotate) src="${work}/model.cu" ;;
        balanced) continue ;;  # needs its own variant plan; not a gate arm
        *) src="${raw}/plan/real_s${seq}_${arm}.cu" ;;
      esac
      "${nvcc}" "${common[@]}" -DTILEMEGA_PLACEMENT="$(arm_macro "${arm}")" \
        "${src}" "${link[@]}" -o "${raw}/bin/real_s${seq}_${arm}" \
        2> "${raw}/log/nvcc_real_s${seq}_${arm}.log"
    done
    real_arms=(legacy_grid_stride rotate eft band wavefront)
    for arm in "${real_arms[@]}"; do
      "${repo}/scripts/gpu_stat_run.sh" -n "${correctness}" -t 300 \
        -l "eft_real_s${seq}_${arm}" -k \
        -o "${raw}/final/correct_real_s${seq}_${arm}" -- \
        "${raw}/bin/real_s${seq}_${arm}" "${fixture}" \
        > "${raw}/log/correct_real_s${seq}_${arm}.runner" 2>&1
      pass="$(grep -rh -c '^RESULT status=PASS' \
        "${raw}/final/correct_real_s${seq}_${arm}" | awk '{n+=$1} END{print n+0}')"
      printf 'real\t%s\t%s\t%s\t%s\n' "${seq}" "${arm}" "${pass}" "${correctness}" \
        >> "${raw}/correctness.tsv"
    done
    for ((round=0; round<runs; ++round)); do
      for ((slot=0; slot<${#real_arms[@]}; ++slot)); do
        arm="${real_arms[$(((round + slot) % ${#real_arms[@]}))]}"
        "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 300 \
          -l "eft_real_${seq}_${arm}" -k \
          -o "${raw}/final/real_s${seq}_${arm}/r${round}" -- \
          "${raw}/bin/real_s${seq}_${arm}" "${fixture}" \
          >> "${raw}/log/real_s${seq}_${arm}.runner" 2>&1
      done
    done
  done
fi

python3 "${here}/summarize.py" "${raw}" > "${raw}/summary.tsv"
cat "${raw}/summary.tsv"
echo PASS > "${raw}/status.txt"
