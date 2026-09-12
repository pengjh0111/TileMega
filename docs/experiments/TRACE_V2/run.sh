#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-D1 evidence: one trace v2 dump per cell, the D1-a correctness gate and the
# D1-c perturbation gate.  The perturbation rounds alternate trace on and off
# inside a single session, because absolute latency is not comparable across
# sessions and only the paired ratio means anything.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-25}"
correctness="${CORRECTNESS_RUNS:-50}"
tick="$(awk -F'\t' '$1=="globaltimer_resolution_ns"{print $2}' "${here}/resolution.tsv")"
commit="$(git -C "${repo}" rev-parse HEAD)"
mkdir -p "${raw}/bin" "${raw}/log" "${raw}/final" "${raw}/dump"

common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)

for model in gqa2 mha4; do
  src="${repo}/docs/experiments/SEQSCAN/raw/src/${model}.cu"
  "${nvcc}" "${common[@]}" -DTILEMEGA_TRACE_V2=1 "${src}" "${link[@]}" \
    -o "${raw}/bin/${model}_on" 2> "${raw}/log/${model}_on.ptxas"
  "${nvcc}" "${common[@]}" "${src}" "${link[@]}" \
    -o "${raw}/bin/${model}_off" 2> "${raw}/log/${model}_off.ptxas"
done

# Placement is a compile-time switch and these builds carry none, so every
# cell here is TILEMEGA_PLACEMENT=0 as the coverage requires.
export TILEMEGA_COMMIT="${commit}"
export TILEMEGA_GLOBALTIMER_NS="${tick}"

# --- dumps --------------------------------------------------------------
for model in gqa2 mha4; do
  for seq in 4 128; do
    fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
    out="${raw}/dump/${model}_s${seq}"
    rm -rf "${out}"
    TILEMEGA_MODEL_NAME="${model}" TILEMEGA_TRACE_V2=1 TILEMEGA_TRACE_V2_OUT="${out}" \
      "${raw}/bin/${model}_on" "${fixture}" > "${raw}/log/dump_${model}_s${seq}.out" 2>&1
    grep -q '^RESULT status=PASS' "${raw}/log/dump_${model}_s${seq}.out" || {
      echo "dump ${model} seq=${seq}: not PASS" >&2; exit 1; }
    grep -h '^E2E_TRACE_V2' "${raw}/log/dump_${model}_s${seq}.out"
  done
done

# --- D1-a: correctness of the instrumented build ------------------------
for model in gqa2 mha4; do
  for seq in 4 128; do
    fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
    TILEMEGA_MODEL_NAME="${model}" TILEMEGA_TRACE_V2=1 \
      "${repo}/scripts/gpu_stat_run.sh" -n "${correctness}" -t 120 \
      -l "tracev2_${model}_s${seq}" -k \
      -o "${raw}/final/correct_${model}_s${seq}" -- \
      "${raw}/bin/${model}_on" "${fixture}" \
      | tee -a "${raw}/log/correctness.tsv"
  done
done

# --- D1-c: paired perturbation, trace on against trace off --------------
arms=(off on)
for model in gqa2 mha4; do
  for seq in 4 128; do
    fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
    for ((round=0; round<runs; ++round)); do
      for ((slot=0; slot<${#arms[@]}; ++slot)); do
        arm="${arms[$(((round + slot) % ${#arms[@]}))]}"
        # The on arm carries the env switch too, so it pays allocation, zeroing
        # and the device writes -- the whole cost a user would pay.
        env_on=()
        [[ "${arm}" == on ]] && env_on=(TILEMEGA_TRACE_V2=1)
        env TILEMEGA_MODEL_NAME="${model}" "${env_on[@]}" \
          "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 \
          -l "perturb_${model}_${seq}_${arm}" -k \
          -o "${raw}/final/perturb_${model}_s${seq}_${arm}/r${round}" -- \
          "${raw}/bin/${model}_${arm}" "${fixture}" \
          >> "${raw}/log/perturb_${model}_s${seq}_${arm}.runner" 2>&1
      done
    done
  done
done

python3 "${here}/analyze.py" "${raw}/dump"/* --out "${here}"
python3 "${here}/perturbation.py" "${raw}" --runs "${runs}" > "${raw}/perturbation.txt"
cat "${raw}/perturbation.txt"
echo PASS > "${raw}/status.txt"
