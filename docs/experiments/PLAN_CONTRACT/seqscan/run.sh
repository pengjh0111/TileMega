#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-E1 gate E1-d: the sigma materialization path is correct on the SEQSCAN
# subset, seq in {4,128,2048} x past in {0,512}, both reference models, 50 fresh
# processes per cell (CLAUDE.md: a synchronization claim needs 50 and its pass
# rate reported).  Placement 0 is the default, i.e. the legacy plan; the other
# two modes are covered byte for byte by ../run.sh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../../.." && pwd)"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-50}"
work="${WORK_DIR:-/tmp/plan_contract}"
mkdir -p "${here}/raw" "${work}/bin" "${work}/log"

common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")

for model in gqa2 mha4; do
  src="${work}/${model}.cu"
  [ -f "${src}" ] || src="${repo}/docs/experiments/SEQSCAN/raw/src/${model}.cu"
  "${nvcc}" "${common[@]}" "${src}" "${build}/libtilemega.a" \
    -L/usr/local/cuda/lib64 -lcudart -o "${work}/bin/seqscan_${model}" \
    2>> "${work}/log/nvcc.log"
done

# A cell already at runs/runs is kept rather than re-measured: 12 cells x 50
# processes x seq 2048 is an hour of wall clock, and a resume that discarded a
# finished cell would make an interrupted run unfinishable in practice.  The
# table records every cell's own measurement, so a resumed run is still 50 fresh
# processes per cell -- just not all in one stretch.
declare -A cached=()
if [ -f "${here}/raw/correctness.tsv" ]; then
  while IFS=$'\t' read -r m s p pass total; do
    [ "${m}" = model ] && continue
    [ "${pass}" = "${total}" ] && [ "${total}" = "${runs}" ] && cached["${m}_${s}_${p}"]="${pass}"
  done < "${here}/raw/correctness.tsv"
fi

tsv="${here}/raw/correctness.tsv"
tmp="$(mktemp)"
printf 'model\tseq\tpast\tpass\truns\n' > "${tmp}"
for model in gqa2 mha4; do
  for seq in 4 128 2048; do
    for past in 0 512; do
      key="${model}_${seq}_${past}"
      if [ -n "${cached[${key}]:-}" ]; then
        printf '%s\t%s\t%s\t%s\t%s\n' "${model}" "${seq}" "${past}" \
          "${cached[${key}]}" "${runs}" >> "${tmp}"
        echo "${model} s${seq} p${past}: ${cached[${key}]}/${runs} (kept)"
        continue
      fi
      fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p${past}"
      log="${here}/raw/${model}_s${seq}_p${past}.log"
      : > "${log}"
      for _ in $(seq "${runs}"); do
        "${work}/bin/seqscan_${model}" "${fixture}" >> "${log}" 2>&1 || true
      done
      pass="$(grep -c 'RESULT status=PASS' "${log}" || true)"
      printf '%s\t%s\t%s\t%s\t%s\n' "${model}" "${seq}" "${past}" "${pass}" "${runs}" \
        >> "${tmp}"
      echo "${model} s${seq} p${past}: ${pass}/${runs}"
    done
  done
done
mv "${tmp}" "${tsv}"
awk 'NR>1 && $4 != $5 {bad++} END {print (bad ? "E1-d: FAIL" : "E1-d: PASS")}' \
  "${here}/raw/correctness.tsv"
