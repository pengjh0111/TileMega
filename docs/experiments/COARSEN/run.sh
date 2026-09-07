#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# Queue-driven P4.6 ablation.  Every arm consumes the same generated per-worker
# task order; only the event group size changes.  The sweep is interleaved and
# reported as within-round paired ratios with bootstrap CI and Wilcoxon.
#
#   bash docs/experiments/COARSEN/run.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw_taskqueue"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-25}"
# Both models by default; the override exists so an interrupted sweep can be
# resumed on the half that is missing instead of re-measuring the half that is
# already complete.
models="${MODELS:-gqa2 mha4}"
mkdir -p "${raw}/bin" "${raw}/log" "${raw}/final"

gqa_src="${repo}/docs/experiments/SEQSCAN/raw/src/gqa2.cu"
gqa_fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/gqa2_s128_p3"
mha_src="${repo}/docs/experiments/SEQSCAN/raw/src/mha4.cu"
mha_fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/mha4_s128_p3"

for needed in "${build}/libtilemega.a" "${gqa_src}" "${mha_src}" \
              "${gqa_fixture}/manifest.json" "${mha_fixture}/manifest.json"; do
  if [[ ! -e "${needed}" ]]; then
    echo "BLOCKED: ${needed} missing" | tee "${raw}/status.txt" >&2
    exit 77
  fi
done

common=(-std=c++17 -O2 -arch=native -lineinfo -Xptxas=-v
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)

# The tile configuration is the DP's own uniform answer, so the kappa question
# is asked about the kernel the solver recommends rather than about a default.
arms=(k0 k1 k2 k4 k8 k16 k32)
flag_for() {
  case "$1" in
    k0)     echo "-DTILEMEGA_EVENT_KAPPA=0" ;;
    k*)     echo "-DTILEMEGA_EVENT_KAPPA=${1#k}" ;;
  esac
}

median() { sort -n | awk '{v[NR]=$1} END {if(NR==0){print "nan";exit} printf "%.6f\n",(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }

printf 'model\tarm\tl1_ms\tl2_ms\tpass\tstatus\n' > "${raw}/kappa_arms.tsv"
for model in ${models}; do
  src="${gqa_src}"; fixture="${gqa_fixture}"
  if [[ "${model}" == mha4 ]]; then src="${mha_src}"; fixture="${mha_fixture}"; fi
  for arm in "${arms[@]}"; do
    "${nvcc}" "${src}" "${common[@]}" "$(flag_for "${arm}")" \
      "${link[@]}" -o "${raw}/bin/${model}_${arm}" 2> "${raw}/log/${model}_${arm}.ptxas"
  done

  echo "== ${model}: ${runs} interleaved rounds over ${#arms[@]} arms"
  for ((round = 0; round < runs; ++round)); do
    for ((slot = 0; slot < ${#arms[@]}; ++slot)); do
      arm=${arms[$(( (round + slot) % ${#arms[@]} ))]}
      "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 -l "kappa_${model}_${arm}" \
        -k -o "${raw}/final/${model}_${arm}/r${round}" \
        -- "${raw}/bin/${model}_${arm}" "${fixture}" \
        >> "${raw}/log/${model}_${arm}.final" 2>&1
    done
  done

  for arm in "${arms[@]}"; do
    mapfile -t logs < <(find "${raw}/final/${model}_${arm}" -name 'run_*.log' | sort)
    l1=$(grep -h '^E2E_TIME' "${logs[@]}" | sed -E 's/.*l1_ms=([0-9.]+).*/\1/' | median)
    l2=$(grep -h '^E2E_TIME' "${logs[@]}" | sed -E 's/.* l2_ms=([0-9.]+).*/\1/' | median)
    # `grep -c` exits 1 when nothing matches, and under `pipefail` that would
    # kill the sweep on exactly the arm whose count is supposed to be zero.
    pass=$( { grep -h -c '^RESULT status=PASS' "${logs[@]}" || true; } \
            | awk '{t+=$1} END{print t+0}')
    want=PASS
    printf '%s\t%s\t%s\t%s\t%s/%s\t%s\n' "${model}" "${arm}" "${l1}" "${l2}" \
      "${pass}" "${#logs[@]}" "${want}" | tee -a "${raw}/kappa_arms.tsv"
  done
done

# Every kappa is a correct synchronization scheme: coarser events can only add
# ordering, so every arm must pass in every fresh process.
awk -F'\t' 'NR>1 {split($5,p,"/");
    if (p[1]+0 != p[2]+0 || p[2]+0 == 0) {print "FAIL: " $1 " " $2 " passed " $5; bad=1} }
  END {if (bad) exit 1; print "every kappa arm PASSed in every fresh process"}' \
  "${raw}/kappa_arms.tsv"

python3 "${here}/summarize_kappa.py" "${raw}" > "${raw}/kappa_summary.txt"
cat "${raw}/kappa_summary.txt"
echo PASS > "${raw}/status.txt"
