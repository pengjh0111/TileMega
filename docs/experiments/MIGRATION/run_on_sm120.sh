#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# Part 4.4 -- does the sm_89-calibrated cost model still rank on a 5090?
#
# The 4090 and the 5090 have comparable shared memory per SM (~100 KB) but 128
# vs 170 SMs, so the wave-quantization term is the one that moves and the
# ranking is the thing that can break.  This script re-measures a 100-config
# subset of the ORACLE sweep on the migration target and scores the *unchanged*
# 4090 coefficients against it.
#
# Self-contained: it needs a checkout, a CUDA toolkit and the two committed
# fixtures.  Nothing calibrates here -- calibrating on the 5090 would answer a
# different question.
#
#   bash docs/experiments/MIGRATION/run_on_sm120.sh
#   TOP=50 SAMPLE=50 RUNS=3 JOBS=16 bash docs/experiments/MIGRATION/run_on_sm120.sh
#
# Exit codes: 3 = wrong GPU (this is the calibration target), 77 = missing
# prerequisite.  It hard-fails rather than degrading, for the same reason
# CLUSTER/run_on_cluster_gpu.sh does: a fallback result is not the result.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
build="${BUILD:-${repo}/build-migrate}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
label="${LABEL:-sm120}"
jobs="${JOBS:-16}"
runs="${RUNS:-3}"
top="${TOP:-50}"
sample="${SAMPLE:-50}"
seed="${SEED:-20260904}"
mkdir -p "${raw}/bin" "${raw}/log"

gqa_src="${repo}/docs/experiments/E2E_GEN/raw/generated_e2e.cu"
gqa_fixture="${repo}/docs/experiments/E2E/fixture"
mha_src="${repo}/docs/experiments/P3_GENERALIZATION/raw/generated.cu"
mha_fixture="${repo}/docs/experiments/P3_GENERALIZATION/raw/fixture"
for needed in "${gqa_src}" "${gqa_fixture}/manifest.json" \
              "${mha_src}" "${mha_fixture}/manifest.json" \
              "${repo}/docs/experiments/ORACLE/raw/screen_gqa2.tsv" \
              "${repo}/docs/experiments/COST_MODEL/raw/registers_gqa2.tsv"; do
  [[ -e "${needed}" ]] || { echo "BLOCKED: ${needed} missing" >&2; exit 77; }
done
command -v "${nvcc}" >/dev/null 2>&1 || [[ -x "${nvcc}" ]] \
  || { echo "BLOCKED: no nvcc at ${nvcc}" >&2; exit 77; }

echo "== building the host tools and libtilemega =="
cmake -S "${repo}" -B "${build}" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${build}" --target tilemega tilemega-migrate -j "$(nproc)" >/dev/null

echo "== probing the GPU (must not be the calibration target) =="
"${build}/tools/tilemega-migrate" --repo "${repo}" --probe

echo "== selecting the subset from the committed sm_89 sweep =="
"${build}/tools/tilemega-migrate" --repo "${repo}" --out "${raw}" \
  --top "${top}" --sample "${sample}" --seed "${seed}" >/dev/null

common=(-std=c++17 -O2 -arch=native -lineinfo -Xptxas=-v
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)
export common_str="${common[*]}" link_str="${link[*]}" nvcc raw label

compile_one() {  # $1=model $2=src $3=M $4=N $5=K $6=S $7=Kc
  local tag="$1_$3x$4x$5s$6k$7"
  # shellcheck disable=SC2086
  "${nvcc}" "$2" ${common_str} \
    -DTILEMEGA_GEMM_TILE_M="$3" -DTILEMEGA_GEMM_TILE_N="$4" \
    -DTILEMEGA_GEMM_TILE_K="$5" -DTILEMEGA_GEMM_STAGES="$6" \
    -DTILEMEGA_GEMM_SPLIT_K="$7" \
    ${link_str} -o "${raw}/bin/${tag}" 2> "${raw}/log/${tag}.ptxas" \
    && echo "OK ${tag}" || echo "FAIL ${tag}"
}
export -f compile_one

median() { sort -n | awk '{v[NR]=$1} END {if(NR==0){print "nan";exit} printf "%.6f\n",(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
header='tile_m\ttile_n\ttile_k\tstages\tsplit_k\tl05_ms\tl1_ms\tl2_ms\tstatus\tl05_hash\tsmem\tctas_per_sm\tgrid\n'

for model in gqa2 mha4; do
  src="${gqa_src}"; fixture="${gqa_fixture}"
  [[ "${model}" == mha4 ]] && { src="${mha_src}"; fixture="${mha_fixture}"; }
  subset="${raw}/subset_${model}.txt"
  echo "== ${model}: compiling $(wc -l < "${subset}") configurations =="
  while read -r m n k s kc; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${model}" "${src}" "${m}" "${n}" "${k}" "${s}" "${kc}"
  done < "${subset}" \
    | xargs -P "${jobs}" -n 7 bash -c 'compile_one "$0" "$1" "$2" "$3" "$4" "$5" "$6"' \
    > "${raw}/compile_status_${label}_${model}.txt"

  echo "== ${model}: measuring, ${runs} runs per configuration =="
  {
    printf "${header}"
    while read -r m n k s kc; do
      tag="${model}_${m}x${n}x${k}s${s}k${kc}"
      bin="${raw}/bin/${tag}"
      [[ -x "${bin}" ]] || continue
      l05=(); l1=(); l2=(); status=PASS; hash=; grid=; smem=; ctas=
      for ((i = 0; i < runs; ++i)); do
        out="$(timeout 120 "${bin}" "${fixture}" 2>/dev/null)" || { status=RUNFAIL; break; }
        grep -q '^RESULT status=PASS' <<<"${out}" || status=MISMATCH
        l05+=("$(sed -n 's/^E2E_TIME l05_ms=\([0-9.]*\).*/\1/p' <<<"${out}")")
        l1+=("$(sed -n 's/.*l1_ms=\([0-9.]*\).*/\1/p' <<<"${out}")")
        l2+=("$(sed -n 's/.* l2_ms=\([0-9.]*\).*/\1/p' <<<"${out}")")
        hash="$(sed -n 's/^E2E_HASH l05=\([0-9a-f]*\).*/\1/p' <<<"${out}")"
        grid="$(sed -n 's/.*grid=\([0-9]*\).*/\1/p' <<<"${out}")"
        smem="$(sed -n 's/^E2E_RESOURCE.*smem=\([0-9]*\).*/\1/p' <<<"${out}")"
        ctas="$(sed -n 's/.*ctas_per_sm=\([0-9]*\).*/\1/p' <<<"${out}")"
      done
      if [[ "${status}" != PASS ]]; then
        printf '%s\t%s\t%s\t%s\t%s\tnan\tnan\tnan\t%s\t-\t-\t-\t-\n' "${m}" "${n}" "${k}" "${s}" "${kc}" "${status}"
        continue
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tPASS\t%s\t%s\t%s\t%s\n' \
        "${m}" "${n}" "${k}" "${s}" "${kc}" \
        "$(printf '%s\n' "${l05[@]}" | median)" \
        "$(printf '%s\n' "${l1[@]}" | median)" \
        "$(printf '%s\n' "${l2[@]}" | median)" \
        "${hash}" "${smem}" "${ctas}" "${grid}"
    done < "${subset}"
  } > "${raw}/screen_${label}_${model}.tsv"

  # Registers are a tier-3 quantity and this is a different compiler target, so
  # they are read back out of *this* machine's ptxas logs, never reused.
  {
    echo -e "# shape\tmax_registers  (max over entry points, ${label})"
    for f in "${raw}/log/${model}"_*.ptxas; do
      [[ -e "${f}" ]] || continue
      base=${f##*/}; base=${base%.ptxas}; base=${base#${model}_}
      regs=$( { grep -o 'Used [0-9]* registers' "${f}" || true; } |
              awk '{print $2}' | sort -rn | awk 'NR==1')
      [[ -n "${regs}" ]] && echo -e "${base%%k*}\t${regs}"
    done | sort -u -k1,1 -k2,2rn | awk '!seen[$1]++'
  } > "${raw}/registers_${label}_${model}.tsv"
done

echo "== scoring the unchanged sm_89 coefficients against both machines =="
"${build}/tools/tilemega-migrate" --repo "${repo}" --out "${raw}" \
  --label "${label}" --top "${top}" --sample "${sample}" --seed "${seed}" \
  | tee "${raw}/summary_${label}.txt"

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv \
  > "${raw}/device_${label}.txt" 2>/dev/null || true
echo "done: ${raw}/summary_${label}.txt"
