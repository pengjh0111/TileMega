#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# P4.8 Place.  Two halves, and the first one decides whether the second can
# possibly matter:
#
#   place_probe   analytic.  For every operand of every operator of both
#                 accepted models, counts |R(c1) n R(c2)| exactly with
#                 barvinok -- the elements two tasks of the same operator both
#                 read.  A placement objective sums this over co-located
#                 pairs, so an affinity that does not depend on (c1, c2) makes
#                 every placement equal by construction.  No GPU.
#   the arm sweep hardware.  Compiles the megakernel under four bijections of
#                 the CTA index and measures them, plus a second copy of the
#                 identity build whose measured "effect" is the noise floor.
#                 `scatter` is the two-sided half: it destroys index adjacency
#                 outright, so its cost bounds what any placement could win.
#
# Interleaved and reported paired, for the reason F-46 records.  Every arm is
# a bijection, so every arm must produce the right answer in every fresh
# process; the check below hard-fails otherwise.
#
#   bash docs/experiments/PLACE/run.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
build="${repo}/build-phase12"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-60}"
mkdir -p "${raw}/bin" "${raw}/log" "${raw}/final"

gqa_src="${repo}/docs/experiments/E2E_GEN/raw/generated_e2e.cu"
gqa_fixture="${repo}/docs/experiments/E2E/fixture"
mha_src="${repo}/docs/experiments/P3_GENERALIZATION/raw/generated.cu"
mha_fixture="${repo}/docs/experiments/P3_GENERALIZATION/raw/fixture"
solver_raw="${repo}/docs/experiments/SOLVER/raw"

# ------------------------------------------------------- analytic half ------
isl_build="${TILEMEGA_ISL_BUILD_DIR:-${repo}/build-isl}"
polylib_build="${TILEMEGA_POLYLIB_BUILD_DIR:-${repo}/build-polylib}"
barvinok_build="${TILEMEGA_BARVINOK_BUILD_DIR:-${repo}/build-barvinok}"
mlir_src="${TILEMEGA_MLIR_SRC:-/tmp/tilemega-vf-llvm-project}"
mlir_build="${TILEMEGA_MLIR_BUILD:-/tmp/tilemega-vf-build/llvm-project}"
if [[ -d "${barvinok_build}" && -d "${mlir_build}/lib" ]]; then
  g++ -std=c++17 -O2 -I"${repo}/include" -I"${repo}/build-portable/include" \
    -I"${mlir_src}/llvm/include" -I"${mlir_build}/include" \
    -I"${repo}/third_party/barvinok/isl/include" -I"${isl_build}/include" \
    "${here}/place_probe.cpp" \
    -L"${repo}/build-portable" -ltilemega \
    "${barvinok_build}/.libs/libbarvinok.a" "${isl_build}/.libs/libisl.a" \
    "${polylib_build}/.libs/libpolylibgmp.a" \
    -L"${mlir_build}/lib" -lLLVMSupport -lLLVMDemangle \
    -L/usr/local/cuda/lib64 -lcudart -lntl -lgmp -lpthread \
    -o "${raw}/place_probe" 2>&1 | tee "${raw}/place_build.txt"
  "${raw}/place_probe" > "${raw}/affinity.txt" 2>&1
  grep '^SUMMARY' "${raw}/affinity.txt"
else
  echo "SKIPPED: isl/barvinok or MLIR build tree missing" | tee "${raw}/affinity.txt"
fi

# ------------------------------------------------------- hardware half ------
for needed in "${build}/libtilemega.a" "${gqa_src}" "${mha_src}" \
              "${solver_raw}/plan_gqa2_uniform.h" "${solver_raw}/plan_mha4_uniform.h"; do
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

# The tile configuration is the DP's own uniform answer, so the placement
# question is asked about the kernel the solver recommends.
arms=(ident ident2 pair reverse scatter)
flag_for() {
  case "$1" in
    ident|ident2) echo "-DTILEMEGA_PLACEMENT=0" ;;
    pair)         echo "-DTILEMEGA_PLACEMENT=1" ;;
    reverse)      echo "-DTILEMEGA_PLACEMENT=2" ;;
    scatter)      echo "-DTILEMEGA_PLACEMENT=3" ;;
  esac
}

median() { sort -n | awk '{v[NR]=$1} END {if(NR==0){print "nan";exit} printf "%.6f\n",(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }

printf 'model\tarm\tl1_ms\tl2_ms\tpass\n' > "${raw}/place_arms.tsv"
for model in gqa2 mha4; do
  src="${gqa_src}"; fixture="${gqa_fixture}"
  if [[ "${model}" == mha4 ]]; then src="${mha_src}"; fixture="${mha_fixture}"; fi
  plan="${solver_raw}/plan_${model}_uniform.h"
  for arm in "${arms[@]}"; do
    "${nvcc}" "${src}" "${common[@]}" -include "${plan}" "$(flag_for "${arm}")" \
      "${link[@]}" -o "${raw}/bin/${model}_${arm}" 2> "${raw}/log/${model}_${arm}.ptxas"
  done
  # `ident` is the default build, so it must be the binary the rest of the
  # repository already measures.  Anything else means the knob changed code it
  # was not supposed to touch, and the whole sweep is against a moved baseline.
  if ! cmp -s <(cuobjdump -sass "${raw}/bin/${model}_ident" | grep -v '^identifier') \
              <(cuobjdump -sass "${raw}/bin/${model}_ident2" | grep -v '^identifier'); then
    echo "FAIL: ident and ident2 are not the same code" >&2; exit 1
  fi

  echo "== ${model}: ${runs} interleaved rounds over ${#arms[@]} arms"
  for ((round = 0; round < runs; ++round)); do
    for ((slot = 0; slot < ${#arms[@]}; ++slot)); do
      arm=${arms[$(( (round + slot) % ${#arms[@]} ))]}
      "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 -l "place_${model}_${arm}" \
        -k -o "${raw}/final/${model}_${arm}/r${round}" \
        -- "${raw}/bin/${model}_${arm}" "${fixture}" \
        >> "${raw}/log/${model}_${arm}.final" 2>&1
    done
  done

  for arm in "${arms[@]}"; do
    mapfile -t logs < <(find "${raw}/final/${model}_${arm}" -name 'run_*.log' | sort)
    l1=$(grep -h '^E2E_TIME' "${logs[@]}" | sed -E 's/.*l1_ms=([0-9.]+).*/\1/' | median)
    l2=$(grep -h '^E2E_TIME' "${logs[@]}" | sed -E 's/.* l2_ms=([0-9.]+).*/\1/' | median)
    pass=$(grep -h -c '^RESULT status=PASS' "${logs[@]}" | awk '{t+=$1} END{print t+0}')
    printf '%s\t%s\t%s\t%s\t%s/%s\n' "${model}" "${arm}" "${l1}" "${l2}" \
      "${pass}" "${#logs[@]}" | tee -a "${raw}/place_arms.tsv"
  done
done

# A placement is a bijection: it moves work between CTAs and creates none, so
# every arm is numerically exact and must pass in every fresh process.  A
# single mismatch means the permutation and the activity bound disagree, which
# is a correctness bug and not a slow arm.
awk -F'\t' 'NR>1 {split($5,p,"/");
    if (p[1]+0 != p[2]+0 || p[2]+0 == 0) {print "FAIL: " $1 " " $2 " passed " $5; bad=1} }
  END {if (bad) exit 1; print "every placement arm PASSed in every fresh process"}' \
  "${raw}/place_arms.tsv"

python3 "${here}/summarize_place.py" "${raw}" > "${raw}/place_summary.txt"
cat "${raw}/place_summary.txt"
echo PASS > "${raw}/status.txt"
