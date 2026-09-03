#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# P4.6 ablation.  Two halves that meet in the middle:
#
#   kappa_probe   analytic.  Applies §2.3's C_kappa = floor(./kappa) o C to
#                 every derived edge of both accepted models and reports the
#                 two columns that move in opposite directions -- events polled
#                 per consumer task, and producer tasks waited on beyond the
#                 ones actually read.  No GPU.
#   the arm sweep hardware.  Compiles the L2 megakernel at each kappa and
#                 measures it, plus a `nosync` probe that deletes the grid half
#                 of L1's barrier.  `nosync` is numerically wrong on purpose;
#                 it is the ceiling on what *any* synchronization change can be
#                 worth, kappa included, so the analytic benefit and the
#                 measured cost can be compared on one scale.
#
# The sweep is interleaved and reported paired, for the reason F-46 records.
# It has its own floor built in: kappa does not touch L1, so every arm's l1_ms
# is a null control against `stage`'s, and `nosync` does not touch L2, so its
# l2_ms is one too.
#
#   bash docs/experiments/COARSEN/run.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
build="${repo}/build-phase12"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-60}"
# Both models by default; the override exists so an interrupted sweep can be
# resumed on the half that is missing instead of re-measuring the half that is
# already complete.
models="${MODELS:-gqa2 mha4}"
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
    "${here}/kappa_probe.cpp" \
    -L"${repo}/build-portable" -ltilemega \
    "${barvinok_build}/.libs/libbarvinok.a" "${isl_build}/.libs/libisl.a" \
    "${polylib_build}/.libs/libpolylibgmp.a" \
    -L"${mlir_build}/lib" -lLLVMSupport -lLLVMDemangle \
    -L/usr/local/cuda/lib64 -lcudart -lntl -lgmp -lpthread \
    -o "${raw}/kappa_probe" 2>&1 | tee "${raw}/kappa_build.txt"
  "${raw}/kappa_probe" > "${raw}/kappa.txt" 2>&1
  grep '^POINT' "${raw}/kappa.txt"
else
  echo "SKIPPED: isl/barvinok or MLIR build tree missing" | tee "${raw}/kappa.txt"
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

# The tile configuration is the DP's own uniform answer, so the kappa question
# is asked about the kernel the solver recommends rather than about a default.
arms=(stage k1 k2 k4 k8 k16 k32 k64 k128 k256 nosync)
flag_for() {
  case "$1" in
    stage)  echo "-DTILEMEGA_EVENT_KAPPA=0" ;;
    nosync) echo "-DTILEMEGA_UNSAFE_NO_GRID_SYNC=1" ;;
    k*)     echo "-DTILEMEGA_EVENT_KAPPA=${1#k}" ;;
  esac
}

median() { sort -n | awk '{v[NR]=$1} END {if(NR==0){print "nan";exit} printf "%.6f\n",(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }

printf 'model\tarm\tl1_ms\tl2_ms\tpass\tstatus\n' > "${raw}/kappa_arms.tsv"
for model in ${models}; do
  src="${gqa_src}"; fixture="${gqa_fixture}"
  if [[ "${model}" == mha4 ]]; then src="${mha_src}"; fixture="${mha_fixture}"; fi
  plan="${solver_raw}/plan_${model}_uniform.h"
  for arm in "${arms[@]}"; do
    "${nvcc}" "${src}" "${common[@]}" -include "${plan}" "$(flag_for "${arm}")" \
      "${link[@]}" -o "${raw}/bin/${model}_${arm}" 2> "${raw}/log/${model}_${arm}.ptxas"
  done

  echo "== ${model}: ${runs} interleaved rounds over ${#arms[@]} arms"
  for ((round = 0; round < runs; ++round)); do
    for ((slot = 0; slot < ${#arms[@]}; ++slot)); do
      arm=${arms[$(( (round + slot) % ${#arms[@]} ))]}
      # gpu_stat_run.sh exits non-zero when a run mismatches, and the nosync
      # arm mismatches on purpose -- so only that arm is allowed to fail here.
      # It is still required to fail *every* time, by the check below.
      "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 -l "kappa_${model}_${arm}" \
        -k -o "${raw}/final/${model}_${arm}/r${round}" \
        -- "${raw}/bin/${model}_${arm}" "${fixture}" \
        >> "${raw}/log/${model}_${arm}.final" 2>&1 \
        || [[ "${arm}" == nosync ]]
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
    want=PASS; [[ "${arm}" == nosync ]] && want=MISMATCH
    printf '%s\t%s\t%s\t%s\t%s/%s\t%s\n' "${model}" "${arm}" "${l1}" "${l2}" \
      "${pass}" "${#logs[@]}" "${want}" | tee -a "${raw}/kappa_arms.tsv"
  done
done

# Every kappa is a *correct* synchronization scheme -- coarser events can only
# add ordering -- so every kappa arm must pass in every fresh process.  The
# nosync arm must fail in every one of them: it is a timing probe and the
# harness is built so it cannot report otherwise.
awk -F'\t' 'NR>1 {split($5,p,"/");
    if ($6 == "PASS"  && (p[1]+0 != p[2]+0 || p[2]+0 == 0)) {print "FAIL: " $1 " " $2 " passed " $5; bad=1}
    if ($6 == "MISMATCH" && p[1]+0 != 0) {print "FAIL: " $1 " " $2 " reported PASS with no grid sync"; bad=1} }
  END {if (bad) exit 1; print "every kappa arm PASSed in every fresh process; nosync never did"}' \
  "${raw}/kappa_arms.tsv"

python3 "${here}/summarize_kappa.py" "${raw}" > "${raw}/kappa_summary.txt"
cat "${raw}/kappa_summary.txt"
echo PASS > "${raw}/status.txt"
