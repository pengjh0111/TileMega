#!/usr/bin/env bash
# Part 1 dependency-feasibility experiment: does isl/barvinok coexist with
# MLIR, and can isl represent the symbolic-divisor expressions TileMega's
# coupling table needs? See result.md for the analysis; this reproduces the
# two probes and captures raw output.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
mkdir -p "${raw}"

isl_build="${TILEMEGA_ISL_BUILD_DIR:-${repo}/build-isl}"
polylib_build="${TILEMEGA_POLYLIB_BUILD_DIR:-${repo}/build-polylib}"
barvinok_build="${TILEMEGA_BARVINOK_BUILD_DIR:-${repo}/build-barvinok}"
for d in "${isl_build}" "${polylib_build}" "${barvinok_build}"; do
  if [[ ! -d "${d}" ]]; then
    echo "BLOCKED: ${d} not found; build isl/polylib/barvinok first (see" \
         "docs/DEPENDENCIES.md)" | tee "${raw}/run_status.txt"
    exit 77
  fi
done

mlir_src="${TILEMEGA_MLIR_SRC:-/tmp/tilemega-vf-llvm-project}"
mlir_build="${TILEMEGA_MLIR_BUILD:-/tmp/tilemega-vf-build/llvm-project}"
if [[ ! -d "${mlir_build}/lib" ]]; then
  echo "BLOCKED: MLIR build tree ${mlir_build} not found (see docs/BUILD_MLIR.md)" \
    | tee "${raw}/run_status.txt"
  exit 77
fi

echo "== iscc smoke tests (barvinok's isl-syntax counting tool) ==" \
  | tee "${raw}/iscc_smoke.txt"
"${barvinok_build}/iscc" <<< 'card [n] -> { [i] : 0 <= i < n };' \
  >> "${raw}/iscc_smoke.txt" 2>&1
"${barvinok_build}/iscc" <<< \
  'A := { [i] : 0 <= i < 10 }; B := { [i] : 0 <= i < 5 }; B <= A;' \
  >> "${raw}/iscc_smoke.txt" 2>&1
echo "-- parametric divisor: g (Tm) literal, theta (S) symbolic --" \
  >> "${raw}/iscc_smoke.txt"
"${barvinok_build}/iscc" <<< \
  'card [S] -> { [m] : 0 <= m < (S + 127)/128 };' >> "${raw}/iscc_smoke.txt" 2>&1
cat "${raw}/iscc_smoke.txt"

echo "== crosslink_probe.cpp: MLIR + isl + barvinok in one process ==" \
  | tee "${raw}/crosslink_build.txt"
g++ -std=c++17 -O0 -g \
  -I"${mlir_src}/mlir/include" -I"${mlir_build}/tools/mlir/include" \
  -I"${mlir_src}/llvm/include" -I"${mlir_build}/include" \
  -I"${repo}/third_party/barvinok/isl/include" -I"${isl_build}/include" \
  -I"${repo}/third_party/barvinok" -I"${repo}/third_party/barvinok/barvinok" \
  -I"${barvinok_build}" \
  -c "${here}/crosslink_probe.cpp" -o "${raw}/crosslink_probe.o" \
  2>&1 | tee -a "${raw}/crosslink_build.txt"
g++ -O0 -g "${raw}/crosslink_probe.o" \
  -L"${mlir_build}/lib" \
  -Wl,--start-group -lMLIRIR -lMLIRSupport -lMLIRPresburger -Wl,--end-group \
  -lLLVMSupport -lLLVMDemangle \
  "${barvinok_build}/.libs/libbarvinok.a" \
  "${isl_build}/.libs/libisl.a" \
  "${polylib_build}/.libs/libpolylibgmp.a" \
  -lntl -lgmp -lpthread \
  -o "${raw}/crosslink_probe" 2>&1 | tee -a "${raw}/crosslink_build.txt"
"${raw}/crosslink_probe" 2>&1 | tee "${raw}/crosslink_run.txt"

echo "== parametric_div_probe.c: does isl_aff_div accept a parametric divisor? ==" \
  | tee "${raw}/parametric_div_build.txt"
gcc -O0 -g \
  -I"${repo}/third_party/barvinok/isl/include" -I"${isl_build}/include" \
  -I"${repo}/third_party/barvinok" -I"${repo}/third_party/barvinok/barvinok" \
  -I"${barvinok_build}" \
  -c "${here}/parametric_div_probe.c" -o "${raw}/parametric_div_probe.o" \
  2>&1 | tee -a "${raw}/parametric_div_build.txt"
gcc "${raw}/parametric_div_probe.o" \
  "${barvinok_build}/.libs/libbarvinok.a" \
  "${isl_build}/.libs/libisl.a" \
  "${polylib_build}/.libs/libpolylibgmp.a" \
  -lntl -lgmp -lstdc++ -lm \
  -o "${raw}/parametric_div_probe" 2>&1 | tee -a "${raw}/parametric_div_build.txt"
"${raw}/parametric_div_probe" > "${raw}/parametric_div_run.txt" 2>&1 || true
cat "${raw}/parametric_div_run.txt"

echo "== coarsen_probe.cpp: C_kappa for kappa in {1,2,4}, with expression sizes ==" \
  | tee "${raw}/coarsen_build.txt"
tilemega_link=(-L"${repo}/build-portable" -ltilemega
  "${barvinok_build}/.libs/libbarvinok.a" "${isl_build}/.libs/libisl.a"
  "${polylib_build}/.libs/libpolylibgmp.a"
  -L"${mlir_build}/lib" -lLLVMSupport -lLLVMDemangle
  -L/usr/local/cuda/lib64 -lcudart -lntl -lgmp -lpthread)
tilemega_inc=(-I"${repo}/include" -I"${repo}/build-portable/include"
  -I"${mlir_src}/llvm/include" -I"${mlir_build}/include")
for probe in coarsen quasipoly; do
  g++ -std=c++17 "${tilemega_inc[@]}" \
    -I"${repo}/third_party/barvinok/isl/include" -I"${isl_build}/include" \
    "${here}/${probe}_probe.cpp" "${tilemega_link[@]}" \
    -o "${raw}/${probe}_probe" 2>&1 | tee -a "${raw}/coarsen_build.txt"
  "${raw}/${probe}_probe" > "${raw}/${probe}.txt" 2>&1
  cat "${raw}/${probe}.txt"
done

echo PASS > "${raw}/run_status.txt"
