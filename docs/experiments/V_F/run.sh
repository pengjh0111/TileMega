#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
compiler="${repo}/third_party/cutlass/cutlass_compiler"
llvm="${compiler}/external/llvm-project"
build="${repo}/build-cutlass-compiler"
raw="${here}/raw"
commit="$(tr -d '[:space:]' < "${compiler}/LLVM_COMMIT")"
jobs="${TILEMEGA_LLVM_JOBS:-32}"
mkdir -p "${raw}" "${compiler}/external"

{
  echo "LLVM_COMMIT=${commit}"
  echo "jobs=${jobs}"
  echo "cmake=$(cmake --version | head -1)"
  echo "ninja=$(ninja --version)"
} > "${raw}/environment.txt"

if [[ ! -d "${llvm}/.git" ]]; then
  git init "${llvm}"
  git -C "${llvm}" remote add origin https://github.com/llvm/llvm-project.git
fi
if [[ "$(git -C "${llvm}" rev-parse HEAD 2>/dev/null || true)" != "${commit}" ]]; then
  if ! git -C "${llvm}" fetch --depth=1 origin "${commit}"; then
    echo "BLOCKED: unable to fetch pinned LLVM commit ${commit}" \
      | tee "${raw}/run_status.txt"
    exit 77
  fi
  git -C "${llvm}" checkout --detach FETCH_HEAD
fi

start=$SECONDS
cmake -S "${compiler}" -B "${build}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCUTLASS_COMPILER_USE_BUNDLED_LLVM=ON > "${raw}/configure.txt" 2>&1
set +e
timeout 2h ninja -C "${build}" -j "${jobs}" cute-opt base-opt \
  check-cute check-cute-unittests > "${raw}/build_and_tests.txt" 2>&1
status=$?
set -e
echo "$((SECONDS - start))" > "${raw}/build_test_seconds.txt"
if [[ ${status} -eq 124 ]]; then
  echo "BLOCKED: LLVM/CuTe build exceeded two hours" | tee "${raw}/run_status.txt"
  exit 77
elif [[ ${status} -ne 0 ]]; then
  echo "FAIL: build or test target failed with ${status}" | tee "${raw}/run_status.txt"
  exit "${status}"
fi

cute_opt="${build}/cute_ir/tools/cute-opt/cute-opt"
"${cute_opt}" -cute-fold-static --split-input-file \
  "${compiler}/cute_ir/test/Transforms/CuteFoldStatic/fold_layout_algebra.mlir" \
  > "${raw}/static_layout_algebra.mlir"
"${cute_opt}" -cute-fold-static --split-input-file \
  "${compiler}/cute_ir/test/Transforms/CuteFoldStatic/fold_tiling_partitioning_products.mlir" \
  > "${raw}/static_divide.mlir"
"${cute_opt}" -cute-fold-static --split-input-file \
  "${compiler}/cute_ir/test/Transforms/CuteFoldStatic/no_fold_dynamic.mlir" \
  > "${raw}/dynamic_after_fold.mlir"
"${cute_opt}" -cute-fold-static "${here}/dynamic_layout_probe.mlir" \
  > "${raw}/probe_dynamic_after_fold.mlir"
"${cute_opt}" --verify-diagnostics --split-input-file \
  "${here}/dynamic_inverse_probe.mlir" \
  > "${raw}/probe_inverse_errors.txt" 2>&1
"${cute_opt}" -cute-expand-ops --split-input-file \
  "${compiler}/cute_ir/test/Conversion/CuteExpandOps/LayoutAlgebra/flatten.mlir" \
  > "${raw}/flatten_after_expand.mlir"
"${cute_opt}" -cute-expand-ops --split-input-file \
  "${compiler}/cute_ir/test/Conversion/CuteExpandOps/LayoutAlgebra/coalesce.mlir" \
  > "${raw}/coalesce_after_expand.mlir"
"${cute_opt}" --verify-diagnostics --split-input-file \
  "${compiler}/cute_ir/test/Dialect/Cute/LayoutAlgebra/right_inverse_errors.mlir" \
  > "${raw}/right_inverse_errors.txt" 2>&1
"${cute_opt}" --verify-diagnostics --split-input-file \
  "${compiler}/cute_ir/test/Dialect/Cute/LayoutAlgebra/left_inverse_errors.mlir" \
  > "${raw}/left_inverse_errors.txt" 2>&1

PYTHONPATH="${repo}/third_party/cutlass/python" \
  python3 "${here}/pycute_probe.py" > "${raw}/pycute.json"

unit_tests="$(grep -E '\[  PASSED  \] [0-9]+ tests?\.' "${raw}/build_and_tests.txt" \
  | sed -E 's/.*\] ([0-9]+) tests?.*/\1/' \
  | awk '{sum += $1} END {print sum + 0}')"
lit_tests="$(grep 'Total Discovered Tests:' "${raw}/build_and_tests.txt" \
  | tail -1 | awk '{print $4}')"
{
  echo "cute_lit_discovered=${lit_tests}"
  echo "cute_lit_passed=${lit_tests}"
  echo "cute_unittests_passed=${unit_tests}"
} > "${raw}/test_summary.txt"

{
  echo "PASS"
  echo "cute_opt=${cute_opt}"
  cat "${raw}/test_summary.txt"
} | tee "${raw}/run_status.txt"
