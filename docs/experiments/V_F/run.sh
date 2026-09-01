#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
compiler="${repo}/third_party/cutlass/cutlass_compiler"
raw="${here}/raw"
mkdir -p "${raw}"

{
  echo "LLVM_COMMIT=$(tr -d '[:space:]' < "${compiler}/LLVM_COMMIT")"
  echo "bundled_llvm=$([[ -d "${compiler}/external/llvm-project" ]] && echo present || echo missing)"
  echo "cute_opt=$(command -v cute-opt || echo missing)"
  echo "mlir_opt=$(command -v mlir-opt || echo missing)"
} > "${raw}/environment.txt"

(
  cd "${repo}"
  rg -n "no_fold_(composition|zipped_divide)|cute\.(composition|zipped_divide|right_inverse)" \
    "third_party/cutlass/cutlass_compiler/cute_ir/test/Transforms/CuteFoldStatic/no_fold_dynamic.mlir" \
    "third_party/cutlass/cutlass_compiler/cute_ir/test/Transforms/CuteFoldStatic/fold_layout_algebra.mlir" \
    "third_party/cutlass/cutlass_compiler/cute_ir/test/Conversion/CuteExpandOps/pipeline_with_fold_static.mlir"
) > "${raw}/source_audit.txt"

if [[ ! -d "${compiler}/external/llvm-project" ]] && \
   ! command -v mlir-opt >/dev/null 2>&1; then
  echo "BLOCKED: the pinned LLVM/MLIR checkout is absent and no compatible MLIR installation is available" \
    | tee "${raw}/run_status.txt"
  exit 77
fi

cmake -S "${compiler}" -B "${here}/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCUTLASS_COMPILER_USE_BUNDLED_LLVM=ON
ninja -C "${here}/build" cute-opt
ninja -C "${here}/build" check-cute
"${here}/build/cute_ir/tools/cute-opt/cute-opt" -cute-fold-static \
  "${here}/dynamic_layout_probe.mlir" > "${raw}/dynamic_layout_after_fold.mlir"
echo PASS > "${raw}/run_status.txt"
