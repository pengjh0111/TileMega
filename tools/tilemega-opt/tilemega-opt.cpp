// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// tilemega-opt -- TileMega's mlir-opt driver.
//
// Why this exists (TILEMEGA_SKELETON.md section 4, "dependency notes"):
//   Upstream tensor_ir-opt links cuda-tile as a library, but its
//   registerDialects() does *not* register the cuda_tile dialect, so it cannot
//   parse any IR containing cuda_tile ops. TileMega's codegen emits exactly
//   that: IR mixing nv_tensor_ir and cuda_tile, so both must be registered.

#include "tensor_ir/Registration/Registration.h"

#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  // Register only the generic transform passes (canonicalize / cse / ...).
  // registerAllPasses() and registerAllDialects() are deliberately not called:
  // that would pull all of upstream MLIR's dialects and passes into the link,
  // while TileMega only handles nv_tensor_ir and cuda_tile. Upstream
  // tensor_ir-opt takes the same approach.
  mlir::registerTransformsPasses();

  mlir::DialectRegistry registry;

  // TensorIR side: nv_tensor_ir and the extensions it depends on.
  mlir::nv_tensor_ir::registerDialects(registry);

  // CUDA Tile side: this line is what upstream tensor_ir-opt is missing.
  registry.insert<mlir::cuda_tile::CudaTileDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TileMega optimizer driver\n", registry));
}
