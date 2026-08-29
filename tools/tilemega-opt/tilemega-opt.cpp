// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// tilemega-opt —— TileMega 的 mlir-opt 驱动。
//
// 存在理由（TILEMEGA_SKELETON.md §4「依赖处理要点」）：
//   上游的 tensor_ir-opt 虽然把 cuda-tile 作为库链接了进去，但它的
//   registerDialects() **没有注册 cuda_tile dialect**，因此无法解析
//   任何含 cuda_tile op 的 IR。而 TileMega 的 Codegen 层输出的正是
//   nv_tensor_ir 与 cuda_tile 混合的 IR，必须两个都注册。

#include "tensor_ir/Registration/Registration.h"

#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  // 只注册通用变换 pass（canonicalize / cse / 等）。
  // 刻意不调 registerAllPasses()/registerAllDialects()：那会把整个上游 MLIR
  // 的 dialect 与 pass 拖进链接，而 TileMega 只处理 nv_tensor_ir + cuda_tile
  // 两个 dialect。上游 tensor_ir-opt 也是这么做的。
  mlir::registerTransformsPasses();

  mlir::DialectRegistry registry;

  // TensorIR 侧：nv_tensor_ir 及其依赖的扩展。
  mlir::nv_tensor_ir::registerDialects(registry);

  // CUDA Tile 侧：上游 tensor_ir-opt 缺的就是这一行。
  registry.insert<mlir::cuda_tile::CudaTileDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TileMega optimizer driver\n", registry));
}
