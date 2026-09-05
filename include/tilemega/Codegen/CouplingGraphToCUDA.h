// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 Coupling Graph lowering (Phase 2 stub).
#pragma once
#include <mlir/IR/BuiltinOps.h>
#include <cstdint>
#include <string>
#include <vector>
namespace tilemega::codegen {
struct RuntimeVariantModule {
  mlir::ModuleOp module;
  std::uint32_t seq_begin = 1;
  std::uint32_t seq_end = 65535;
};
class CouplingGraphToCUDA {
 public:
  /// Lower the verified CG dialect module. No JSON or parallel C++ graph
  /// representation is accepted (§2.6, F-14).
  std::string Lower(mlir::ModuleOp module) const;
  /// Fuse independently instantiated/derived granularities into one CUDA
  /// translation unit. Model structure must match; granularity-dependent
  /// GEMM and dependency tables intentionally may differ.
  std::string LowerVariants(
      std::vector<RuntimeVariantModule> const& variants) const;
};
}  // namespace tilemega::codegen
