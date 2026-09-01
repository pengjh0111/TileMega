// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 Coupling Graph lowering (Phase 2 stub).
#pragma once
#include <mlir/IR/BuiltinOps.h>
#include <string>
namespace tilemega::codegen {
class CouplingGraphToCUDA {
 public:
  /// Lower the verified CG dialect module. No JSON or parallel C++ graph
  /// representation is accepted (§2.6, F-14).
  std::string Lower(mlir::ModuleOp module) const;
};
}  // namespace tilemega::codegen
