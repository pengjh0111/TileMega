// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 TaskBody instantiation (Phase 2 stub).
#pragma once
#include <mlir/IR/BuiltinOps.h>
#include <string>
namespace tilemega::codegen {
class TaskBodyEmitter { public: std::string Emit(mlir::ModuleOp module) const; };
}  // namespace tilemega::codegen
