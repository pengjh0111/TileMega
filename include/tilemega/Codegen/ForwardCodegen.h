// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <string>

namespace tilemega::codegen {

/// Lower a verified DnnModelPlan carrying `tmexec.sm90_forward` coordinates
/// into one typed native forward translation unit (prompt §10 N-1).
std::string EmitNativeForwardProgram(mlir::ModuleOp module);

}  // namespace tilemega::codegen
