// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <string>

namespace tilemega::codegen {

/// Lower a verified DnnModelPlan carrying `tmexec.sm90_forward` coordinates
/// into one typed native forward translation unit (prompt §10 N-1).
std::string EmitNativeForwardProgram(mlir::ModuleOp module);

/// The same plan under the F3 skeleton (DNN3 prompt §5.7 item 2). F3 covers
/// every stage of the four DNN3 models; a stage it does not implement stops
/// the build rather than falling back to a legacy body (§1.3).
std::string EmitF3ForwardProgram(mlir::ModuleOp module);

}  // namespace tilemega::codegen
