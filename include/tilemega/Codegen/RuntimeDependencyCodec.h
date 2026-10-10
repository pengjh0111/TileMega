// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/RuntimePlan.h>
#include <mlir/IR/Builders.h>

namespace tilemega::codegen {
// Retain bound contracts before a CG rewrite removes their original owners.
// Legacy window-only records keep the original three-field spelling.
mlir::DictionaryAttr EncodeRuntimeDependency(mlir::OpBuilder& builder,
                                            DependencyRecord const& record);
DependencyRecord DecodeRuntimeDependency(mlir::DictionaryAttr record);
} // namespace tilemega::codegen
