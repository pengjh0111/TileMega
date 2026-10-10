// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CountedDependencyForm.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <mlir/IR/Builders.h>
#include <optional>

namespace tilemega::dialect {
mlir::DictionaryAttr EncodeBoundCountedScatter(mlir::OpBuilder& builder,
    TileSpaceOp producer, TileSpaceOp consumer, std::string const& tensor,
    std::vector<unsigned> const& unit_axes, std::string const& binding_source,
    analysis::ParamBinding const& binding);
// Re-establish thresholds and unit ownership after endpoint phase rebasing.
mlir::DictionaryAttr RebindBoundCountedScatter(mlir::OpBuilder& builder,
    mlir::Operation* coupling, analysis::ParamBinding const& binding);
std::optional<analysis::CountedDependencyForm> ReadBoundCountedScatter(
    mlir::Operation* coupling, analysis::ParamBinding const& binding);
} // namespace tilemega::dialect
