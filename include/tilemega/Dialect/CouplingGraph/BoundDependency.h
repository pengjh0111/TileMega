// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/DependencyTable.h>
#include <mlir/IR/Builders.h>
#include <optional>

namespace tilemega::dialect {
struct BoundTaskGeometry {
  analysis::CouplingRelation relation;
  std::uint32_t producers = 0, consumers = 0;
};
mlir::DictionaryAttr EncodeBoundTaskGeometry(mlir::OpBuilder& builder,
    BoundTaskGeometry const& geometry, analysis::CouplingRelation const& producer_coordinates,
    analysis::CouplingRelation const& consumer_coordinates, analysis::ParamBinding const& binding);
std::optional<BoundTaskGeometry> ReadBoundTaskGeometry(mlir::Operation* coupling,
    analysis::ParamBinding const& binding);
mlir::DictionaryAttr EncodeBoundDependencyTable(mlir::OpBuilder& builder,
    analysis::DependencyTable const& table, analysis::CouplingRelation const& producer_coordinates,
    analysis::CouplingRelation const& consumer_coordinates, analysis::ParamBinding const& binding);
std::optional<analysis::DependencyTable> ReadBoundDependencyTable(mlir::Operation* coupling,
    analysis::ParamBinding const& binding);
}  // namespace tilemega::dialect
