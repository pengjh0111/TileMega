// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskOwnershipGeometry.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <stdexcept>

namespace tilemega::dialect {
inline bool ApplyTaskReductionGeometry(mlir::DictionaryAttr attributes,
    analysis::SemanticOp const& op,analysis::Granularity& geometry) {
  auto index=attributes.get("reduction_index"),chunk=attributes.get("reduction_chunk");
  if(!index) {
    if(chunk)throw std::invalid_argument("indexed chunk has no reduction index");
    return false;
  }
  auto text=mlir::dyn_cast<mlir::StringAttr>(index);
  if(!text)throw std::invalid_argument("task reduction index must be a string payload");
  geometry.IndexReduction(op.name,analysis::DecodeTaskReductionIndex(text.getValue().str()));
  if(chunk) {
    auto value=mlir::dyn_cast<mlir::StringAttr>(chunk);
    if(!value)throw std::invalid_argument("indexed chunk must be a closed form");
    auto extent=analysis::ClosedForm::Parse(value.getValue().str());
    if(extent.IsConstant() && extent.Eval({},{})<=0)
      throw std::invalid_argument("indexed chunk must be positive");
    geometry.Split(op.name,std::move(extent));
  }
  return true;
}
inline void VerifyTaskReductionGeometry(mlir::DictionaryAttr attributes,
    analysis::SemanticOp const& op) {
  analysis::Granularity geometry;
  if(!ApplyTaskReductionGeometry(attributes,op,geometry))return;
  for(unsigned i=0;i<op.task_space.axes.size();++i) {
    auto tile=attributes.getAs<mlir::StringAttr>(op.task_space.axes[i].name);
    if(!tile)throw std::invalid_argument("indexed task lacks its ownership tile");
    geometry.Tile(op.name,analysis::UnitTaskOwnershipDimension(op,i),
                  analysis::ClosedForm::Parse(tile.getValue().str()));
  }
  (void)analysis::Instantiate({{op}},geometry);
}
} // namespace tilemega::dialect
