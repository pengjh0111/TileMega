// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <algorithm>
#include <stdexcept>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Analysis/TaskOwnershipGeometry.h>
#include <tilemega/Analysis/TaskElementRelation.h>

namespace tilemega::solver {
// Mirrors ServingGemv::Supported after virtual binding resolution. A
// conditioned profile narrows the row domain while ownership stays static.
inline bool UsesDmGemv(DerivedTaskInput const& input,BackendTraits const& traits,
    analysis::ParamBinding const& theta,analysis::ParamBinding const& coordinate) {
  if(!input.serving_gemv || !input.task.element_access || traits.tile_n>32 ||
      traits.tile_k<32 || traits.stages<2)return false;
  auto const& op=input.task.element_access->semantic;
  unsigned row_axis=op.task_space.axes.size()==3?1:0;
  if(op.kind!=analysis::OperatorKind::kMatmul || op.task_space.axes.size()<2 ||
      op.task_space.axes.size()>3)return false;
  auto const& row_name=analysis::UnitTaskOwnershipDimension(op,row_axis);
  auto const* row=op.Dim(row_name);
  if(!row || row->runtime || !row->origin.IsLiteral(0))
    throw std::invalid_argument("GEMV pricing lacks a bounded row domain");
  long begin=input.task.IsTiled(row_axis)?coordinate.At(input.task.output.axes[row_axis].name)*traits.tile_m:0;
  long remaining=row->BoundExtent().Eval(theta,{})-begin;
  return remaining>0 && remaining<=std::min(traits.tile_m,4);
}
} // namespace tilemega::solver
