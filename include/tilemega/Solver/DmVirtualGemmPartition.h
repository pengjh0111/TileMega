// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskOwnershipGeometry.h>
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Solver/CostModel.h>
#include <stdexcept>

namespace tilemega::solver {
// A binding block and an MMA row tile are independent coordinates. Keeping
// the virtual id separate prevents a row tile from spanning two experts.
inline void PartitionDmVirtualGemm(analysis::SemanticOp const& op,
    codegen::DmGemmAccess const& access,GemmConfig const& config,
    analysis::Granularity& granularity,analysis::ParamBinding const& known={}) {
  using namespace analysis;
  if(!op.exact_task_access || op.kind!=OperatorKind::kMatmul ||
     access.b!=codegen::DmBAccess::kExpertIndirect || !access.block_rows ||
     op.task_space.axes.size()!=3 || op.task_map.results.size()!=3 ||
     config.tile_m<=0 || config.tile_n<=0 || config.tile_k<=0 || config.split_k<=0)
    throw std::invalid_argument("virtual GEMM requires three exact ownership axes");
  auto const& virtual_dim=UnitTaskOwnershipDimension(op,0);
  auto const& row_dim=UnitTaskOwnershipDimension(op,1);
  auto const& column_dim=UnitTaskOwnershipDimension(op,2);
  if(virtual_dim==row_dim || virtual_dim==column_dim || row_dim==column_dim)
    throw std::invalid_argument("virtual GEMM ownership axes are not independent");
  auto const& v=*op.Dim(virtual_dim);auto const& row=*op.Dim(row_dim);
  auto const& column=*op.Dim(column_dim);
  auto capacity=v.BoundExtent().Substitute(known);
  if(!v.runtime || !v.capacity || v.binding_source.empty() ||
     (v.binding_requirement!="prefix_sum" && v.binding_requirement!="tensor_values") ||
     v.type!=IteratorType::kParallel || row.type!=IteratorType::kParallel ||
     column.type!=IteratorType::kParallel || row.runtime || column.runtime ||
     row.BoundExtent().Eval(known,{})!=access.block_rows ||
     (capacity.IsConstant() && capacity.Eval({},{})<=0))
    throw std::invalid_argument("virtual GEMM binding capacity or row block is inconsistent");
  int factor=op.arithmetic=="swiglu_gemm"?2:1;
  if(config.tile_n%factor)
    throw std::invalid_argument("virtual GEMM packed N tile splits a gate pair");
  granularity.Tile(op.name,virtual_dim,ClosedForm::Constant(1));
  granularity.Tile(op.name,row_dim,ClosedForm::Constant(config.tile_m));
  granularity.Tile(op.name,column_dim,ClosedForm::Constant(config.tile_n/factor));
}
} // namespace tilemega::solver
