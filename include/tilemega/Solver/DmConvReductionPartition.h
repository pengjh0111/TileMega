// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskOwnershipGeometry.h>
#include <tilemega/Backend/ConvIteration.h>
#include <tilemega/Solver/CostModel.h>
#include <limits>
#include <stdexcept>

namespace tilemega::solver {
inline void PartitionDmConvGemm(analysis::SemanticOp const& op,
    codegen::ConvDesc const& conv,codegen::DmBufferLayout const& layout,
    GemmConfig const& config,analysis::Granularity& granularity,
    analysis::ParamBinding const& known={}) {
  using namespace analysis;
  if(!op.exact_task_access || op.kind!=OperatorKind::kMatmul ||
     op.task_space.axes.size()!=2 || !op.reduction.splittable ||
     op.reduction.dim!="c" || config.tile_m<=0 || config.tile_n<=0 ||
     config.tile_k<=0 || config.split_k<=0)
    throw std::invalid_argument("convolution requires exact pixel/channel ownership");
  for(auto const& [name,extent]:std::vector<std::pair<std::string,std::uint32_t>>{
        {"c",conv.c},{"r",conv.r},{"s",conv.s}}) {
    auto dim=op.Dim(name);
    if(!dim || dim->type!=IteratorType::kReduction || !dim->origin.IsLiteral(0) ||
       dim->BoundExtent().Eval(known,{})!=extent)
      throw std::invalid_argument("convolution L-sem differs from its reduction descriptor");
  }
  auto geometry=backend::ConvIterationGeometry::Build(conv,layout,config.tile_k);
  if(geometry.iterations>std::uint64_t(std::numeric_limits<long>::max()) ||
     geometry.iterations%config.split_k)
    throw std::invalid_argument("convolution split must divide issued K iterations");
  auto fixed=[](std::uint64_t value) {return ClosedForm::Constant(static_cast<long>(value));};
  TaskReductionIndex partition;
  auto cp=geometry.channels,tk=geometry.tile_k;
  if(cp>=tk) {
    auto blocks=(std::uint64_t(cp)+tk-1)/tk;
    partition.index.terms={{"r",fixed(geometry.columns*blocks),fixed(1)},
        {"s",fixed(blocks),fixed(1)},{"c",fixed(1),fixed(tk)}};
  } else {
    partition.index.terms={{"r",fixed(std::uint64_t(geometry.columns)*cp),fixed(1)},
        {"s",fixed(cp),fixed(1)},{"c",fixed(1),fixed(1)}};
    partition.index.outer_divisor=fixed(tk);
  }
  partition.capacity=fixed(geometry.iterations);partition.issued_width=fixed(tk);
  granularity.IndexReduction(op.name,std::move(partition));
  granularity.Tile(op.name,UnitTaskOwnershipDimension(op,0),fixed(config.tile_m));
  auto columns=op.task_space.axes[1].extent.Eval(known,known);
  if(columns<=0 || conv.k%columns || (conv.k/columns!=1 && conv.k/columns!=2) ||
     config.tile_n%(conv.k/columns))
    throw std::invalid_argument("convolution output contraction differs from its N tile");
  granularity.Tile(op.name,UnitTaskOwnershipDimension(op,1),fixed(config.tile_n/(conv.k/columns)));
  if(config.split_k>1)granularity.Split(op.name,fixed(geometry.iterations/config.split_k));
}
} // namespace tilemega::solver
