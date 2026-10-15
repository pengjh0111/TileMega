// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Solver/BackendCostQuery.h>
#include <array>
#include <stdexcept>

namespace tilemega::solver {
inline int ServingGemmFamily(frontend::PlanGemm const& g) {
  using Op=frontend::PlanGemm::Epilogue;
  return g.epilogue==Op::kArgmaxPartial?4:g.epilogue==Op::kSwiGLU?2:
      g.epilogue==Op::kResidual?(g.k>g.n?3:1):0;
}
// Seed imports have no solver domain to apply a requested implementation.
// Reject an unsupported shape here rather than generating a different kernel.
inline void ApplyServingSeedImplementations(frontend::ModelPlan const& plan,
    std::vector<frontend::GemmGranularity>& geometry,std::array<int,5> const& requested,
    bool hopper_native,int batch,bool paged,int page_bytes) {
  if(geometry.size()!=plan.gemms.size())throw std::invalid_argument("seed implementation geometry count differs");
  for(std::size_t i=0;i<plan.gemms.size();++i) {
    auto const& g=plan.gemms[i];auto& tile=geometry[i];
    int impl=requested.at(ServingGemmFamily(g));if(impl<0)continue;
    if(impl==1 && (!ServingGemvShapeLegal(tile.tile_m,tile.tile_n,tile.tile_k,tile.stages) || paged))
      throw std::invalid_argument("requested seed GEMV has an unsupported shape or executor");
    if(impl==2 && (!hopper_native || batch<1 || batch>16 || tile.tile_m!=16 ||
        !ServingWgmmaShapeLegal(tile.tile_m,tile.tile_n,tile.tile_k,tile.stages) ||
        (paged && 2*tile.tile_n*tile.tile_k>page_bytes) ||
        g.access.a==codegen::DmAAccess::kIm2Col || g.access.a_scale!=codegen::kDmNoIndex))
      throw std::invalid_argument("requested seed WGMMA requires a legal native Hopper M16 shape and one weight page");
    if(impl>2)throw std::invalid_argument("unknown seed GEMM implementation");
    tile.impl=impl;
  }
}
} // namespace tilemega::solver
