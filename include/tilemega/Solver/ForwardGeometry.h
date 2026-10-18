// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/ServingSeedGeometry.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <cmath>
#include <limits>

namespace tilemega::solver {
// The complete indexed table prevents a model revision from silently inheriting
// a partial tuning table for different stages.
inline void ApplyForwardGeometry(frontend::ModelPlan const& plan,
    std::vector<frontend::GemmGranularity>& geometry,json::Value const& value) {
  auto const& rows=value.At("gemms").AsArray("forward geometry GEMMs");
  if(rows.size()!=plan.gemms.size() || geometry.size()!=rows.size())
    throw std::invalid_argument("forward geometry must cover every GEMM");
  for(auto const& [key,item]:value.AsObject("forward geometry"))
    if(key!="gemms" && key!="comment")
      throw std::invalid_argument("unknown forward geometry field: "+key);
  auto integer=[](json::Value const& v,char const* key) {
    double n=v.At(key).AsNumber(key);
    if(!std::isfinite(n) || n<0 || n>std::numeric_limits<int>::max() || std::floor(n)!=n)
      throw std::invalid_argument(std::string("invalid forward geometry integer: ")+key);
    return int(n);
  };
  auto proposed=geometry;std::vector<bool> seen(rows.size(),false);
  for(auto const& row:rows) {
    int index=integer(row,"index");
    if(index>=int(rows.size()) || seen[index])
      throw std::invalid_argument("duplicate or out-of-range forward GEMM index");
    seen[index]=true;
    for(auto const& [key,item]:row.AsObject("forward GEMM geometry"))
      if(key!="index" && key!="tile_m" && key!="tile_n" && key!="tile_k" &&
         key!="stages" && key!="split_k" && key!="impl")
        throw std::invalid_argument("unknown forward GEMM geometry field: "+key);
    frontend::GemmGranularity tile{integer(row,"tile_m"),integer(row,"tile_n"),
        integer(row,"tile_k"),integer(row,"stages"),integer(row,"split_k"),
        SeedGemmImpl(row.At("impl").AsString("impl"))};
    bool legal=tile.impl==0?(tile.stages<=8 &&
        DmServingBF16ShapeLegal(tile.tile_m,tile.tile_n,tile.tile_k,tile.stages)):
        tile.impl==2 && ServingWgmmaShapeLegal(tile.tile_m,tile.tile_n,tile.tile_k,tile.stages);
    auto const& g=plan.gemms[index];
    if(!legal || tile.split_k<1 || (tile.split_k>1 &&
        (std::uint64_t(tile.tile_k)*tile.split_k>g.k || g.k%(std::uint64_t(tile.tile_k)*tile.split_k))))
      throw std::invalid_argument("illegal forward tile or split at GEMM "+std::to_string(index));
    if(g.access.b==codegen::DmBAccess::kExpertIndirect)
      throw std::invalid_argument("forward geometry table does not admit expert-indirect tasks");
    proposed[index]=tile;
  }
  geometry.swap(proposed);
}
} // namespace tilemega::solver
