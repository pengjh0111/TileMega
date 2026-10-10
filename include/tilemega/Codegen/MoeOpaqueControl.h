// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tilemega::codegen {
struct OpaqueMoeStage {
  bool gemm=false,norm=false,dispatch=false,combine=false;
  std::uint32_t gemm_id=0,router_gemm=0;
};
struct OpaqueMoeRegion {std::uint32_t first=0,last=0;};

// The control surrounds the router's optional explicit norm, dispatch,
// expert tasks and combine. Attention and other decoder work stay outside.
inline std::vector<OpaqueMoeRegion> FindOpaqueMoeRegions(
    std::vector<OpaqueMoeStage> const& stages) {
  std::vector<OpaqueMoeRegion> regions;
  std::uint32_t first=~std::uint32_t(0),router=first;
  for(std::uint32_t i=0;i<stages.size();++i) {
    auto const& stage=stages[i];
    if(stage.dispatch) {
      if(first!=~std::uint32_t(0)) {
        if(router!=stage.router_gemm)throw std::invalid_argument("overlapping opaque MoE regions");
        continue;
      }
      std::uint32_t found=i;
      while(found && !(stages[found-1].gemm && stages[found-1].gemm_id==stage.router_gemm))--found;
      if(!found)throw std::invalid_argument("opaque MoE dispatch has no preceding router");
      first=found-1;router=stage.router_gemm;
      if(first && stages[first-1].norm)--first;
      if(!regions.empty() && first<=regions.back().last)
        throw std::invalid_argument("opaque MoE regions overlap");
    }
    if(stage.combine) {
      if(first==~std::uint32_t(0) || router!=stage.router_gemm)
        throw std::invalid_argument("opaque MoE combine has no matching dispatch");
      regions.push_back({first,i});first=router=~std::uint32_t(0);
    }
  }
  if(first!=~std::uint32_t(0) || regions.empty())
    throw std::invalid_argument("opaque MoE control requires complete regions");
  return regions;
}
} // namespace tilemega::codegen
