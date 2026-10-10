// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/MoeRegionPattern.h>

namespace tilemega::frontend {
struct MoeRegionOptions {
  unsigned tokens=1;
  bool grouped=false;
  unsigned block_rows=16;
  unsigned router_tile_n=64;
  unsigned combine_token_tile=1,combine_channel_tile=128;
};
ModelPlan BuildMoeRegion(std::vector<FxNodeRecord> const&,
    std::vector<SignatureInput> const&,std::vector<std::string> const&,
    MoeRegionOptions const& = {});
void AppendMoeBlock(ModelPlan&,MoeRegionMatch const&,
    std::vector<FxNodeRecord> const&,std::vector<SignatureInput> const&,
    unsigned input,unsigned output,MoeRegionOptions const&,
    unsigned norm_stats=codegen::kDmNoIndex,unsigned next_norm_stats=codegen::kDmNoIndex);
// Router partial storage follows the selected N geometry, not export geometry.
void MaterializeMoeRegionStorage(ModelPlan&,unsigned router_tile_n,
                                unsigned down_tile_n=0,unsigned router_gemm=codegen::kDmNoIndex);
} // namespace tilemega::frontend
