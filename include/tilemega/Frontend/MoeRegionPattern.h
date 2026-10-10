// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/ModelPlan.h>

namespace tilemega::frontend {
struct MoeRegionMatch {
  std::string input,normalized,router,topk,indices,weights,experts,output;
  std::string norm_weight,router_weight,gate_up_weight,down_weight;
  unsigned hidden=0,intermediate=0,expert_count=0,top_k=0;
  double epsilon=0;
};

// Preserve tuple selectors and dtype boundaries while proving the entire
// h + MoE(RMSNorm(h)) region. Generic Value() deliberately skips getitem.
MoeRegionMatch MatchMoeRegion(std::vector<FxNodeRecord> const&,
    std::vector<SignatureInput> const&,std::vector<std::string> const&);
std::vector<MoeRegionMatch> FindDecoderMoeBlocks(std::vector<FxNodeRecord> const&,
    std::vector<SignatureInput> const&);
} // namespace tilemega::frontend
