// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/ModelPlan.h>
#include <cstdint>
#include <vector>
namespace tilemega::codegen {
// Stage and buffer indices refer to the original frontend ModelPlan.
struct MxMoeLayer {
  unsigned qkv, attention, o, router, dispatch, gate_up, down, combine;
  unsigned q_norm, k_norm, k_cache, v_cache;
  DmEpilogueChain gate_chain;
};
enum class MxMoeSlotV2 {
  kUnused, kTokens, kEmbed, kCosine, kSine, kHead,
  kQkv, kO, kRouter, kGateUp, kDown, kQNorm, kKNorm, kKCache, kVCache
};
struct MxMoeWeightTilesV2 {
  unsigned qkv, o, router, gate_up, down, head;
};
struct MxMoeExternalBuffer {
  unsigned original_index;
  frontend::PlanBuffer buffer;
  MxMoeSlotV2 slot = MxMoeSlotV2::kUnused;
  unsigned layer = 0;
};
struct MxMoeModel {
  unsigned batch, capacity, hidden, heads, hkv, dim, queries, experts, top_k, intermediate, vocabulary;
  unsigned tokens, embed, cosine, sine, head_gemm;
  int token_id_bits;
  double epsilon;
  bool routing_weights_fp32, rope_fp32_phase;
  bool row_qk_norm = false, softmax_all = false;
  std::vector<MxMoeLayer> layers;
  std::vector<MxMoeExternalBuffer> external_buffers;
};
MxMoeModel AnalyzeMxMoe(frontend::ModelPlan const& plan);
int MxMoeExpectedActiveV2(unsigned experts,unsigned batch);
std::vector<MxMoeExternalBuffer> PrepareMxMoeBuffersV2(
    frontend::ModelPlan const& plan, MxMoeWeightTilesV2 const& tiles);
}
