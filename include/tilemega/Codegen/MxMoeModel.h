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
struct MxMoeExternalBuffer {
  unsigned original_index;
  frontend::PlanBuffer buffer;
};
struct MxMoeModel {
  unsigned batch, capacity, hidden, heads, hkv, dim, queries, experts, top_k, intermediate, vocabulary;
  unsigned tokens, embed, cosine, sine, head_gemm;
  int token_id_bits;
  double epsilon;
  bool routing_weights_fp32, rope_fp32_phase;
  std::vector<MxMoeLayer> layers;
  std::vector<MxMoeExternalBuffer> external_buffers;
};
MxMoeModel AnalyzeMxMoe(frontend::ModelPlan const& plan);
}
