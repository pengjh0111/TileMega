// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/MxMoeModel.h>
#include <tilemega/Codegen/MxDecodeProgramV2.h>
namespace tilemega::codegen {
struct MxMoeConfigV2 {
  MxDecodeClassV2 qkv, o, router, gate_up, down, head;
  int threads=576, b_min=1, attn_prefetch=0;
};
void CheckMxMoeConfigV2(frontend::ModelPlan const& plan,MxMoeConfigV2 const& config);
std::string EmitMxMoeProgramV2(frontend::ModelPlan const& plan,MxMoeConfigV2 const& config,
                              bool prompt,int past_lo=64,int past_hi=1086);
}
