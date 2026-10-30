// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/MxMoeProgramV2.h>

namespace tilemega::codegen {
struct MxMoeConfigV3 {
  MxDecodeClassV2 qkv,o,router,gate_up,down,head;
  int threads=512, b_min=1, attn_prefetch=0, split=1;
};
void CheckMxMoeConfigV3(frontend::ModelPlan const& plan,MxMoeConfigV3 const& config);
int MxMoeBodySmemBytesV3(frontend::ModelPlan const& plan,MxMoeConfigV3 const& config);
std::string EmitMxMoeProgramV3(frontend::ModelPlan const& plan,MxMoeConfigV3 const& config,
                              bool prompt,int past_lo=64,int past_hi=1086);
}
