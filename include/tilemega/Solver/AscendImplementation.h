// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/BackendCostQuery.h>
namespace tilemega::solver {
inline BackendTraits As1Bf16GemmTraits() {
  BackendTraits t;
  t.tile_m=32;t.tile_n=64;t.tile_k=64;t.stages=2;
  t.shape_legal=true;t.arch_tag="ascend_3510";
  ascend::TaskImplementation impl;
  impl.impl_id="catlass_950_bf16_32x64x64_gm";
  impl.required_role=ascend::ImplementationRole::kAicOnly;
  impl.input_layout="A:ND_MK_paddedK16;B:ND_KN_paddedK16;logical_weight:NK";
  impl.output_layout="ND_MN_bf16";
  // CATLASS Arch::Resource reserves entire AIC banks, even for this small
  // tile. These are source-declared buffers, not a guessed occupancy model.
  impl.aic_scratch={524288,65536,65536,262144,0,0,0};
  t.ascend_implementation=impl;
  return t;
}
} // namespace tilemega::solver
