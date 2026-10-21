// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/BackendCostQuery.h>
#include <tilemega/Support/Json.h>
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
inline BackendTraits As2Bf16GemmTraits(unsigned implementation) {
  if(implementation>1)throw std::invalid_argument("Unknown native BF16 GEMM implementation");
  auto t=As1Bf16GemmTraits();
  if(implementation){t.tile_m=16;t.tile_n=128;t.tile_k=128;}
  t.alignment={16,16};
  auto& impl=*t.ascend_implementation;
  impl.impl_id=implementation?"as2_catlass_bf16_16x128x128_nk":"as2_catlass_bf16_32x64x64_nk";
  impl.input_layout="A:ND_MK;B:logical_NK_as_column_major_KN;N,K multiples of 16";
  return t;
}
inline ascend::TaskImplementation As1AivImplementation(codegen::TaskKind kind,TargetSpec const& target) {
  auto config=json::ParseFile(std::string(TILEMEGA_SOURCE_DIR)+"/configs/ascend/as1_impl_bindings.json");
  std::string id;
  if(kind==codegen::TaskKind::kRMSNorm)id="aiv_950_bf16_rmsnorm_row";
  else if(kind==codegen::TaskKind::kElementwise)id="aiv_950_bf16_silu_row";
  else if(kind==codegen::TaskKind::kAdd)id="aiv_950_bf16_add_tile";
  else throw std::invalid_argument("Unimplemented AIV TaskKind");
  bool bound=false;
  for(auto const& b:config.At("bindings").AsArray("bindings")) {
    if(!b.At("enabled").AsBool("enabled") || b.At("impl_id").AsString("impl_id")!=id)continue;
    if(bound || b.At("task_kind").AsNumber("task_kind")!=int(kind) ||
        b.At("required_role").AsString("required_role")!="AIV_ONLY" ||
        b.At("event_triggers").AsNumber("event_triggers")!=1 ||
        b.At("ub_bytes").AsNumber("ub_bytes")!=6656 || b.At("dtype").AsString("dtype")!="bf16")
      throw std::invalid_argument("AIV binding disagrees with the implemented body");
    bound=true;
  }
  if(!bound || !target.ascend_capability)throw std::invalid_argument("Missing enabled native AIV binding");
  ascend::TaskImplementation impl;impl.kind=kind;impl.impl_id=id;
  impl.required_role=ascend::ImplementationRole::kAivOnly;
  impl.input_layout=impl.output_layout="ND_bf16";impl.aiv_scratch.ub=6656;
  impl.Validate(*target.ascend_capability);return impl;
}
inline ascend::TaskImplementation As1EncoderImplementation(codegen::TaskKind kind,TargetSpec const& target) {
  std::string id;
  if(kind==codegen::TaskKind::kLayerNorm)id="aiv_950_bf16_layernorm_row";
  else if(kind==codegen::TaskKind::kElementwise)id="aiv_950_bf16_erf_gelu_row";
  else if(kind==codegen::TaskKind::kEncoderAttention)id="aiv_950_bf16_encoder_attention_q1";
  else if(kind==codegen::TaskKind::kAdd)id="aiv_950_bf16_encoder_add_tile";
  else throw std::invalid_argument("Unimplemented encoder AIV TaskKind");
  auto config=json::ParseFile(std::string(TILEMEGA_SOURCE_DIR)+"/configs/ascend/as1_impl_bindings.json");
  bool bound=false;
  for(auto const& b:config.At("bindings").AsArray("bindings")) {
    if(b.At("impl_id").AsString("impl_id")!=id || !b.At("enabled").AsBool("enabled"))continue;
    if(bound || b.At("task_kind").AsNumber("task_kind")!=int(kind) ||
       b.At("required_role").AsString("required_role")!="AIV_ONLY" ||
       b.At("ub_bytes").AsNumber("ub_bytes")!=16384 || b.At("dtype").AsString("dtype")!="bf16" ||
       b.At("event_triggers").AsNumber("event_triggers")!=1)
      throw std::invalid_argument("Encoder binding disagrees with its native body");
    bound=true;
  }
  if(!bound || !target.ascend_capability)throw std::invalid_argument("Missing enabled encoder implementation: "+id);
  ascend::TaskImplementation impl;impl.kind=kind;impl.impl_id=id;
  impl.required_role=ascend::ImplementationRole::kAivOnly;
  impl.input_layout=impl.output_layout="ND_bf16; mask:i64";impl.aiv_scratch.ub=16384;
  impl.Validate(*target.ascend_capability);return impl;
}
} // namespace tilemega::solver
