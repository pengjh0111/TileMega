// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DnnDwPwFusion.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <algorithm>
#include <stdexcept>
namespace tilemega::frontend {
unsigned ApplyDnnDwPwFusion(ModelPlan& destination,std::set<unsigned> const* selected) {
  using namespace codegen;
  if(!destination.dm || !destination.forward || destination.forward_token_axis)
    throw std::invalid_argument("dw-pw fusion requires a stateless DNN plan");
  auto plan=destination;std::set<unsigned> removed;unsigned count=0;
  for(unsigned producer=0;producer<plan.stages.size();++producer) {
    auto dw=plan.stages[producer];
    if(dw.kind!=PlanTaskKind::kDepthwiseConv || dw.chain.side_count ||
        dw.operands[3]!=kDmNoIndex || dw.conv>=plan.convolutions.size())continue;
    auto const& conv=plan.convolutions[dw.conv];
    if(conv.c%8 || conv.c!=conv.k ||
        std::any_of(std::begin(dw.chain.operations),std::begin(dw.chain.operations)+dw.chain.count,
          [](auto const& op){return op.kind==DmEpilogueKind::kGatePair;}))continue;
    auto intermediate=dw.operands[2];unsigned consumers=0,fused=0;
    bool observed=std::any_of(plan.outputs.begin(),plan.outputs.end(),
        [&](auto const& out){return out.buffer==intermediate;});
    for(unsigned consumer=producer+1;consumer<plan.stages.size();++consumer) {
      auto& pw=plan.stages[consumer];
      if(pw.kind!=PlanTaskKind::kGemm) {
        for(auto id:pw.operands)if(id==intermediate)observed=true;
        continue;
      }
      auto& gemm=plan.gemms.at(pw.gemm);
      bool main=gemm.a==intermediate;
      if(main)++consumers;
      if(gemm.c==intermediate && gemm.c!=gemm.d)observed=true;
      for(unsigned op=0;op<gemm.chain.count;++op)
        for(auto id:gemm.chain.operations[op].parameter)if(id==intermediate)observed=true;
      if(!main || (selected && !selected->count(pw.gemm)) || gemm.k!=conv.c ||
          gemm.access.b!=DmBAccess::kDense || gemm.access.a_scale!=kDmNoIndex ||
          gemm.access.a_row_offset || gemm.access.a_row_stride>1 ||
          gemm.access.rows_per_batch!=conv.p*conv.q)continue;
      bool contiguous=gemm.access.a==DmAAccess::kDense;
      if(gemm.access.a==DmAAccess::kIm2Col && gemm.access.conv<plan.convolutions.size()) {
        auto const& pointwise=plan.convolutions[gemm.access.conv];
        contiguous=pointwise.r==1 && pointwise.s==1 && pointwise.stride_h==1 &&
            pointwise.stride_w==1 && !pointwise.pad_h && !pointwise.pad_w &&
            pointwise.h==conv.p && pointwise.w==conv.q && pointwise.c==conv.c;
      }
      if(!contiguous)continue;
      pw.kind=PlanTaskKind::kDwPwFused;pw.conv=dw.conv;pw.chain=dw.chain;
      pw.operands=dw.operands;pw.operands[2]=kDmNoIndex;
      pw.rows_per_batch=conv.p*conv.q;pw.width=conv.c;
      gemm.access.a=DmAAccess::kDense;gemm.access.conv=kDmNoIndex;
      ++fused;++count;
    }
    // Other users still receive the ordinary depthwise materialization.
    // Fused N tiles always recompute privately and never duplicate that store.
    if(fused && fused==consumers && !observed)removed.insert(producer);
  }
  std::vector<PlanStage> stages;
  for(unsigned index=0;index<plan.stages.size();++index)
    if(!removed.count(index))stages.push_back(std::move(plan.stages[index]));
  plan.stages=std::move(stages);ValidateDmModelPlan(plan);
  destination=std::move(plan);return count;
}
}
