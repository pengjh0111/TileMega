// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DnnDeferredNorm.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/JSON.h>
#include <algorithm>
#include <set>
#include <stdexcept>

namespace tilemega::frontend {
namespace {
using namespace codegen;
llvm::json::Value Recipe(PlanBuffer const& buffer) {
  auto value=llvm::json::parse(buffer.pack_json);
  if(!value)throw std::invalid_argument("deferred LN requires a valid weight recipe");
  return std::move(*value);
}
std::string Json(llvm::json::Value value) {
  return llvm::formatv("{0}",std::move(value)).str();
}
std::string Source(PlanBuffer const& buffer) {
  auto recipe=Recipe(buffer);auto* object=recipe.getAsObject();
  auto source=object?object->getString("source"):std::nullopt;
  if(!object || object->getString("kind")!="alias" || !source)
    throw std::invalid_argument("deferred LN affine input must retain its parameter FQN");
  return source->str();
}
bool SameRows(DmBufferLayout const& x,DmBufferLayout const& y,unsigned width) {
  if(!x.rank || x.rank!=y.rank || x.kind!=y.kind ||
      x.logical[x.rank-1]!=width || y.logical[y.rank-1]!=width ||
      x.strides[x.rank-1]!=1 || y.strides[y.rank-1]!=1)return false;
  return std::equal(std::begin(x.logical),std::end(x.logical),std::begin(y.logical));
}
}
unsigned ApplyDnnDeferredLayerNorm(ModelPlan& destination) {
  if(!destination.dm || !destination.forward || destination.forward_token_axis)
    throw std::invalid_argument("deferred DNN normalization requires a DNN forward plan");
  auto plan=destination;std::set<unsigned> removed;unsigned count=0;
  auto weight=[&](std::string name,llvm::json::Value recipe,unsigned elements) {
    PlanBuffer b;b.name=std::move(name);b.dtype="f32";b.source=PlanBuffer::Source::kWeight;
    b.role="external";b.external_name=b.name;b.constant=elements;b.pack_json=Json(std::move(recipe));
    unsigned id=plan.buffers.size();plan.buffers.push_back(std::move(b));return id;
  };
  for(unsigned index=0;index<plan.stages.size();++index) {
    auto const norm=plan.stages[index];if(norm.kind!=PlanTaskKind::kLayerNorm)continue;
    unsigned x=norm.operands[0],y=norm.operands[3],width=norm.width;
    if(!SameRows(plan.buffers.at(x).layout,plan.buffers.at(y).layout,width) ||
        std::any_of(plan.outputs.begin(),plan.outputs.end(),[&](auto const& o){return o.buffer==y;}))continue;
    unsigned producer=kDmNoIndex;
    for(unsigned before=0;before<index;++before) {
      auto const& s=plan.stages[before];
      if(s.kind==PlanTaskKind::kGemm && plan.gemms.at(s.gemm).d==x)producer=before;
      if(s.kind==PlanTaskKind::kEmbeddingSum && s.operands[5]==x)producer=before;
    }
    if(producer==kDmNoIndex)continue;
    auto& producing=plan.stages[producer];
    if(producing.kind==PlanTaskKind::kGemm) {
      auto const& g=plan.gemms.at(producing.gemm);
      if(g.access.write.kind!=DmWriteKind::kDense || g.n!=width || g.chain.side_count==5 ||
          std::any_of(std::begin(g.chain.operations),std::begin(g.chain.operations)+g.chain.count,
              [](auto const& op){return op.kind==DmEpilogueKind::kGatePair;}))continue;
    }
    std::vector<unsigned> consumers;bool valid=true;
    for(unsigned after=index+1;after<plan.stages.size();++after) {
      auto const& s=plan.stages[after];
      if(s.kind!=PlanTaskKind::kGemm) {
        for(auto id:s.operands)if(id==y)valid=false;
        continue;
      }
      auto const& g=plan.gemms.at(s.gemm);bool uses=g.a==y;
      if(g.c==y && g.c!=g.d)valid=false;
      if(g.a==y) {
        // Equality of logical rows plus a complete channel contraction is the
        // access proof: each dot reads exactly LN's one row, without a spatial
        // window, gather, row selection, channel gate or broadcast scale.
        bool pointwise=g.access.a==DmAAccess::kDense;
        if(g.access.a==DmAAccess::kIm2Col && g.access.conv<plan.convolutions.size()) {
          auto const& c=plan.convolutions[g.access.conv];
          pointwise=c.r==1 && c.s==1 && c.stride_h==1 && c.stride_w==1 &&
              !c.pad_h && !c.pad_w && c.c==width;
        }
        valid&=pointwise && g.k==width && !g.access.a_row_offset &&
            g.access.a_row_stride<=1 && g.access.a_scale==kDmNoIndex && g.chain.count<8 &&
            g.access.b==DmBAccess::kDense;
      }
      for(unsigned op=0;op<g.chain.count;++op) {
        auto const& value=g.chain.operations[op];
        for(auto id:value.parameter)if(id==y) {
          uses=true;
          valid&=value.kind==DmEpilogueKind::kResidual && value.parameter[0]==y &&
              value.parameter[1]==kDmNoIndex && value.residual_map.kind==DmWriteKind::kDense &&
              g.n==width;
        }
      }
      if(uses)consumers.push_back(s.gemm);
    }
    if(!valid || consumers.empty())continue;
    auto gamma=Source(plan.buffers.at(norm.operands[1]));
    auto beta=Source(plan.buffers.at(norm.operands[2]));
    PlanBuffer stats;stats.name=plan.buffers[x].name+".deferred_ln_stats";stats.dtype="f32";
    stats.per_batch=norm.rows_per_batch*2;stats.layout.rank=2;
    stats.layout.logical[0]=stats.layout.physical[0]=plan.buffers[x].layout.logical[0];
    if(plan.buffers[x].layout.kind==DmLayout::kNHWC)
      stats.layout.logical[0]=stats.layout.physical[0]*=norm.rows_per_batch;
    stats.layout.logical[1]=stats.layout.physical[1]=2;
    stats.layout.strides[0]=2;stats.layout.strides[1]=1;
    unsigned stat_id=plan.buffers.size();plan.buffers.push_back(std::move(stats));
    if(producing.kind==PlanTaskKind::kEmbeddingSum)producing.operands[6]=stat_id;
    else {
      auto& chain=plan.gemms[producing.gemm].chain;
      chain.side[chain.side_count++]={DmSideOutputKind::kRowStats,stat_id,kDmNoIndex,0};
    }
    bool image=plan.buffers.at(x).layout.kind==DmLayout::kNHWC;
    unsigned gamma_fp32=kDmNoIndex,beta_fp32=kDmNoIndex;
    for(unsigned consumer:consumers) {
      auto& g=plan.gemms[consumer];
      if(g.a==y) {
        unsigned bias=kDmNoIndex;
        if(g.chain.count && g.chain.operations[0].kind==DmEpilogueKind::kBias)
          bias=g.chain.operations[0].parameter[0];
        auto folded=[&](char const* part) {
          llvm::json::Object recipe{{"kind","fold_layernorm"},{"source",Recipe(plan.buffers[g.b])},
              {"gamma",gamma},{"beta",beta},{"part",part}};
          if(image)recipe["channel_axis"]=1;
          if(bias!=kDmNoIndex)recipe["bias"]=Recipe(plan.buffers[bias]);
          return llvm::json::Value(std::move(recipe));
        };
        auto u=weight(plan.buffers[g.b].name+".ln_u",folded("u"),g.n);
        auto v=weight(plan.buffers[g.b].name+".ln_v",folded("v"),g.n);
        plan.buffers[g.b].pack_json=Json(folded("weight"));g.a=x;
        if(g.access.a==DmAAccess::kIm2Col)plan.convolutions[g.access.conv].input_layout=x;
        if(bias!=kDmNoIndex) {
          for(unsigned op=1;op<g.chain.count;++op)g.chain.operations[op-1]=g.chain.operations[op];
          --g.chain.count;
        }
        for(unsigned op=g.chain.count;op>0;--op)g.chain.operations[op]=g.chain.operations[op-1];
        ++g.chain.count;auto& op=g.chain.operations[0];op={};op.kind=DmEpilogueKind::kDeferredLayerNorm;
        op.parameter[0]=stat_id;op.parameter[1]=u;op.parameter[2]=v;
        op.norm_width=width;op.norm_epsilon=norm.norm_epsilon;
      }
      for(unsigned operation=0;operation<g.chain.count;++operation) {
        auto& op=g.chain.operations[operation];
        if(op.kind!=DmEpilogueKind::kResidual || op.parameter[0]!=y)continue;
        if(gamma_fp32==kDmNoIndex) {
          llvm::json::Object gamma_recipe{{"kind","linear_bias"},{"source",gamma}};
          llvm::json::Object beta_recipe{{"kind","linear_bias"},{"source",beta}};
          if(image){gamma_recipe["channel_axis"]=1;beta_recipe["channel_axis"]=1;}
          gamma_fp32=weight(plan.buffers[x].name+".ln_gamma",std::move(gamma_recipe),width);
          beta_fp32=weight(plan.buffers[x].name+".ln_beta",std::move(beta_recipe),width);
        }
        op.kind=DmEpilogueKind::kResidualLN;op.parameter[0]=x;op.parameter[1]=stat_id;
        op.parameter[2]=gamma_fp32;op.parameter[3]=beta_fp32;op.residual_map.layout=x;
        op.norm_width=width;op.norm_epsilon=norm.norm_epsilon;
      }
    }
    for(auto& [name,id]:plan.node_buffer)if(id==y)id=x;
    removed.insert(index);++count;
  }
  std::vector<PlanStage> stages;
  for(unsigned index=0;index<plan.stages.size();++index)
    if(!removed.count(index))stages.push_back(std::move(plan.stages[index]));
  plan.stages=std::move(stages);ValidateDmModelPlan(plan);
  destination=std::move(plan);return count;
}
}
