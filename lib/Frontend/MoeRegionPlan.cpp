// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/MoeRegionPlan.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Codegen/MoeBinding.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/JSON.h>
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

namespace tilemega::frontend {
namespace {
using namespace codegen;
unsigned Checked(std::uint64_t value) {
  if(value>std::numeric_limits<unsigned>::max())
    throw std::invalid_argument("MoE region buffer exceeds descriptor capacity");
  return unsigned(value);
}
void Matrix(PlanBuffer& b,unsigned rows,unsigned columns) {
  b.constant=Checked(std::uint64_t(rows)*columns);b.per_seq=b.per_batch=0;
  b.layout={};b.layout.rank=2;
  b.layout.logical[0]=b.layout.physical[0]=rows;
  b.layout.logical[1]=b.layout.physical[1]=columns;
  b.layout.strides[0]=columns;b.layout.strides[1]=1;
}
}

static ModelPlan BuildMatchedMoeRegion(MoeRegionMatch const& match,std::vector<FxNodeRecord> const& nodes,
    std::vector<SignatureInput> const& inputs,
    MoeRegionOptions const& options) {
  if(!options.tokens || options.tokens>4096 || !options.router_tile_n ||
      (options.grouped && options.block_rows!=16 && options.block_rows!=32 &&
       options.block_rows!=64 && options.block_rows!=128) ||
      !options.combine_token_tile || options.combine_token_tile>128 ||
      options.combine_channel_tile<16 || options.combine_channel_tile>256 ||
      options.combine_channel_tile%16)
    throw std::invalid_argument("invalid MoE region geometry");
  unsigned bm=options.grouped?options.block_rows:1,capacity=0;
  if(!MoeVirtualCapacity(options.tokens,match.top_k,match.expert_count,bm,
       options.grouped,&capacity))throw std::invalid_argument("invalid MoE virtual capacity");
  ModelPlan plan;plan.dm=plan.forward=plan.forward_token_axis=true;
  plan.dtype="bf16";plan.serving_seq=options.tokens;plan.norm_epsilon=match.epsilon;
  std::map<std::string,std::string> parameters;
  std::map<std::string,int> indices;
  for(auto const& s:inputs)if(s.kind=="PARAMETER")parameters.emplace(s.name,s.target);
  for(auto const& n:nodes)indices.emplace(n.name,n.index);
  auto buffer=[&](std::string name,unsigned rows,unsigned columns,char const* dtype="bf16") {
    PlanBuffer b;b.name=std::move(name);b.dtype=dtype;Matrix(b,rows,columns);
    unsigned id=plan.buffers.size();plan.buffers.push_back(std::move(b));return id;
  };
  auto weight=[&](std::string name,unsigned count,llvm::json::Object recipe) {
    PlanBuffer b;b.name=std::move(name);b.source=PlanBuffer::Source::kWeight;
    b.constant=count;b.role="external";b.external_name=b.name;
    b.pack_json=llvm::formatv("{0}",llvm::json::Value(std::move(recipe))).str();
    unsigned id=plan.buffers.size();plan.buffers.push_back(std::move(b));return id;
  };
  auto alias=[&](std::string const& node,unsigned count) {
    auto fqn=parameters.at(node);
    return weight(fqn,count,llvm::json::Object{{"kind","alias"},{"source",fqn}});
  };
  auto stage=[&](PlanTaskKind kind,std::string const& anchor,std::initializer_list<unsigned> operands) {
    PlanStage s;s.kind=kind;s.operands.fill(kDmNoIndex);
    std::copy(operands.begin(),operands.end(),s.operands.begin());
    s.representative=anchor;s.representative_index=indices.at(anchor);
    unsigned id=plan.stages.size();plan.stages.push_back(s);return id;
  };
  unsigned t=options.tokens,h=match.hidden,i=match.intermediate,e=match.expert_count,k=match.top_k;
  auto x=buffer(match.input,t,h),y=buffer(match.output,t,h);
  for(auto id:{x,y}) {auto& b=plan.buffers[id];b.role="external";b.external_name=b.name;}
  plan.node_buffer[match.input]=x;plan.node_buffer[match.output]=y;
  auto normalized=buffer(match.normalized,t,h);
  auto gamma=alias(match.norm_weight,h);
  auto norm=stage(PlanTaskKind::kRMSNorm,match.normalized,{x,gamma,normalized});
  plan.stages[norm].width=h;
  // An external region input has no producer epilogue from which to transfer
  // RMS statistics. Full decoder regions can defer this normalization only
  // after proving the preceding residual producer's statistics access.
  plan.node_buffer[match.normalized]=normalized;
  auto logits=buffer(match.router,t,e);
  auto router_weight=alias(match.router_weight,Checked(std::uint64_t(e)*h));
  auto partial_logits=buffer(match.router+".topk_logits",t,e*k,"f32");
  auto partial_indices=buffer(match.router+".topk_indices",t,e*k,"i32");
  auto top_indices=buffer(match.indices,t,k,"i32"),top_weights=buffer(match.weights,t,k);
  auto bindings=buffer(match.experts+".bindings",capacity,4,"i32");
  auto rows=buffer(match.experts+".rows",t*k,4,"i32");
  unsigned chunks=(t+127)/128;
  auto histogram=buffer(match.experts+".histogram",chunks,e,"i32");
  auto expert_offsets=buffer(match.experts+".expert_offsets",1,e+1,"i32");
  auto block_offsets=buffer(match.experts+".block_offsets",1,e+1,"i32");
  PlanGemm router;router.n=e;router.k=h;router.a=normalized;router.b=router_weight;
  router.c=router.d=logits;router.access.write.layout=logits;
  router.chain.side_count=1;
  router.chain.side[0]={DmSideOutputKind::kTopKPartial,partial_logits,partial_indices,k};
  plan.gemms.push_back(router);
  auto route=stage(PlanTaskKind::kGemm,match.router,{});plan.stages[route].gemm=0;
  plan.node_buffer[match.router]=logits;
  DmMoeStage config;config.experts=e;config.top_k=k;config.block_rows=bm;
  config.binding_capacity=capacity;config.row_capacity=t*k;config.router_gemm=0;
  config.grouped=options.grouped;
  auto dispatch=[&](DmMoeStep step,unsigned group) {
    auto id=stage(PlanTaskKind::kMoETopK,match.topk,{partial_logits,partial_indices,
        top_indices,top_weights,bindings,rows,histogram,expert_offsets,block_offsets});
    auto& s=plan.stages[id];s.extent=e;s.width=k;s.group=group;s.moe=config;s.moe.step=step;
    return id;
  };
  unsigned binding_producer;
  if(std::uint64_t(t)*k<=4096)binding_producer=dispatch(DmMoeStep::kSelectAndDispatch,t);
  else {
    dispatch(DmMoeStep::kSelect,128);
    if(options.grouped) {dispatch(DmMoeStep::kHistogram,128);dispatch(DmMoeStep::kPrefix,128);}
    binding_producer=dispatch(DmMoeStep::kScatter,128);
  }
  plan.node_buffer[match.indices]=top_indices;plan.node_buffer[match.weights]=top_weights;
  auto prefix=parameters.at(match.gate_up_weight),down_name=parameters.at(match.down_weight);
  std::string suffix="gate_up_proj";
  if(prefix.size()<suffix.size() || prefix.substr(prefix.size()-suffix.size())!=suffix)
    throw std::invalid_argument("MoE expert weight lacks the stacked gate_up_proj FQN");
  prefix.resize(prefix.size()-suffix.size());
  if(down_name!=prefix+"down_proj")throw std::invalid_argument("MoE expert FQN namespaces disagree");
  auto expert_weight=[&](char const* part,unsigned count) {
    return weight(prefix+part+".packed",count,llvm::json::Object{{"kind","expert_stack"},
        {"prefix",prefix},{"part",part},{"experts",e},{"hidden",h},{"intermediate",i},{"u",16}});
  };
  auto gate_weight=expert_weight("gate_up",Checked(std::uint64_t(e)*2*i*h));
  auto down_weight=expert_weight("down",Checked(std::uint64_t(e)*h*i));
  auto gated=buffer(match.experts+".gated",Checked(std::uint64_t(capacity)*bm),i);
  auto partials=buffer(match.experts+".partials",t*k,h);
  auto expert=[&](unsigned a,unsigned b,unsigned d,unsigned n,unsigned inner,bool gate) {
    PlanGemm g;g.a=a;g.b=b;g.c=g.d=d;g.n=n;g.k=inner;
    auto& access=g.access;access.a=gate?DmAAccess::kRowGather:DmAAccess::kDense;
    access.b=DmBAccess::kExpertIndirect;access.binding=bindings;access.rows=rows;
    access.binding_blocks=capacity;access.binding_rows=t*k;access.experts=e;
    access.block_rows=bm;access.expert_stride=std::uint64_t(n)*inner;access.routing_topk=k;
    access.write.layout=d;
    if(gate) {
      g.chain.count=1;g.chain.operations[0].kind=DmEpilogueKind::kGatePair;
      g.chain.operations[0].gate=DmGatePair::kSwiGLU;g.chain.operations[0].unit=16;
      g.chain.operations[0].input_rounding=DmRounding::kBF16;
      g.chain.operations[0].output_rounding=DmRounding::kBF16;
    }else {access.write.kind=DmWriteKind::kRowScatter;access.write.rows=rows;}
    unsigned index=plan.gemms.size();plan.gemms.push_back(g);
    auto id=stage(PlanTaskKind::kGemm,match.experts,{});
    plan.stages[id].gemm=index;plan.stages[id].binding_producer=binding_producer;
  };
  expert(normalized,gate_weight,gated,2*i,h,true);expert(gated,down_weight,partials,h,i,false);
  unsigned stats=buffer(match.output+".row_stats",t,2*((h+options.combine_channel_tile-1)/options.combine_channel_tile),"f32");
  auto combine=stage(PlanTaskKind::kMoECombine,match.output,{partials,top_weights,x,y,stats});
  auto& s=plan.stages[combine];s.extent=h;s.width=options.combine_channel_tile;
  s.group=options.combine_token_tile;s.moe=config;s.moe.step=DmMoeStep::kCombine;
  plan.outputs.push_back({y,""});
  MaterializeMoeRegionStorage(plan,options.router_tile_n);
  ValidateDmModelPlan(plan);return plan;
}

ModelPlan BuildMoeRegion(std::vector<FxNodeRecord> const& nodes,
    std::vector<SignatureInput> const& inputs,std::vector<std::string> const& outputs,
    MoeRegionOptions const& options) {
  return BuildMatchedMoeRegion(MatchMoeRegion(nodes,inputs,outputs),nodes,inputs,options);
}

void AppendMoeBlock(ModelPlan& destination,MoeRegionMatch const& match,
    std::vector<FxNodeRecord> const& nodes,std::vector<SignatureInput> const& inputs,
    unsigned input,unsigned output,MoeRegionOptions const& options,
    unsigned norm_stats,unsigned next_norm_stats) {
  auto plan=destination;
  if(input>=plan.buffers.size() || output>=plan.buffers.size() || input==output ||
      plan.buffers[input].dtype!="bf16" || plan.buffers[output].dtype!="bf16")
    throw std::invalid_argument("MoE decoder block needs distinct BF16 input/output storage");
  auto block=BuildMatchedMoeRegion(match,nodes,inputs,options);
  // Legacy decoder activations describe their allocation with per_seq and
  // per_batch rather than a physical layout. The MoE region still needs an
  // exact row map after its external endpoints are replaced by those buffers.
  auto rows=[&](unsigned global,unsigned local) {
    auto& target=plan.buffers.at(global).layout;
    auto const& required=block.buffers.at(local).layout;
    if(!target.rank) {target=required;return;}
    if(target.kind!=required.kind || target.rank!=required.rank ||
        !std::equal(std::begin(target.logical),std::end(target.logical),std::begin(required.logical)) ||
        !std::equal(std::begin(target.physical),std::end(target.physical),std::begin(required.physical)) ||
        !std::equal(std::begin(target.strides),std::end(target.strides),std::begin(required.strides)) || target.halo_top || target.halo_bottom ||
        target.halo_left || target.halo_right)
      throw std::invalid_argument("MoE decoder endpoint requires contiguous token rows");
  };
  rows(input,block.node_buffer.at(match.input));
  rows(output,block.node_buffer.at(match.output));
  auto statistics=[&](unsigned id) {
    if(id==kDmNoIndex)return;
    if(id>=plan.buffers.size() || plan.buffers[id].dtype!="f32")
      throw std::invalid_argument("MoE decoder statistics require FP32 storage");
    auto const& layout=plan.buffers[id].layout;
    if(layout.rank!=2 || layout.logical[0]!=options.tokens ||
        layout.logical[1]!=match.hidden/32 || layout.strides[1]!=1 || match.hidden%32)
      throw std::invalid_argument("MoE decoder statistics disagree with 32-channel ownership");
  };
  statistics(norm_stats);statistics(next_norm_stats);
  if(next_norm_stats!=kDmNoIndex && options.combine_channel_tile%32)
    throw std::invalid_argument("MoE RMS output statistics require complete 32-channel groups");
  unsigned local_stats=kDmNoIndex,local_next=kDmNoIndex;
  auto binding=[&](unsigned source) {
    auto value=plan.buffers.at(source);value.role.clear();value.external_name.clear();
    unsigned local=block.buffers.size();block.buffers.push_back(std::move(value));return local;
  };
  if(norm_stats!=kDmNoIndex) {
    local_stats=binding(norm_stats);
    block.stages.erase(block.stages.begin());
    for(auto& stage:block.stages)if(stage.binding_producer!=kDmNoIndex)--stage.binding_producer;
    std::string gamma;
    for(auto const& parameter:inputs)if(parameter.name==match.norm_weight)gamma=parameter.target;
    if(gamma.empty())throw std::invalid_argument("MoE RMS fold lacks a source parameter");
    for(unsigned index:{0u,1u}) {
      auto& gemm=block.gemms.at(index);gemm.a=block.node_buffer.at(match.input);
      auto& weight=block.buffers.at(gemm.b);
      auto recipe=llvm::json::parse(weight.pack_json);
      if(!recipe)throw std::invalid_argument("invalid MoE weight recipe before RMS folding");
      weight.pack_json=llvm::formatv("{0}",llvm::json::Value(llvm::json::Object{
          {"kind","fold_rmsnorm"},{"norm",gamma},{"source",std::move(*recipe)}})).str();
      for(unsigned op=gemm.chain.count;op>0;--op)gemm.chain.operations[op]=gemm.chain.operations[op-1];
      ++gemm.chain.count;auto& norm=gemm.chain.operations[0];norm={};
      norm.kind=DmEpilogueKind::kDeferredRMSNorm;norm.parameter[0]=local_stats;
      norm.output_rounding=DmRounding::kBF16;
    }
  }
  if(next_norm_stats!=kDmNoIndex) {
    local_next=binding(next_norm_stats);block.stages.back().operands[5]=local_next;
  }
  std::vector<unsigned> remap(block.buffers.size());
  for(unsigned id=0;id<block.buffers.size();++id) {
    if(id==local_stats)remap[id]=norm_stats;
    else if(id==local_next)remap[id]=next_norm_stats;
    else if(id==block.node_buffer.at(match.input))remap[id]=input;
    else if(id==block.node_buffer.at(match.output))remap[id]=output;
    else {remap[id]=plan.buffers.size();plan.buffers.push_back(std::move(block.buffers[id]));}
  }
  auto id=[&](unsigned value) {return value==kDmNoIndex?value:remap.at(value);};
  auto map=[&](DmWriteMap& write) {write.layout=id(write.layout);write.rows=id(write.rows);};
  auto chain=[&](DmEpilogueChain& chain) {
    for(unsigned op=0;op<chain.count;++op) {
      for(auto& parameter:chain.operations[op].parameter)parameter=id(parameter);
      map(chain.operations[op].residual_map);
    }
    for(unsigned side=0;side<chain.side_count;++side) {
      auto& value=chain.side[side];value.buffer=id(value.buffer);value.auxiliary=id(value.auxiliary);
    }
  };
  unsigned gemm_base=plan.gemms.size(),stage_base=plan.stages.size();
  for(auto& g:block.gemms) {
    g.a=id(g.a);g.b=id(g.b);g.c=id(g.c);g.d=id(g.d);g.norm_ss=id(g.norm_ss);g.ss_out=id(g.ss_out);
    auto& a=g.access;a.rows=id(a.rows);a.binding=id(a.binding);a.a_scale=id(a.a_scale);
    map(a.write);chain(g.chain);plan.gemms.push_back(std::move(g));
  }
  for(auto& s:block.stages) {
    for(auto& operand:s.operands)operand=id(operand);
    if(s.kind==PlanTaskKind::kGemm)s.gemm+=gemm_base;
    if(s.binding_producer!=kDmNoIndex)s.binding_producer+=stage_base;
    if(s.moe.step!=DmMoeStep::kNone)s.moe.router_gemm+=gemm_base;
    chain(s.chain);plan.stages.push_back(std::move(s));
  }
  for(auto const& [name,buffer]:block.node_buffer)plan.node_buffer[name]=id(buffer);
  plan.dm=true;
  ValidateDmModelPlan(plan);
  destination=std::move(plan);
}

void MaterializeMoeRegionStorage(ModelPlan& plan,unsigned tile_n,unsigned down_tile_n,
                                unsigned router_gemm) {
  if(!plan.dm || !(plan.serving || (plan.forward && plan.forward_token_axis)) || !tile_n)
    throw std::invalid_argument("router storage needs a MoE plan and positive tile N");
  for(auto const& stage:plan.stages) {
    auto const& config=stage.moe;
    if(router_gemm!=kDmNoIndex && config.router_gemm!=router_gemm)continue;
    if(config.step!=DmMoeStep::kSelect && config.step!=DmMoeStep::kSelectAndDispatch)continue;
    if(config.router_gemm>=plan.gemms.size())throw std::invalid_argument("invalid router GEMM index");
    auto const& router=plan.gemms[config.router_gemm];
    unsigned parts=(router.n+tile_n-1)/tile_n;
    unsigned tokens=config.row_capacity/config.top_k;
    for(auto operand:{0,1})Matrix(plan.buffers.at(stage.operands[operand]),tokens,Checked(std::uint64_t(parts)*config.top_k));
  }
  if(down_tile_n) {
    if(down_tile_n<16 || down_tile_n>256 || down_tile_n%16)
      throw std::invalid_argument("invalid MoE down/combine channel alignment");
    for(auto& stage:plan.stages)if(stage.kind==PlanTaskKind::kMoECombine) {
      if(router_gemm!=kDmNoIndex && stage.moe.router_gemm!=router_gemm)continue;
      if(stage.operands[5]!=kDmNoIndex && down_tile_n%32)
        throw std::invalid_argument("MoE residual squares require aligned 32-channel writers");
      // A scatter publishes one contribution per row to its channel tile.
      // Matching the consumer's columns prevents partial or duplicate units.
      stage.width=down_tile_n;
      Matrix(plan.buffers.at(stage.operands[4]),stage.moe.row_capacity/stage.moe.top_k,
             2*((stage.extent+down_tile_n-1)/down_tile_n));
    }
  }
}
} // namespace tilemega::frontend
