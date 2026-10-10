// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <map>
#include <stdexcept>

namespace tilemega::frontend {
namespace {
using namespace analysis;
using namespace codegen;
ClosedForm C(long value) {return ClosedForm::Constant(value);}
IndexResult I(char const* dim,long scale=1,long divisor=1) {
  return IndexResult::Dim(dim,C(scale),C(divisor));
}
IndexResult Sum(std::initializer_list<IndexResult> terms,long offset=0) {
  auto result=IndexResult::Affine({},C(offset));
  for(auto const& term:terms)
    result.terms.insert(result.terms.end(),term.terms.begin(),term.terms.end());
  return result;
}
IterationDim D(char const* name,ClosedForm extent,bool reduction=false) {
  IterationDim result;result.name=name;result.extent=std::move(extent);
  if(reduction)result.type=IteratorType::kReduction;
  return result;
}
struct Builder {
  ModelPlan const& plan;
  ClosedForm tokens;
  LiftedModel result;
  std::map<unsigned,TensorSpace> spaces;
  std::map<unsigned,std::string> writers;
  Builder(ModelPlan const& p,ClosedForm t):plan(p),tokens(std::move(t)) {
    result.has_plan=true;result.written.assign(p.buffers.size(),0);
  }
  TensorSpace Space(unsigned id,std::vector<TensorAxis> axes) const {
    return {plan.buffers.at(id).name,std::move(axes)};
  }
  TensorSpace const& At(unsigned id) const {return spaces.at(id);}
  void Matrix(unsigned id,ClosedForm rows,ClosedForm columns) {
    if(!spaces.count(id))spaces[id]=Space(id,{{"row",rows},{"column",columns}});
  }
  SemanticOperand Read(unsigned id,std::vector<IndexResult> map) const {
    SemanticOperand operand;operand.tensor=At(id);operand.map.results=std::move(map);
    if(auto it=writers.find(id);it!=writers.end())operand.producer=it->second;
    return operand;
  }
  void Own(SemanticOp& op,std::vector<TensorAxis> axes,std::vector<IndexResult> map) {
    op.exact_task_access=true;op.task_space={op.name+".owners",std::move(axes)};
    op.task_map.results=std::move(map);
  }
  std::vector<IndexResult> StoreMap(SemanticOp& op,unsigned id,std::vector<IndexResult> map) {
    for(unsigned axis=0;axis<map.size();++axis)if(map[axis].kind==IndexResult::Kind::kFullRange) {
      auto name="store"+std::to_string(op.domain.size());
      IterationDim dim;dim.name=name;dim.extent=At(id).axes.at(axis).extent;
      dim.origin=map[axis].offset;op.domain.push_back(std::move(dim));
      map[axis]=IndexResult::Dim(name);
    }
    return map;
  }
  void Side(SemanticOp& op,unsigned id,std::vector<IndexResult> map) {
    ElementWrite write;write.tensor=At(id);write.map.results=StoreMap(op,id,std::move(map));
    write.effect.kind=EffectKind::kWrite;op.additional_writes.push_back(std::move(write));
    writers[id]=op.name;result.written[id]=1;
  }
  void Store(unsigned index,SemanticOp op,OpRole role,unsigned output,
      std::vector<IndexResult> map) {
    op.result=At(output);op.result_map.results=StoreMap(op,output,std::move(map));
    op.result_effect.kind=EffectKind::kWrite;
    op.dtype=plan.buffers.at(output).dtype=="f32"?ScalarType::kF32:ScalarType::kBF16;
    writers[output]=op.name;result.written[output]=1;
    result.ops.push_back({op.name,role,OwnershipKind::kTilePerBlock,int(index),0,
        plan.stages[index].representative});
    result.sem.ops.push_back(std::move(op));
  }
  ClosedForm Capacity(DmMoeStage const& cfg) const {
    auto assignments=tokens*C(cfg.top_k);
    return cfg.grouped?assignments.CeilDiv(C(cfg.block_rows))+C(cfg.experts).Min(assignments):assignments;
  }
  void RoutingSpaces(PlanStage const& s) {
    auto const& cfg=s.moe;auto const& ids=s.operands;
    for(unsigned slot:{0u,1u,2u,3u})Matrix(ids[slot],tokens,C(cfg.top_k));
    Matrix(ids[4],Capacity(cfg),C(4));Matrix(ids[5],tokens*C(cfg.top_k),C(4));
    Matrix(ids[6],tokens.CeilDiv(C(cfg.chunk_tokens)),C(cfg.experts));
    Matrix(ids[7],C(1),C(cfg.experts+1));Matrix(ids[8],C(1),C(cfg.experts+1));
  }
  void Normalization(SemanticOp& op,PlanGemm const& gemm,IndexResult row) {
    for(unsigned i=0;i<gemm.chain.count;++i) {
      auto const& step=gemm.chain.operations[i];
      if(step.kind!=DmEpilogueKind::kDeferredRMSNorm)continue;
      auto id=step.parameter[0];auto const& layout=plan.buffers.at(id).layout;
      if(layout.rank!=2 || layout.logical[1]!=gemm.k/32 || gemm.k%32)
        throw std::invalid_argument("MoE RMS semantic statistics require 32-channel squares");
      Matrix(id,tokens,C(gemm.k/32));
      op.operands.push_back(Read(id,{row,IndexResult::FullRange()}));
    }
  }
};
}

namespace {
void LiftStage(Builder& b,unsigned index) {
  auto const& plan=b.plan;
    auto const& s=plan.stages[index];auto const& ids=s.operands;
    SemanticOp op;op.name="moe.s"+std::to_string(index);
    if(s.kind==PlanTaskKind::kRMSNorm) {
      b.Matrix(ids[0],b.tokens,C(s.width));b.Matrix(ids[1],C(1),C(s.width));
      b.Matrix(ids[2],b.tokens,C(s.width));
      op.kind=OperatorKind::kReduction;op.arithmetic="rmsnorm";
      op.domain={D("m",b.tokens),D("n",C(s.width)),D("c",C(s.width),true)};
      op.operands={b.Read(ids[0],{I("m"),I("c")}),
          b.Read(ids[1],{IndexResult::Affine({},C(0)),I("n")})};
      b.Own(op,{{"m",b.tokens},{"n",C(s.width)}},{I("m"),I("n")});
      b.Store(index,std::move(op),OpRole::kNorm,ids[2],{I("m"),I("n")});
    }else if(s.kind==PlanTaskKind::kGemm) {
      auto const& g=plan.gemms.at(s.gemm);auto const& a=g.access;
      op.kind=OperatorKind::kMatmul;op.arithmetic="gemm";
      if(a.b==DmBAccess::kDense) {
        // A region can begin at an external hidden-state matrix without a
        // preceding normalization stage. Decoder inputs are already seeded.
        b.Matrix(g.a,b.tokens,C(g.k));
        b.Matrix(g.b,C(g.n),C(g.k));b.Matrix(g.d,b.tokens,C(g.n));
        op.domain={D("m",b.tokens),D("n",C(g.n)),D("k",C(g.k),true)};
        op.operands={b.Read(g.a,{I("m"),I("k")}),b.Read(g.b,{I("n"),I("k")})};
        b.Normalization(op,g,I("m"));
        b.Own(op,{{"m",b.tokens},{"n",C(g.n)}},{I("m"),I("n")});
        for(unsigned side=0;side<g.chain.side_count;++side) {
          auto const& partial=g.chain.side[side];
          if(partial.kind!=DmSideOutputKind::kTopKPartial)
            throw std::invalid_argument("unsupported router side output");
          for(auto id:{partial.buffer,partial.auxiliary}) {
            b.Matrix(id,b.tokens,C(partial.count));
            for(unsigned rank=0;rank<partial.count;++rank)
              b.Side(op,id,{I("m"),IndexResult::Affine({},C(rank))});
            op.tile_storage.push_back({b.At(id).name,"n",1});
          }
        }
        op.reduction={"k","add",op.name+".partials",op.name+".combine",true,{"m","n"}};
        b.Store(index,std::move(op),OpRole::kProjection,g.d,{I("m"),I("n")});
      }else if(a.b==DmBAccess::kExpertIndirect) {
        unsigned unit=0;
        for(unsigned chain=0;chain<g.chain.count;++chain) {
          auto const& step=g.chain.operations[chain];
          if(step.kind==DmEpilogueKind::kDeferredRMSNorm)continue;
          if(step.kind!=DmEpilogueKind::kGatePair || step.gate!=DmGatePair::kSwiGLU || unit)
            throw std::invalid_argument("unsupported MoE expert epilogue semantics");
          unit=step.unit;
        }
        bool paired=unit!=0;unsigned columns=paired?g.n/2:g.n;
        auto cfg=plan.stages.at(s.binding_producer).moe;
        auto capacity=b.Capacity(cfg),bm=C(a.block_rows);
        op.domain={D("v",capacity),D("row",bm),D("n",C(columns)),D("k",C(g.k),true)};
        auto& virtual_dim=op.domain.front();virtual_dim.runtime=true;virtual_dim.capacity=capacity;
        virtual_dim.binding_source=b.At(a.binding).name;
        virtual_dim.binding_requirement=cfg.grouped?"prefix_sum":"tensor_values";
        b.Own(op,{{"v",capacity},{"row",bm},{"n",C(columns)}},{I("v"),I("row"),I("n")});
        auto requests=std::vector<std::string>{"v","row"};
        auto token=cfg.grouped?IndexResult::DataDependent(b.At(a.rows).name,requests):I("v",1,a.routing_topk);
        auto row=cfg.grouped?IndexResult::DataDependent(b.At(a.binding).name,requests):I("v");
        auto expert=IndexResult::DataDependent(b.At(a.binding).name,{"v"});
        if(a.a==DmAAccess::kRowGather)op.operands.push_back(b.Read(g.a,{token,I("k")}));
        else op.operands.push_back(b.Read(g.a,{I("v"),I("row"),I("k")}));
        b.Normalization(op,g,token);
        b.spaces[g.b]=b.Space(g.b,{{"expert",C(a.experts)},{"n",C(g.n)},{"k",C(g.k)}});
        auto column=paired?Sum({I("n"),I("n",unit,unit)}):I("n");
        op.operands.push_back(b.Read(g.b,{expert,column,I("k")}));
        if(paired) {
          column.offset=column.offset+C(unit);op.operands.push_back(b.Read(g.b,{expert,column,I("k")}));
          op.arithmetic="swiglu_gemm";
        }
        op.operands.push_back(b.Read(a.binding,{I("v"),IndexResult::FullRange()}));
        op.operands.push_back(b.Read(a.rows,{row,IndexResult::FullRange()}));
        op.reduction={"k","add",op.name+".partials",op.name+".combine",true,
            {"v","row","n"},paired?2u:1u};
        if(a.write.kind==DmWriteKind::kRowScatter) {
          b.spaces[g.d]=b.Space(g.d,{{"token",b.tokens},{"rank",C(a.routing_topk)},{"n",C(columns)}});
          auto rank=cfg.grouped?IndexResult::DataDependent(b.At(a.rows).name,requests):
              Sum({I("v"),I("v",-long(a.routing_topk),a.routing_topk)});
          b.Store(index,std::move(op),OpRole::kProjection,g.d,{token,rank,I("n")});
        }else {
          b.spaces[g.d]=b.Space(g.d,{{"v",capacity},{"row",bm},{"n",C(columns)}});
          b.Store(index,std::move(op),OpRole::kProjection,g.d,{I("v"),I("row"),I("n")});
        }
      }else throw std::invalid_argument("unsupported MoE GEMM access mode");
    }else if(s.kind==PlanTaskKind::kMoETopK) {
      auto cfg=s.moe;b.RoutingSpaces(s);
      op.kind=OperatorKind::kGather;op.arithmetic="moe_dispatch";
      op.domain={D("m",b.tokens),D("n",C(cfg.top_k))};
      b.Own(op,{{"m",b.tokens},{"n",C(cfg.top_k)}},{I("m"),I("n")});
      auto full=IndexResult::FullRange();
      auto assignment=Sum({I("m",cfg.top_k),I("n")});
      auto selected=cfg.step==DmMoeStep::kSelect || cfg.step==DmMoeStep::kSelectAndDispatch;
      if(selected) {
        op.arithmetic="moe_topk";
        op.operands={b.Read(ids[0],{I("m"),full}),b.Read(ids[1],{I("m"),full})};
        b.Side(op,ids[3],{I("m"),I("n")});
        if(cfg.step==DmMoeStep::kSelectAndDispatch) {
          auto row=cfg.grouped?IndexResult::DataDependent(b.At(ids[2]).name,{"m","n"}):assignment;
          b.Side(op,ids[5],{row,full});
          b.Side(op,ids[4],{cfg.grouped?IndexResult::DataDependent(b.At(ids[2]).name,{"m","n"}):assignment,full});
          if(cfg.grouped)for(unsigned slot:{6u,7u,8u})b.Side(op,ids[slot],{full,full});
        }
        b.Store(index,std::move(op),OpRole::kMoERouting,ids[2],{I("m"),I("n")});
      }else if(cfg.step==DmMoeStep::kHistogram) {
        op.operands={b.Read(ids[2],{I("m"),full})};
        b.Store(index,std::move(op),OpRole::kMoERouting,ids[6],{I("m",1,cfg.chunk_tokens),full});
      }else if(cfg.step==DmMoeStep::kPrefix) {
        op.operands={b.Read(ids[6],{full,full})};
        b.Side(op,ids[6],{full,full});b.Side(op,ids[7],{full,full});b.Side(op,ids[8],{full,full});
        // Active records depend on histogram values; every tail valid flag is
        // cleared by this same CTA. The I2 envelope includes both sets.
        b.Store(index,std::move(op),OpRole::kMoERouting,ids[4],
            {IndexResult::DataDependent(b.At(ids[6]).name,{"m","n"}),full});
      }else if(cfg.step==DmMoeStep::kScatter) {
        op.operands={b.Read(ids[2],{I("m"),I("n")}),b.Read(ids[3],{I("m"),I("n")})};
        auto row=assignment;
        if(cfg.grouped) {
          auto expert=IndexResult::DataDependent(b.At(ids[2]).name,{"m","n"});
          op.operands.push_back(b.Read(ids[6],{I("m",1,cfg.chunk_tokens),expert}));
          op.operands.push_back(b.Read(ids[7],{IndexResult::Affine({},C(0)),expert}));
          row=IndexResult::DataDependent(b.At(ids[2]).name,{"m","n"});
        }else b.Side(op,ids[4],{assignment,full});
        b.Store(index,std::move(op),OpRole::kMoERouting,ids[5],{row,full});
      }else throw std::invalid_argument("invalid MoE routing semantic step");
    }else if(s.kind==PlanTaskKind::kMoECombine) {
      auto const& cfg=s.moe;b.Matrix(ids[1],b.tokens,C(cfg.top_k));
      b.Matrix(ids[2],b.tokens,C(s.extent));b.Matrix(ids[3],b.tokens,C(s.extent));
      b.Matrix(ids[4],b.tokens,C(2));
      op.kind=OperatorKind::kReduction;op.arithmetic="moe_combine";
      op.domain={D("m",b.tokens),D("n",C(s.extent)),D("rank",C(cfg.top_k),true)};
      op.operands={b.Read(ids[0],{I("m"),I("rank"),I("n")}),
          b.Read(ids[1],{I("m"),I("rank")}),b.Read(ids[2],{I("m"),I("n")})};
      b.Own(op,{{"m",b.tokens},{"n",C(s.extent)}},{I("m"),I("n")});
      for(unsigned stat=0;stat<2;++stat)b.Side(op,ids[4],{I("m"),IndexResult::Affine({},C(stat))});
      op.tile_storage.push_back({b.At(ids[4]).name,"n",1});
      if(ids[5]!=kDmNoIndex) {
        b.Matrix(ids[5],b.tokens,C(s.extent/32));
        b.Side(op,ids[5],{I("m"),I("n",1,32)});
      }
      b.Store(index,std::move(op),OpRole::kMoECombine,ids[3],{I("m"),I("n")});
    }else throw std::invalid_argument("MoE region has an unsupported stage kind");
}
}

LiftedModel LiftMoeRegionSemantics(ModelPlan const& plan,LiftOptions const& options) {
  if(!plan.dm || !plan.forward || !plan.forward_token_axis || !options.forward ||
      options.seq_symbol.empty())
    throw std::invalid_argument("MoE region semantics require a symbolic token-axis forward plan");
  ValidateDmModelPlan(plan);
  Builder b(plan,ClosedForm::Symbol(options.seq_symbol));
  for(unsigned index=0;index<plan.stages.size();++index)LiftStage(b,index);
  return std::move(b.result);
}

LiftedModel LiftMoeStageSemantics(ModelPlan const& plan,LiftOptions const& options,
    unsigned index,LiftedModel const& preceding) {
  if(!plan.dm || !plan.serving || !options.serving || options.batch_symbol.empty() ||
      options.static_seq!=plan.serving_seq || index>=plan.stages.size())
    throw std::invalid_argument("MoE decoder semantics require bound serving dimension roles");
  Builder b(plan,ClosedForm::Symbol(options.batch_symbol)*C(plan.serving_seq));
  std::map<std::string,unsigned> buffers;
  for(unsigned id=0;id<plan.buffers.size();++id)buffers.emplace(plan.buffers[id].name,id);
  auto seed=[&](TensorSpace const& tensor,std::string const& writer) {
    auto found=buffers.find(tensor.name);if(found==buffers.end())return;
    b.spaces[found->second]=tensor;b.writers[found->second]=writer;
  };
  for(auto const& op:preceding.sem.ops) {
    seed(op.result,op.name);for(auto const& write:op.additional_writes)seed(write.tensor,op.name);
  }
  LiftStage(b,index);return std::move(b.result);
}
} // namespace tilemega::frontend
