// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/MxMoeModel.h>
#include <llvm/Support/JSON.h>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
namespace tilemega::codegen {
namespace {
using namespace frontend;
using K=PlanTaskKind;
struct Reader {
  ModelPlan const& p;
  unsigned layer=0;
  [[noreturn]] void Fail(std::string const& where,std::string const& why)const {
    throw std::invalid_argument("MX MoE layer "+std::to_string(layer)+" "+where+": "+why);
  }
  void Need(bool ok,std::string const& where,std::string const& why)const {if(!ok)Fail(where,why);}
  PlanBuffer const& Buffer(unsigned id,std::string const& where)const {
    Need(id<p.buffers.size(),where,"invalid buffer index");return p.buffers[id];
  }
  PlanGemm const& Gemm(PlanStage const& s,std::string const& where)const {
    Need(s.kind==K::kGemm && s.gemm<p.gemms.size(),where,"expected GEMM stage");
    auto const& g=p.gemms[s.gemm];
    for(auto id:{g.a,g.b,g.c,g.d})Buffer(id,where);
    Need(Buffer(g.b,where).dtype=="bf16",where,"BF16 weights required");
    Need(Buffer(g.d,where).dtype==(g.epilogue==PlanGemm::Epilogue::kArgmaxPartial ? "f32" : "bf16"),where,"projection output dtype disagrees with epilogue");
    return g;
  }
  llvm::json::Object Recipe(unsigned id,std::string const& where)const {
    auto value=llvm::json::parse(Buffer(id,where).pack_json);
    Need(bool(value),where,"malformed weight recipe");auto* object=value->getAsObject();
    Need(object!=nullptr,where,"weight recipe must be object");return *object;
  }
  std::string String(llvm::json::Object const& o,char const* name,std::string const& where)const {
    auto v=o.getString(name);Need(bool(v),where,std::string("missing recipe ")+name);return v->str();
  }
  unsigned Integer(llvm::json::Object const& o,char const* name,std::string const& where)const {
    auto v=o.getInteger(name);Need(bool(v)&&*v>0&&std::uint64_t(*v)<=UINT32_MAX,where,std::string("invalid recipe ")+name);return unsigned(*v);
  }
  llvm::json::Object Fold(unsigned id,std::string const& where,std::string* gamma=nullptr)const {
    auto o=Recipe(id,where);Need(String(o,"kind",where)=="fold_rmsnorm",where,"normalization must be folded");
    auto norm=String(o,"norm",where);Need(!norm.empty(),where,"empty normalization name");if(gamma)*gamma=norm;
    auto* src=o.getObject("source");Need(src!=nullptr,where,"missing folded source");return *src;
  }
  void Deferred(PlanGemm const& g,std::string const& where)const {
    Need(g.chain.count && g.chain.operations[0].kind==DmEpilogueKind::kDeferredRMSNorm,where,"missing leading deferred RMS");
    auto const& op=g.chain.operations[0];Need(op.input_rounding==DmRounding::kFP32&&op.output_rounding==DmRounding::kBF16,where,"RMS must scale FP32 then round BF16");
    Need(Buffer(op.parameter[0],where).dtype=="f32",where,"RMS statistics must be FP32");
  }
};
}
MxMoeModel AnalyzeMxMoe(frontend::ModelPlan const& p) {
  Reader r{p};MxMoeModel m{};
  r.Need(p.dm && p.serving && !p.forward && p.serving_seq==1,"entry","requires decoder DM decode plan");
  r.Need(std::isfinite(p.norm_epsilon)&&p.norm_epsilon>0,"entry","missing positive normalization epsilon");
  r.Need(p.serving_capacity>0 && p.serving_capacity%32==0,"entry","capacity must contain complete 32-token blocks");
  r.Need(p.stages.size()>=11 && p.stages[0].kind==K::kEmbedding,"embedding","missing embedding prefix");
  auto const& emb=p.stages[0];m.tokens=emb.operands[0];m.embed=emb.operands[1];m.hidden=emb.width;m.vocabulary=emb.extent;
  auto const& tokens=r.Buffer(m.tokens,"embedding");auto const& weight=r.Buffer(m.embed,"embedding");
  r.Need(m.hidden>0&&m.vocabulary>0&&weight.dtype=="bf16","embedding","invalid embedding geometry/dtype");
  m.token_id_bits=p.token_id_bits;r.Need((m.token_id_bits==32&&tokens.dtype=="i32")||(m.token_id_bits==64&&tokens.dtype=="i64"),"embedding","token dtype/bits disagree");
  m.capacity=unsigned(p.serving_capacity);m.epsilon=p.norm_epsilon;m.rope_fp32_phase=p.rope_fp32_phase;
  unsigned current=emb.operands[2],at=1;
  auto take=[&](K kind,char const* where)->unsigned {
    r.Need(at<p.stages.size()&&p.stages[at].kind==kind,where,"unexpected stage sequence");return at++;
  };
  while(at+2<p.stages.size()) {
    r.layer=unsigned(m.layers.size());MxMoeLayer l{};
    l.qkv=take(K::kGemm,"qkv");l.attention=take(K::kFusedAttention,"attention");
    auto const& qkv=r.Gemm(p.stages[l.qkv],"qkv");auto const& a=p.stages[l.attention];
    if(at<p.stages.size()&&p.stages[at].kind==K::kAttentionMerge) {
      auto const& merge=p.stages[at++];
      r.Need(merge.extent==a.extent&&merge.width==a.width&&merge.group==a.group&&merge.operands[2]==a.operands[7],"attention merge","shape/context mismatch");
    }
    l.o=take(K::kGemm,"o");l.router=take(K::kGemm,"router");l.dispatch=take(K::kMoETopK,"dispatch");
    l.gate_up=take(K::kGemm,"gate_up");l.down=take(K::kGemm,"down");l.combine=take(K::kMoECombine,"combine");
    auto const& o=r.Gemm(p.stages[l.o],"o");auto const& router=r.Gemm(p.stages[l.router],"router");
    auto const& gu=r.Gemm(p.stages[l.gate_up],"gate_up");auto const& down=r.Gemm(p.stages[l.down],"down");
    auto const& top=p.stages[l.dispatch];auto const& combine=p.stages[l.combine];
    auto const& cfg=top.moe;
    r.Need(cfg.step==DmMoeStep::kSelectAndDispatch,"dispatch","requires one SelectAndDispatch stage");
    r.Need(cfg.experts==128&&cfg.top_k==8&&cfg.row_capacity%cfg.top_k==0,"dispatch","unsupported expert/top-k/row geometry");
    unsigned const batch=cfg.row_capacity/cfg.top_k;
    r.Need(batch>0&&batch<=16&&cfg.router_gemm==p.stages[l.router].gemm,"dispatch","invalid batch/router binding");
    auto const& tw=r.Buffer(top.operands[3],"routing weights");r.Need(tw.dtype=="f32"||tw.dtype=="bf16","routing weights","unsupported routing dtype");
    if(m.layers.empty()) {
      m.batch=batch;m.hkv=a.extent;m.dim=a.width;m.queries=a.group;m.heads=m.hkv*m.queries;
      m.experts=cfg.experts;m.top_k=cfg.top_k;m.intermediate=down.k;m.cosine=a.operands[3];m.sine=a.operands[4];m.routing_weights_fp32=tw.dtype=="f32";
    }
    r.Need(batch==m.batch&&a.extent==m.hkv&&a.width==m.dim&&a.group==m.queries&&cfg.experts==m.experts&&cfg.top_k==m.top_k,"layer geometry","layers disagree");
    r.Need(m.dim==128&&m.queries==8&&m.hkv==4,"attention","covered MoE attention geometry required");
    r.Need(qkv.n==(m.queries+2)*m.hkv*m.dim&&qkv.k==m.hidden&&qkv.a==current&&a.operands[0]==qkv.d,"qkv","projection/input layout mismatch");
    r.Need(o.n==m.hidden&&o.k==m.heads*m.dim&&o.a==a.operands[7]&&o.c==current&&o.epilogue==PlanGemm::Epilogue::kResidual&&o.beta==1.0f,"o","context/residual geometry mismatch");
    r.Need(router.n==m.experts&&router.k==m.hidden&&router.a==o.d,"router","router geometry/input mismatch");
    r.Deferred(router,"router");r.Need(router.chain.count==1,"router","unexpected router epilogue step");
    r.Need(router.chain.side_count==1&&router.chain.side[0].kind==DmSideOutputKind::kTopKPartial&&router.chain.side[0].buffer==top.operands[0]&&router.chain.side[0].auxiliary==top.operands[1]&&router.chain.side[0].count==m.top_k,"router","missing rounded-logit top-k side output");
    r.Need(gu.n==2*m.intermediate&&gu.k==m.hidden&&down.n==m.hidden&&down.k==m.intermediate&&m.hidden%32==0&&m.intermediate%32==0,"experts","GEMM shapes disagree");
    r.Need(gu.a==o.d&&gu.access.a==DmAAccess::kRowGather&&gu.access.b==DmBAccess::kExpertIndirect&&down.a==gu.d&&down.access.a==DmAAccess::kDense&&down.access.b==DmBAccess::kExpertIndirect,"experts","invalid gather/indirect access");
    r.Need(gu.access.binding==top.operands[4]&&down.access.binding==top.operands[4]&&gu.access.rows==top.operands[5]&&down.access.write.rows==top.operands[5]&&down.access.write.kind==DmWriteKind::kRowScatter,"experts","dispatch/scatter bindings disagree");
    r.Need(gu.access.expert_stride==std::uint64_t(2)*m.intermediate*m.hidden&&down.access.expert_stride==std::uint64_t(m.hidden)*m.intermediate,"experts","expert stride mismatch");
    r.Deferred(gu,"gate_up");r.Need(gu.chain.count==2,"gate_up","unexpected epilogue length");auto const& gate=gu.chain.operations[1];
    r.Need(gate.kind==DmEpilogueKind::kGatePair&&gate.gate==DmGatePair::kSwiGLU&&gate.unit==16&&gate.input_rounding==DmRounding::kBF16&&gate.output_rounding==DmRounding::kBF16,"gate_up","SwiGLU numerical contract mismatch");
    r.Need(down.chain.count==0,"down","unexpected down epilogue");
    r.Need(combine.moe.step==DmMoeStep::kCombine&&combine.moe.experts==m.experts&&combine.moe.top_k==m.top_k&&combine.extent==m.hidden&&combine.operands[0]==down.d&&combine.operands[1]==top.operands[3]&&combine.operands[2]==o.d,"combine","rank partial/residual bindings disagree");
    r.Need((tw.dtype=="f32")==m.routing_weights_fp32,"routing weights","routing dtype differs by layer");
    std::string router_gamma,gu_gamma;r.Fold(router.b,"router recipe",&router_gamma);auto gu_leaf=r.Fold(gu.b,"gate_up recipe",&gu_gamma);auto down_leaf=r.Recipe(down.b,"down recipe");
    r.Need(router_gamma==gu_gamma,"expert recipes","router/gate normalization differs");
    r.Need(r.String(gu_leaf,"kind","gate_up recipe")=="expert_stack"&&r.String(down_leaf,"kind","down recipe")=="expert_stack"&&r.String(gu_leaf,"part","gate_up recipe")=="gate_up"&&r.String(down_leaf,"part","down recipe")=="down"&&r.String(gu_leaf,"prefix","gate_up recipe")==r.String(down_leaf,"prefix","down recipe"),"expert recipes","stack namespace/part mismatch");
    for(auto const* leaf:{&gu_leaf,&down_leaf})r.Need(r.Integer(*leaf,"experts","expert recipe")==m.experts&&r.Integer(*leaf,"hidden","expert recipe")==m.hidden&&r.Integer(*leaf,"intermediate","expert recipe")==m.intermediate&&r.Integer(*leaf,"u","expert recipe")==16,"expert recipes","stack shape/interleave mismatch");
    auto qleaf=r.Fold(qkv.b,"qkv recipe");r.Need(r.String(qleaf,"kind","qkv recipe")=="qkv_group_interleave"&&r.Integer(qleaf,"hkv","qkv recipe")==m.hkv&&r.Integer(qleaf,"qperkv","qkv recipe")==m.queries&&r.Integer(qleaf,"head_dim","qkv recipe")==m.dim,"qkv recipe","grouped projection layout mismatch");
    l.q_norm=a.operands[5];l.k_norm=a.operands[6];l.k_cache=a.operands[1];l.v_cache=a.operands[2];
    for(auto id:{l.q_norm,l.k_norm,l.k_cache,l.v_cache,m.cosine,m.sine,a.operands[7]})r.Need(r.Buffer(id,"attention buffers").dtype=="bf16","attention buffers","BF16 normalization/cache/rope/context required");
    for(auto id:{l.q_norm,l.k_norm}) {
      auto const& gamma=r.Buffer(id,"qk norm");
      r.Need(gamma.constant==m.dim&&!gamma.per_seq&&!gamma.per_batch&&!gamma.per_past&&!gamma.per_total,"qk norm","normalization weight must cover exactly one head");
    }
    r.Need(qkv.norm_ss!=kDmNoIndex&&r.Buffer(qkv.norm_ss,"qkv norm").dtype=="f32","qkv norm","missing FP32 pre-normalization statistics");
    r.Need(a.operands[3]==m.cosine&&a.operands[4]==m.sine,"rope","rope tables differ by layer");l.gate_chain=gu.chain;
    current=combine.operands[3];r.Buffer(current,"combine output");m.layers.push_back(l);
  }
  r.layer=unsigned(m.layers.size());r.Need(!m.layers.empty()&&at+2==p.stages.size(),"head","incomplete layer/tail sequence");
  auto const& head_stage=p.stages[take(K::kGemm,"head")];auto const& head=r.Gemm(head_stage,"head");auto const& arg=p.stages[take(K::kArgmaxReduce,"argmax")];
  r.Need(head.n==m.vocabulary&&head.k==m.hidden&&head.a==current&&head.epilogue==PlanGemm::Epilogue::kArgmaxPartial&&arg.operands[2]==m.tokens,"head","head/argmax geometry or token binding mismatch");r.Need(head.d==arg.operands[0]&&r.Buffer(arg.operands[0],"argmax values").dtype=="f32"&&r.Buffer(arg.operands[1],"argmax indices").dtype=="i32","head","argmax partial binding/dtype mismatch");r.Fold(head.b,"head recipe");m.head_gemm=head_stage.gemm;
  std::set<std::string> names;
  for(unsigned id=0;id<p.buffers.size();++id) {
    auto const& b=p.buffers[id];if(b.role!="external"&&b.source!=PlanBuffer::Source::kWeight)continue;
    r.Need(!b.name.empty()&&names.insert(b.name).second,"external buffers","empty/duplicate external name");
    r.Need(b.external_name.empty()||b.external_name==b.name,"external buffers","external name alias unsupported");
    m.external_buffers.push_back({id,b});
  }
  return m;
}
}
