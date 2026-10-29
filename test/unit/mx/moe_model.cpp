// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/MxMoeModel.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <llvm/Support/JSON.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>
using namespace tilemega::frontend;
using namespace tilemega::codegen;
namespace {
ModelPlan Fixture(unsigned batch,unsigned layers,bool merge,bool fp32) {
  ModelPlan p;p.dm=p.serving=true;p.serving_seq=1;p.serving_capacity=1088;
  p.norm_epsilon=1e-6;p.token_id_bits=32;
  auto buffer=[&](std::string name,std::string dtype="bf16",bool weight=false,std::string recipe="") {
    unsigned id=p.buffers.size();PlanBuffer b;b.name=std::move(name);b.dtype=std::move(dtype);b.pack_json=std::move(recipe);
    if(weight)b.source=PlanBuffer::Source::kWeight;
    p.buffers.push_back(b);return id;
  };
  auto stage=[&](PlanTaskKind kind) -> PlanStage& {p.stages.emplace_back();p.stages.back().kind=kind;return p.stages.back();};
  auto gemm=[&](unsigned n,unsigned k,unsigned a,unsigned w,unsigned out)->unsigned {
    unsigned id=p.gemms.size();PlanGemm g;g.n=n;g.k=k;g.a=a;g.b=w;g.c=a;g.d=out;p.gemms.push_back(g);
    stage(PlanTaskKind::kGemm).gemm=id;return id;
  };
  auto fold=[](std::string leaf){return "{\"kind\":\"fold_rmsnorm\",\"norm\":\"gamma\",\"source\":"+leaf+"}";};
  unsigned tokens=buffer("tokens","i32"),embed=buffer("embed","bf16",true),cos=buffer("cos"),sin=buffer("sin"),current=buffer("x0");p.buffers[tokens].role="external";p.buffers[cos].role=p.buffers[sin].role="external";
  auto& embedding=stage(PlanTaskKind::kEmbedding);embedding.operands[0]=tokens;embedding.operands[1]=embed;embedding.operands[2]=current;embedding.width=2048;embedding.extent=151936;
  for(unsigned l=0;l<layers;++l) {
    std::string const prefix="layer"+std::to_string(l)+".";
    unsigned qkv_out=buffer(prefix+"qkv"),context=buffer(prefix+"context"),xprime=buffer(prefix+"xprime"),statistics=buffer(prefix+"stats","f32");
    unsigned qweight=buffer(prefix+"qweight","bf16",true,fold(R"({"kind":"qkv_group_interleave","hkv":4,"qperkv":8,"head_dim":128})"));
    unsigned q=gemm(5120,2048,current,qweight,qkv_out);p.gemms[q].norm_ss=statistics;
    unsigned kc=buffer(prefix+"k"),vc=buffer(prefix+"v"),qn=buffer(prefix+"qn","bf16",true),kn=buffer(prefix+"kn","bf16",true);p.buffers[qn].constant=p.buffers[kn].constant=128;p.buffers[kc].role=p.buffers[vc].role="external";
    auto& att=stage(PlanTaskKind::kFusedAttention);att.extent=4;att.width=128;att.group=8;
    att.operands[0]=qkv_out;att.operands[1]=kc;att.operands[2]=vc;att.operands[3]=cos;att.operands[4]=sin;att.operands[5]=qn;att.operands[6]=kn;att.operands[7]=context;
    if(merge){auto& a=stage(PlanTaskKind::kAttentionMerge);a.extent=4;a.width=128;a.group=8;a.operands[2]=context;}
    unsigned ow=buffer(prefix+"ow","bf16",true);unsigned o=gemm(2048,4096,context,ow,xprime);p.gemms[o].c=current;p.gemms[o].beta=1;p.gemms[o].epilogue=PlanGemm::Epilogue::kResidual;
    unsigned logits=buffer(prefix+"logits"),partial=buffer(prefix+"partial","f32"),indices=buffer(prefix+"indices","i32"),routing=buffer(prefix+"weights",fp32?"f32":"bf16"),bindings=buffer(prefix+"bindings","i32"),rows=buffer(prefix+"rows","i32");
    unsigned rw=buffer(prefix+"rw","bf16",true,fold(R"({"kind":"alias","source":"router"})"));unsigned router=gemm(128,2048,xprime,rw,logits);
    auto& chain=p.gemms[router].chain;chain.count=1;chain.operations[0].kind=DmEpilogueKind::kDeferredRMSNorm;chain.operations[0].parameter[0]=statistics;chain.operations[0].output_rounding=DmRounding::kBF16;
    chain.side_count=1;chain.side[0].kind=DmSideOutputKind::kTopKPartial;chain.side[0].buffer=partial;chain.side[0].auxiliary=indices;chain.side[0].count=8;
    auto& top=stage(PlanTaskKind::kMoETopK);top.moe.step=DmMoeStep::kSelectAndDispatch;top.moe.experts=128;top.moe.top_k=8;top.moe.row_capacity=batch*8;top.moe.router_gemm=router;top.operands[0]=partial;top.operands[1]=indices;top.operands[3]=routing;top.operands[4]=bindings;top.operands[5]=rows;
    auto expert_recipe=[](char const* part){return std::string("{\"kind\":\"expert_stack\",\"part\":\"")+part+"\",\"prefix\":\"experts.\",\"experts\":128,\"hidden\":2048,\"intermediate\":768,\"u\":16}";};
    unsigned guw=buffer(prefix+"guw","bf16",true,fold(expert_recipe("gate_up"))),dw=buffer(prefix+"dw","bf16",true,expert_recipe("down")),h=buffer(prefix+"h"),partials=buffer(prefix+"partials"),output=buffer(prefix+"out");
    unsigned gu=gemm(1536,2048,xprime,guw,h);auto& g=p.gemms[gu];g.access.a=DmAAccess::kRowGather;g.access.b=DmBAccess::kExpertIndirect;g.access.binding=bindings;g.access.rows=rows;g.access.expert_stride=1536*2048;
    g.chain=p.gemms[router].chain;g.chain.side_count=0;g.chain.count=2;auto& gate=g.chain.operations[1];gate.kind=DmEpilogueKind::kGatePair;gate.gate=DmGatePair::kSwiGLU;gate.unit=16;gate.input_rounding=gate.output_rounding=DmRounding::kBF16;
    unsigned down=gemm(2048,768,h,dw,partials);auto& d=p.gemms[down];d.access.b=DmBAccess::kExpertIndirect;d.access.binding=bindings;d.access.expert_stride=2048*768;d.access.write.kind=DmWriteKind::kRowScatter;d.access.write.rows=rows;
    auto& combine=stage(PlanTaskKind::kMoECombine);combine.moe.step=DmMoeStep::kCombine;combine.moe.experts=128;combine.moe.top_k=8;combine.extent=2048;combine.operands[0]=partials;combine.operands[1]=routing;combine.operands[2]=xprime;combine.operands[3]=output;current=output;
  }
  unsigned hw=buffer("head.weight","bf16",true,fold(R"({"kind":"alias","source":"head"})")),values=buffer("head.values","f32"),indices=buffer("head.indices","i32");unsigned head=gemm(151936,2048,current,hw,values);p.gemms[head].epilogue=PlanGemm::Epilogue::kArgmaxPartial;
  auto& arg=stage(PlanTaskKind::kArgmaxReduce);arg.operands[0]=values;arg.operands[1]=indices;arg.operands[2]=tokens;return p;
}
}
int main(int argc,char** argv) {
 try {
  unsigned positive=0,negative=0;
  for(unsigned b:{1u,2u,4u,8u,16u})for(unsigned layers:{1u,2u,24u})for(bool merge:{false,true})for(bool fp32:{false,true}) {
    auto p=Fixture(b,layers,merge,fp32);auto m=AnalyzeMxMoe(p);
    assert(m.batch==b&&m.layers.size()==layers&&m.hidden==2048&&m.hkv==4&&m.queries==8&&m.routing_weights_fp32==fp32);++positive;
    auto prepared=PrepareMxMoeBuffersV2(p,{16,32,64,64,128,128});
    assert(prepared.size()==m.external_buffers.size());unsigned wrapped=0;
    for(unsigned index=0;index<prepared.size();++index) {
      auto const& native=prepared[index];auto const& original=m.external_buffers[index];
      assert(native.original_index==original.original_index&&native.buffer.name==original.buffer.name&&native.buffer.dtype==original.buffer.dtype&&native.buffer.constant==original.buffer.constant&&native.buffer.per_batch==original.buffer.per_batch);
      if(native.slot==MxMoeSlotV2::kQkv||native.slot==MxMoeSlotV2::kO||native.slot==MxMoeSlotV2::kRouter||native.slot==MxMoeSlotV2::kGateUp||native.slot==MxMoeSlotV2::kDown||native.slot==MxMoeSlotV2::kHead) {
        auto recipe=llvm::json::parse(native.buffer.pack_json);assert(recipe);auto* tile=recipe->getAsObject();assert(tile&&tile->getString("kind")=="tile_pages"&&tile->getString("layout")=="xcore_mma"&&tile->getInteger("tile_k")==32);++wrapped;
      } else assert(native.buffer.pack_json==original.buffer.pack_json);
    }
    assert(wrapped==5*layers+1);
    for(auto invalid:{MxMoeWeightTilesV2{16,32,64,16,128,128},MxMoeWeightTilesV2{16,32,64,64,128,0}}) {
      bool rejected=false;try{PrepareMxMoeBuffersV2(p,invalid);}catch(std::invalid_argument const&){rejected=true;}assert(rejected);
    }
    auto reject=[&](std::string const& location,std::function<void(ModelPlan&)> mutate){auto q=p;mutate(q);bool failed=false;try{AnalyzeMxMoe(q);}catch(std::invalid_argument const& e){failed=std::string(e.what()).find(location)!=std::string::npos;}if(!failed)throw std::runtime_error(std::string("missing rejection: ")+location);++negative;};
    reject("layer 0 qkv",[&](auto& q){q.gemms[q.stages[m.layers[0].qkv].gemm].n+=16;});
    reject("layer 0 dispatch",[&](auto& q){q.stages[m.layers[0].dispatch].moe.step=DmMoeStep::kSelect;});
    reject("layer 0 router recipe",[&](auto& q){q.buffers[q.gemms[q.stages[m.layers[0].router].gemm].b].pack_json="{";});
    reject("layer 0 router recipe",[&](auto& q){q.buffers[q.gemms[q.stages[m.layers[0].router].gemm].b].pack_json="{}";});
    reject("layer 0 expert recipes",[&](auto& q){auto& recipe=q.buffers[q.gemms[q.stages[m.layers[0].gate_up].gemm].b].pack_json;auto pos=recipe.find("gamma");recipe.replace(pos,5,"other");});
    reject("layer 0 gate_up",[&](auto& q){q.gemms[q.stages[m.layers[0].gate_up].gemm].chain.operations[1].output_rounding=DmRounding::kFP32;});
    reject("layer 0 experts",[&](auto& q){q.gemms[q.stages[m.layers[0].down].gemm].access.write.kind=DmWriteKind::kDense;});
    reject("layer "+std::to_string(layers)+" head",[&](auto& q){q.gemms[m.head_gemm].n+=16;});
  }
  if(argc==2){auto bridge=ReadExportBridge(argv[1]);for(unsigned b:{1u,2u,4u,8u,16u}){ServingOptions o;o.moe_batch=b;o.capacity=1088;o.phase=ServingOptions::Phase::kDecode;o.seq=1;o.deferred_norm=true;auto p=BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,o);auto m=AnalyzeMxMoe(p);assert(m.batch==b&&m.layers.size()==24&&m.external_buffers.size()==246);auto buffers=PrepareMxMoeBuffersV2(p,{16,32,64,64,128,128});assert(buffers.size()==246);unsigned unused=0;for(auto const& row:buffers)unused+=row.slot==MxMoeSlotV2::kUnused;assert(unused==25);++positive;}}
  std::cout<<"PASS portable MoE analysis positive="<<positive<<" negative="<<negative<<" GPU_calls=0\n";return 0;
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
