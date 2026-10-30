// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/MxMoeProgramV2.h>
#include <tilemega/Backend/Mx/DecodeGeometryV2.h>
#include "StageProgram.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace tilemega::codegen {
namespace {
using namespace backend::mx;
[[noreturn]] void Fail(char const* why){throw std::invalid_argument(std::string("MX MoE program: ")+why);}
int Width(MxDecodeClassV2 const& c,int n,int k,int threads){return MxDecodePrefetchV2(c,n,k,104,threads);}
int GemmSmem(int b,MxDecodeClassV2 const& c){int rounds=MxReduceRounds(b,c.tn);int squares=c.wn==8?0:(rounds==1?8:4)*b*c.tn;int slots=rounds==2&&c.wn>1?c.wn/2:c.wn;int flags=squares+512+(c.wn==8?0:slots)*b*(c.tn+1);return ((flags+c.wn)*4+15)/16*16;}
int Smem(MxMoeModel const& m,MxMoeConfigV2 const& c){int slots=std::min(int(m.capacity/32),104);int wave=64+m.dim/2+m.queries*(m.dim+2);int scratch=std::max(slots*int(m.queries)*3,int(m.queries*(m.dim+2)));int attn=(8*wave+scratch+4)*4;int head=(std::max(GemmSmem(m.batch,c.head),int(m.batch)*128*8)+31)/16*16;int dispatch=(m.batch*m.experts+2*m.batch*8+m.experts)*4;int maximum=std::max({32784,attn,head,dispatch,GemmSmem(m.batch,c.qkv),GemmSmem(m.batch,c.o),GemmSmem(m.batch,c.router)+16,GemmSmem(m.batch,c.gate_up),GemmSmem(m.batch,c.down)});return (maximum+15)/16*16;}
std::string Step(DmEpilogueOp const& op){std::ostringstream s;s<<"tilemega::codegen::DmEpilogueStep<tilemega::codegen::DmEpilogueKind("<<unsigned(op.kind)<<"),tilemega::codegen::DmActivation("<<unsigned(op.activation)<<"),tilemega::codegen::DmGatePair("<<unsigned(op.gate)<<"),"<<op.unit<<",tilemega::codegen::DmRounding("<<unsigned(op.input_rounding)<<"),tilemega::codegen::DmRounding("<<unsigned(op.output_rounding)<<"),tilemega::codegen::DmWriteKind("<<unsigned(op.residual_map.kind)<<"),"<<op.residual_map.factor<<">";return s.str();}
}
void CheckMxMoeConfigV2(frontend::ModelPlan const& plan,MxMoeConfigV2 const& c){
  auto m=AnalyzeMxMoe(plan);if(c.threads!=512&&c.threads!=576)Fail("threads must be 512 or 576");
  if(m.token_id_bits!=32)Fail("native token binding requires I32");
  if(c.b_min!=1&&c.b_min!=2&&c.b_min!=4&&c.b_min!=8)Fail("invalid attention b_min");
  if(c.attn_prefetch<0||c.attn_prefetch>2)Fail("invalid attention prefetch");
  auto check=[&](MxDecodeClassV2 const& item,int n,int k,MxOpV2 op,bool grouped){
    if(n%item.tn)Fail("incomplete output tile");
    if(grouped&&item.stream_k)Fail("grouped units require TileRange");
    if(!MxGemmLegalV2(n,k,m.batch,item.tn,item.wn,item.d,op,item.stream_k?MxPart::kStreamK:MxPart::kTileRange))Fail("illegal GEMM geometry");
    if(grouped&&op==MxOpV2::kSwiGLU){if(item.prefetch!=0)Fail("gate/up has no producer-independent prefetch");}
    else {
      if(Width(item,n,k,c.threads)<item.d)Fail("prefetch below ring depth");
      if(item.r_next>=0){auto derived=item;derived.prefetch=-1;if(Width(item,n,k,c.threads)>Width(derived,n,k,c.threads))Fail("prefetch exceeds measured resource budget");}
    }
  };
  int qn=(m.queries+2)*m.hkv*m.dim,active=MxMoeExpectedActiveV2(m.experts,m.batch);
  check(c.qkv,qn,m.hidden,MxOpV2::kStore,false);check(c.o,m.hidden,m.heads*m.dim,MxOpV2::kResidual,false);
  check(c.router,m.experts,m.hidden,MxOpV2::kStore,false);if(c.router.wn!=1)Fail("router has at most eight tiles and WN must be one");
  check(c.gate_up,active*2*m.intermediate,m.hidden,MxOpV2::kSwiGLU,true);check(c.down,active*m.hidden,m.intermediate,MxOpV2::kStore,true);
  check(c.head,m.vocabulary,m.hidden,MxOpV2::kStore,false);if(c.head.stream_k||(c.head.wn!=4&&c.head.wn!=8))Fail("head requires TileRange WN4/8");
  auto loads=[&](MxDecodeClassV2 const& item,int n,int k){return Width(item,n,k,c.threads)*(item.tn/16);};
  auto wait=[&](MxDecodeClassV2 const& item,int next){if((item.d-1)*(item.tn/16+1)+next>63)Fail("next-stage wait exceeds 63");};
  wait(c.qkv,c.attn_prefetch*(m.dim/16));wait(c.o,loads(c.router,m.experts,m.hidden));wait(c.router,0);
  wait(c.gate_up,loads(c.down,active*m.hidden,m.intermediate));wait(c.down,loads(c.qkv,qn,m.hidden));wait(c.down,loads(c.head,m.vocabulary,m.hidden));
  if(Smem(m,c)+((m.batch*4+15)/16*16)+32>65536)Fail("shared memory exceeds 64 KiB");
}
int MxMoeBodySmemBytesV2(frontend::ModelPlan const& plan,MxMoeConfigV2 const& c){
  CheckMxMoeConfigV2(plan,c);return Smem(AnalyzeMxMoe(plan),c);
}
std::string EmitMxMoeProgramV2(frontend::ModelPlan const& plan,MxMoeConfigV2 const& c,bool prompt,int past_lo,int past_hi){
  CheckMxMoeConfigV2(plan,c);auto m=AnalyzeMxMoe(plan);if(prompt){past_lo=0;past_hi=63;}
  if(past_lo<0||past_hi<past_lo||past_hi>=int(m.capacity)-1)Fail("invalid past interval");
  int qn=(m.queries+2)*m.hkv*m.dim,active=MxMoeExpectedActiveV2(m.experts,m.batch);
  auto width=[&](MxDecodeClassV2 const& item,int n,int k){return Width(item,n,k,c.threads);};
  auto shape=[](int n,int k){return "MxStaticShapeV2<"+std::to_string(n)+","+std::to_string(k)+",kBatch,kThreads>";};
  auto dense=[&](MxDecodeClassV2 const& item,char const* op,bool norm,int n,int k){return "MxGemm<"+std::to_string(item.tn)+","+std::to_string(item.wn)+","+std::to_string(item.d)+",MxOpV2::"+op+","+(norm?"true":"false")+",MxPart::"+(item.stream_k?"kStreamK":"kTileRange")+","+std::to_string(width(item,n,k))+">";};
  auto grouped=[&](MxDecodeClassV2 const& item,bool gate){return "MxGroupedGemm<"+std::to_string(item.tn)+","+std::to_string(item.wn)+","+std::to_string(item.d)+",MxOpV2::"+(gate?"kSwiGLU,true":"kStore,false")+",Chain,"+(gate?std::string("MxNoUnitCompletionV2"):"MxMoeDownCompletionV2<"+std::string(m.routing_weights_fp32?"true":"false")+">")+","+std::to_string(gate?0:width(item,active*m.hidden,m.intermediate))+">";};
  StageProgramSpec s;s.kind=prompt?StageProgramKind::kPrompt:StageProgramKind::kDecode;s.includes={"tilemega/Codegen/tasks/mx/MoeProgramV2.cuh"};s.runtime_header="tilemega/Codegen/tasks/mx/MoeRuntimeV2.cuh";s.loop.end="kLayers";if(prompt)s.terminal_condition="past == 63";
  s.operand_parameters="Binding const& b,int past,backend::mx::MxSharedTokensV2* tokens,mx::MxMoeSharedMetadataV2* metadata";s.run_parameters="Binding const& b,int past,char* smem,backend::mx::MxSharedTokensV2* tokens,mx::MxMoeSharedMetadataV2* metadata";
  std::ostringstream decl;decl<<"  static constexpr int kLayers="<<m.layers.size()<<",kHidden="<<m.hidden<<",kIntermediate="<<m.intermediate<<",kCapacity="<<m.capacity<<",kBatch="<<m.batch<<",kThreads="<<c.threads<<",kGrid=104,kPastLo="<<past_lo<<",kPastHi="<<past_hi<<",kBodySmemBytes="<<Smem(m,c)<<",kQ="<<m.queries<<",kHeadDim="<<m.dim<<",kExperts="<<m.experts<<";\n"<<"  static constexpr bool kPrompt="<<(prompt?"true":"false")<<",kRoutingFp32="<<(m.routing_weights_fp32?"true":"false")<<",kRowQkNorm="<<(m.row_qk_norm?"true":"false")<<",kNormalizeTopK="<<(m.softmax_all?"false":"true")<<",kQkvStreamK="<<(c.qkv.stream_k?"true":"false")<<",kOStreamK="<<(c.o.stream_k?"true":"false")<<",kRouterStreamK="<<(c.router.stream_k?"true":"false")<<";\n  static constexpr float kEpsilon="<<std::hexfloat<<float(m.epsilon)<<"f;\n  using Binding=mx::MxMoeBindingV2<kLayers>;\n  using Rms="<<Step(m.layers[0].gate_chain.operations[0])<<";\n  using Gate="<<Step(m.layers[0].gate_chain.operations[1])<<";\n  using Chain=MxMoeGateChainV2<Rms,Gate>;\n  using Dispatch=MxMoeDispatchV2<kBatch,kHidden,kIntermediate,"<<c.gate_up.tn<<","<<c.down.tn<<",kRoutingFp32,kThreads"<<(m.softmax_all?",kExperts,kNormalizeTopK":"")<<">;";s.plan_declarations=decl.str();
  std::string ops="mx::MoeOperandsV2<Self>::",down=grouped(c.down,false);
  s.stages={
    {"Qkv",dense(c.qkv,"kStore",true,qn,m.hidden),shape(qn,m.hidden),ops+"Qkv(b,l,past,tokens)",ops+"QkvWeights(b,l)","Attn","",0},
    {"Attn","MxAttentionV2<"+std::to_string(m.dim)+","+std::to_string(m.queries)+","+std::to_string(c.b_min)+",true,"+std::to_string(c.attn_prefetch)+",kCapacity"+(m.row_qk_norm?","+std::to_string(m.hkv):std::string())+">","MxStaticAttentionShapeV2<kBatch,"+std::to_string(m.hkv)+",kThreads>",ops+"Attn(b,l,past)",ops+"Attn(b,l,past)","O","",1},
    {"O",dense(c.o,"kResidual",false,m.hidden,m.heads*m.dim),shape(m.hidden,m.heads*m.dim),ops+"O(b,l,tokens)",ops+"O(b,l,tokens)","Router","",2},
    {"Router","MxRouterGemmV2<"+std::to_string(c.router.tn)+","+std::to_string(c.router.d)+",MxPart::"+(c.router.stream_k?"kStreamK":"kTileRange")+",Dispatch,"+std::to_string(width(c.router,m.experts,m.hidden))+">",shape(m.experts,m.hidden),ops+"Router(b,l,past)",ops+"Router(b,l,past)","GateUp","",3},
    {"GateUp",grouped(c.gate_up,true),shape(2*m.intermediate,m.hidden),ops+"template Grouped<true>(b,l,metadata)","backend::mx::MxGroupedGemmOperandsV2{}","Down","",4},
    {"Down",down,shape(m.hidden,m.intermediate),ops+"template Grouped<false>(b,l,metadata)",ops+"template Grouped<false>(b,l,metadata)","Qkv",ops+"QkvWeights(b,l+1==kLayers?0:l+1)",5,down,"Head",ops+"Head(b,past)"}};
  s.tail_stages={{"Head","MxHead<"+std::to_string(c.head.tn)+","+std::to_string(c.head.wn)+","+std::to_string(c.head.d)+","+std::to_string(width(c.head,m.vocabulary,m.hidden))+">",shape(m.vocabulary,m.hidden),ops+"Head(b,past)",ops+"Head(b,past)","Qkv",ops+"QkvWeights(b,0)",6}};s.entry_prefetch_operands=ops+"QkvWeights(b,0)";
  auto rows=PrepareMxMoeBuffersV2(plan,{unsigned(c.qkv.tn),unsigned(c.o.tn),unsigned(c.router.tn),unsigned(c.gate_up.tn),unsigned(c.down.tn),unsigned(c.head.tn)});
  for(auto const& r:rows){if(r.buffer.dtype!="bf16"&&r.buffer.dtype!="f32"&&r.buffer.dtype!="i32")Fail("unsupported external dtype");s.external_buffers.push_back({r.buffer.name,r.buffer.dtype=="bf16"?"TM_DTYPE_BF16":r.buffer.dtype=="f32"?"TM_DTYPE_F32":"TM_DTYPE_I32",r.buffer.constant,r.buffer.per_batch,r.buffer.pack_json,std::to_string(int(r.slot)),int(r.layer)});}
  return EmitStageProgram(s);
}
}
