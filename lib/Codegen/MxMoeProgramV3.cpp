// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/MxMoeProgramV3.h>
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
int Smem(MxMoeModel const& m,MxMoeConfigV3 const& c){int slots=std::min(int(m.capacity/32),104);int wave=64+m.dim/2+m.queries*(m.dim+2);int scratch=std::max(slots*int(m.queries)*3,int(m.queries*(m.dim+2)));int attn=(8*wave+scratch+4)*4;int head=(std::max(GemmSmem(m.batch,c.head),int(m.batch)*128*8)+31)/16*16;int dispatch=(2*m.batch*m.experts+2*m.batch*8+m.experts+m.experts/32)*4+m.batch*m.experts;int maximum=std::max({32784,attn,head,dispatch,GemmSmem(m.batch,c.qkv),GemmSmem(m.batch,c.o),GemmSmem(m.batch,c.router)+16,GemmSmem(m.batch,c.gate_up),GemmSmem(m.batch,c.down)});return (maximum+15)/16*16;}
std::string Step(DmEpilogueOp const& op){std::ostringstream s;s<<"tilemega::codegen::DmEpilogueStep<tilemega::codegen::DmEpilogueKind("<<unsigned(op.kind)<<"),tilemega::codegen::DmActivation("<<unsigned(op.activation)<<"),tilemega::codegen::DmGatePair("<<unsigned(op.gate)<<"),"<<op.unit<<",tilemega::codegen::DmRounding("<<unsigned(op.input_rounding)<<"),tilemega::codegen::DmRounding("<<unsigned(op.output_rounding)<<"),tilemega::codegen::DmWriteKind("<<unsigned(op.residual_map.kind)<<"),"<<op.residual_map.factor<<">";return s.str();}
}
void CheckMxMoeConfigV3(frontend::ModelPlan const& plan,MxMoeConfigV3 const& c){
  auto m=AnalyzeMxMoe(plan);if(c.threads!=512)Fail("V3 MoE requires 512 threads");
  if(m.routing_weights_fp32)Fail("V3 MoE requires the BF16 routing contract");
  if(c.attn_version!=2&&c.attn_version!=3)Fail("invalid attention version");
  if(c.split!=1&&c.split!=2&&c.split!=4)Fail("invalid attention split");
  if(c.attn_version==2&&c.split!=1)Fail("attention V2 requires split1");
  if(c.split>1&&m.batch*m.hkv*c.split>104)Fail("attention split exceeds grid104");
  if(c.combine_mode=="unit_owner")Fail("combine_mode unit_owner is disqualified (moe_unit_owner litmus)");
  if(c.combine_mode!="owner16"&&c.combine_mode!="owner_tile"&&c.combine_mode!="unit_la")Fail("invalid combine_mode");
  if(c.route_table_mode!="both"&&c.route_table_mode!="weights"&&c.route_table_mode!="ordinal"&&c.route_table_mode!="serial")Fail("invalid route_table_mode");
  if(c.router.stream_k||c.router.tn!=16||c.router.wn!=1)Fail("router logits require TileRange TN16 WN1");
  if(m.token_id_bits!=32)Fail("native token binding requires I32");
  if(c.attn_version==2&&c.b_min!=1&&c.b_min!=2&&c.b_min!=4&&c.b_min!=8)Fail("invalid attention b_min");
  if(c.attn_prefetch<0||c.attn_prefetch>2)Fail("invalid attention prefetch");
  auto check=[&](MxDecodeClassV2 const& item,int n,int k,MxOpV2 op,bool grouped){
    if(n%item.tn)Fail("incomplete output tile");
    if(grouped&&item.stream_k)Fail("grouped units require TileRange");
    if(!MxGemmLegalV3(n,k,m.batch,item.tn,item.wn,item.d,op,item.stream_k?MxPart::kStreamK:MxPart::kTileRange,grouped))Fail("illegal GEMM geometry");
    if(grouped&&op==MxOpV2::kSwiGLU){if(item.prefetch!=0)Fail("gate/up has no producer-independent prefetch");}
    else {
      if(Width(item,n,k,c.threads)<item.d)Fail("prefetch below ring depth");
      if(item.r_next>=0){auto derived=item;derived.prefetch=-1;if(Width(item,n,k,c.threads)>Width(derived,n,k,c.threads))Fail("prefetch exceeds measured resource budget");}
    }
  };
  int qn=(m.queries+2)*m.hkv*m.dim,active=std::min(int(m.batch)*8,int(m.experts));
  check(c.qkv,qn,m.hidden,MxOpV2::kStore,false);check(c.o,m.hidden,m.heads*m.dim,MxOpV2::kResidual,false);
  check(c.router,m.experts,m.hidden,MxOpV2::kStore,false);if(c.router.wn!=1)Fail("router has at most eight tiles and WN must be one");
  check(c.gate_up,active*2*m.intermediate,m.hidden,MxOpV2::kSwiGLU,true);check(c.down,active*m.hidden,m.intermediate,MxOpV2::kStore,true);
  check(c.head,m.vocabulary,m.hidden,MxOpV2::kStore,false);if(c.head.stream_k||(c.head.wn!=4&&c.head.wn!=8))Fail("head requires TileRange WN4/8");
  auto loads=[&](MxDecodeClassV2 const& item,int n,int k){return Width(item,n,k,c.threads)*(item.tn/16);};
  auto wait=[&](MxDecodeClassV2 const& item,int next){if((item.d-1)*(item.tn/16+1)+next>63)Fail("next-stage wait exceeds 63");};
  wait(c.qkv,c.attn_prefetch*(m.dim/16));wait(c.o,loads(c.router,m.experts,m.hidden));wait(c.router,0);
  wait(c.gate_up,loads(c.down,active*m.hidden,m.intermediate));wait(c.down,loads(c.qkv,qn,m.hidden));wait(c.down,loads(c.head,m.vocabulary,m.hidden));
  if(Smem(m,c)+((m.batch*4+15)/16*16)+32+2576+16+16>65536)Fail("shared memory exceeds 64 KiB");
}
int MxMoeBodySmemBytesV3(frontend::ModelPlan const& plan,MxMoeConfigV3 const& c){
  CheckMxMoeConfigV3(plan,c);return Smem(AnalyzeMxMoe(plan),c);
}
std::string EmitMxMoeProgramV3(frontend::ModelPlan const& plan,MxMoeConfigV3 const& c,bool prompt,int past_lo,int past_hi){
  CheckMxMoeConfigV3(plan,c);auto m=AnalyzeMxMoe(plan);if(prompt){past_lo=0;past_hi=63;}
  if(past_lo<0||past_hi<past_lo||past_hi>=int(m.capacity)-1)Fail("invalid past interval");
  int qn=(m.queries+2)*m.hkv*m.dim,active=std::min(int(m.batch)*8,int(m.experts));
  auto width=[&](MxDecodeClassV2 const& item,int n,int k){return Width(item,n,k,c.threads);};
  auto shape=[](int n,int k){return "mx::MxStaticShapeV3<"+std::to_string(n)+","+std::to_string(k)+",kBatch,kThreads>";};
  auto dense=[&](MxDecodeClassV2 const& item,char const* op,bool norm,int n,int k){return "MxGemm<"+std::to_string(item.tn)+","+std::to_string(item.wn)+","+std::to_string(item.d)+",MxOpV2::"+op+","+(norm?"true":"false")+",MxPart::"+(item.stream_k?"kStreamK":"kTileRange")+","+std::to_string(width(item,n,k))+">";};
  auto grouped=[&](MxDecodeClassV2 const& item,bool gate){return "MxGroupedGemmV3<"+std::to_string(item.tn)+","+std::to_string(item.wn)+","+std::to_string(item.d)+",MxOpV2::"+(gate?"kSwiGLU,true":"kStore,false")+",Chain,"+std::to_string(gate?0:width(item,active*m.hidden,m.intermediate))+",false,MxCombineModeV3::"+(gate||c.combine_mode=="owner16"?"kOwner16":c.combine_mode=="owner_tile"?"kOwnerTile":c.combine_mode=="unit_owner"?"kUnitOwner":"kUnitLa")+">";};
  StageProgramSpec s;s.version=3;s.kind=prompt?StageProgramKind::kPrompt:StageProgramKind::kDecode;s.includes={"tilemega/Codegen/tasks/mx/MoeProgramV3.cuh"};s.runtime_header="tilemega/Codegen/tasks/mx/MoeRuntimeV3.cuh";s.loop.end="kLayers";if(prompt)s.terminal_condition="past == 63";
  s.operand_parameters="Binding const& b,int past,backend::mx::MxSharedTokensV2* tokens,mx::MxMoeSharedRouteTableV3* table";s.run_parameters="Binding const& b,int past,char* smem,backend::mx::MxSharedTokensV2* tokens,mx::MxMoeSharedRouteTableV3* table";
  std::ostringstream decl;decl<<"  static constexpr int kLayers="<<m.layers.size()<<",kHidden="<<m.hidden<<",kIntermediate="<<m.intermediate<<",kCapacity="<<m.capacity<<",kBatch="<<m.batch<<",kThreads="<<c.threads<<",kGrid=104,kAttentionVersion="<<c.attn_version<<",kAttentionSplit="<<c.split<<",kPastLo="<<past_lo<<",kPastHi="<<past_hi<<",kBodySmemBytes="<<Smem(m,c)<<",kQ="<<m.queries<<",kHeadDim="<<m.dim<<",kExperts="<<m.experts<<";\n"<<"  static constexpr bool kPrompt="<<(prompt?"true":"false")<<",kRoutingFp32="<<(m.routing_weights_fp32?"true":"false")<<",kRowQkNorm="<<(m.row_qk_norm?"true":"false")<<",kNormalizeTopK="<<(m.softmax_all?"false":"true")<<",kQkvStreamK="<<(c.qkv.stream_k?"true":"false")<<",kOStreamK="<<(c.o.stream_k?"true":"false")<<",kRouterStreamK="<<(c.router.stream_k?"true":"false")<<";\n  static constexpr float kEpsilon="<<std::hexfloat<<float(m.epsilon)<<"f;\n  using Binding=mx::MxMoeBindingV3<kLayers>;\n  using Rms="<<Step(m.layers[0].gate_chain.operations[0])<<";\n  using Gate="<<Step(m.layers[0].gate_chain.operations[1])<<";\n  using Chain=MxMoeGateChainV3<Rms,Gate>;\n";s.plan_declarations=decl.str();
  std::string ops="mx::MoeOperandsV3<Self>::",down=grouped(c.down,false);
  std::string gate="MxMoeGateUpV3<"+grouped(c.gate_up,true)+",kBatch,kHidden,kIntermediate,"+
      std::to_string(c.gate_up.tn)+","+std::to_string(c.down.tn)+",kExperts,kNormalizeTopK,"+(c.route_table_mode=="both"||c.route_table_mode=="weights"?"true":"false")+","+(c.route_table_mode=="both"||c.route_table_mode=="ordinal"?"true":"false")+">";
  std::string attention;
  if(c.attn_version==2)attention="MxAttentionV2<"+std::to_string(m.dim)+","+std::to_string(m.queries)+","+std::to_string(c.b_min)+",true,"+std::to_string(c.attn_prefetch)+",kCapacity,"+std::to_string(m.row_qk_norm?m.hkv:0)+">";
  else attention="MxAttentionV3<"+std::to_string(m.dim)+","+std::to_string(m.queries)+",true,"+std::to_string(c.attn_prefetch)+",kCapacity,"+std::to_string(m.row_qk_norm?m.hkv:0)+","+std::to_string(c.split)+">";
  s.stages={
    {"Qkv",dense(c.qkv,"kStore",true,qn,m.hidden),shape(qn,m.hidden),ops+"Qkv(b,l,past,tokens)",ops+"QkvWeights(b,l)","Attn","",0},
    {"Attn",attention,"mx::MxControlAttentionShapeV3<kBatch,"+std::to_string(m.hkv)+",kThreads>",ops+"Attn(b,l,past)",ops+"Attn(b,l,past)","O","",1},
    {"O",dense(c.o,"kResidual",false,m.hidden,m.heads*m.dim),shape(m.hidden,m.heads*m.dim),ops+"O(b,l,tokens)",ops+"O(b,l,tokens)","Router","",2},
    {"Router","MxRouterLogitsV3<16,"+std::to_string(c.router.d)+","+std::to_string(width(c.router,m.experts,m.hidden))+">",shape(m.experts,m.hidden),ops+"Router(b,l,past)",ops+"Router(b,l,past)","GateUp","",3},
    {"GateUp",gate,shape(2*m.intermediate,m.hidden),ops+"template Grouped<true>(b,l,past,table)","backend::mx::MxGroupedGemmOperandsV3<false>{}","Down","",4},
    {"Down",down,shape(m.hidden,m.intermediate),ops+"template Grouped<false>(b,l,past,table)",ops+"template Grouped<false>(b,l,past,table)","Qkv",ops+"QkvWeights(b,l+1==kLayers?0:l+1)",5,down,"Head",ops+"Head(b,past)"}};
  s.tail_stages={{"Head","MxHead<"+std::to_string(c.head.tn)+","+std::to_string(c.head.wn)+","+std::to_string(c.head.d)+","+std::to_string(width(c.head,m.vocabulary,m.hidden))+">",shape(m.vocabulary,m.hidden),ops+"Head(b,past)",ops+"Head(b,past)","Qkv",ops+"QkvWeights(b,0)",6}};s.entry_prefetch_operands=ops+"QkvWeights(b,0)";
  auto rows=PrepareMxMoeBuffersV2(plan,{unsigned(c.qkv.tn),unsigned(c.o.tn),unsigned(c.router.tn),unsigned(c.gate_up.tn),unsigned(c.down.tn),unsigned(c.head.tn)});
  for(auto const& r:rows){if(r.buffer.dtype!="bf16"&&r.buffer.dtype!="f32"&&r.buffer.dtype!="i32")Fail("unsupported external dtype");s.external_buffers.push_back({r.buffer.name,r.buffer.dtype=="bf16"?"TM_DTYPE_BF16":r.buffer.dtype=="f32"?"TM_DTYPE_F32":"TM_DTYPE_I32",r.buffer.constant,r.buffer.per_batch,r.buffer.pack_json,std::to_string(int(r.slot)),int(r.layer)});}
  return EmitStageProgram(s);
}
}
