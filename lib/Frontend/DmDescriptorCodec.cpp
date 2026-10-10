// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Codegen/MoeBinding.h>
#include <limits>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace tilemega::frontend {
namespace {
using namespace codegen;
using Words = std::vector<std::int64_t>;
template<class T> void Put(Words& out, T value) {
  auto word = static_cast<std::uint64_t>(value);
  if (word > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument("DM descriptor word exceeds signed 64-bit range");
  out.push_back(static_cast<std::int64_t>(word));
}
void PutWrite(Words& out, DmWriteMap const& x) {
  Put(out,x.kind); Put(out,x.factor); Put(out,x.layout); Put(out,x.rows);
}
struct Reader {
  llvm::ArrayRef<std::int64_t> words;
  std::size_t cursor = 0;
  Reader(mlir::Attribute attr, std::size_t count) {
    auto array = llvm::dyn_cast_or_null<mlir::DenseI64ArrayAttr>(attr);
    if (!array || array.size()!=count)
      throw std::invalid_argument("DM descriptor has invalid word count or type");
    words=array.asArrayRef();
  }
  std::uint64_t U64() {
    if (cursor>=words.size() || words[cursor]<0)
      throw std::invalid_argument("DM descriptor has negative or missing word");
    return words[cursor++];
  }
  std::uint32_t U32() {
    auto value=U64();
    if (value>std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("DM descriptor word exceeds 32-bit range");
    return static_cast<std::uint32_t>(value);
  }
  template<class T> T Enum(std::uint32_t maximum) {
    auto value=U32();
    if (value>maximum) throw std::invalid_argument("invalid DM descriptor enum");
    return static_cast<T>(value);
  }
  DmWriteMap Write() {
    DmWriteMap x; x.kind=Enum<DmWriteKind>(3); x.factor=U32();
    x.layout=U32(); x.rows=U32();
    if (!x.factor || (x.kind!=DmWriteKind::kPixelShuffle && x.factor!=1))
      throw std::invalid_argument("invalid DM write-map factor");
    return x;
  }
};

std::string WriteLiteral(DmWriteMap const& x) {
  return "{static_cast<DmWriteKind>("+std::to_string(unsigned(x.kind))+"u), "+
      std::to_string(x.factor)+"u, "+std::to_string(x.layout)+"u, "+
      std::to_string(x.rows)+"u}";
}
template<class T, std::size_t N> void ArrayLiteral(std::ostream& out, T const (&values)[N]) {
  out<<'{'; for (std::size_t i=0;i<N;++i) { if(i)out<<", "; out<<values[i]<<"ull"; } out<<'}';
}
}  // namespace

void ValidateDmModelPlan(ModelPlan const& plan) {
  if (plan.forward_token_axis && !plan.forward)
    throw std::invalid_argument("forward token axis requires a forward plan");
  if (plan.forward && (!plan.dm || plan.serving || plan.serving_seq <= 0 ||
                       plan.serving_capacity != 0))
    throw std::invalid_argument("forward plans require DM, a positive seq and no KV capacity");
  if (plan.forward) {
    for (auto const& b : plan.buffers)
      if (b.per_past || b.per_total)
        throw std::invalid_argument("forward buffer cannot depend on KV state");
    for (auto const& stage : plan.stages) {
      switch (stage.kind) {
        case PlanTaskKind::kRoPE: case PlanTaskKind::kKVAppend:
        case PlanTaskKind::kAttention: case PlanTaskKind::kFusedAttention:
        case PlanTaskKind::kAttentionMerge: case PlanTaskKind::kArgmaxReduce:
        case PlanTaskKind::kEmbedding: case PlanTaskKind::kQKNorm:
          throw std::invalid_argument("decoder task is invalid in a forward plan");
        default: break;
      }
    }
    if (plan.forward_token_axis)
      for (auto const& gemm : plan.gemms)
        if (gemm.access.rows_per_batch)
          throw std::invalid_argument("token-axis forward GEMM cannot use batch rows");
  }
  if(!plan.dm) {
    bool extended=!plan.convolutions.empty();
    for(auto const& buffer:plan.buffers)extended|=buffer.layout.rank!=0;
    for(auto const& gemm:plan.gemms) {
      auto const& a=gemm.access;
      extended|=a.a!=DmAAccess::kDense || a.b!=DmBAccess::kDense ||
          a.rows_per_batch || a.a_row_stride || a.a_row_offset ||
          a.conv!=kDmNoIndex || a.rows!=kDmNoIndex || a.binding!=kDmNoIndex ||
          a.a_scale!=kDmNoIndex || a.expert_stride ||
          a.binding_blocks || a.binding_rows || a.experts || a.block_rows || a.routing_topk ||
          a.write.kind!=DmWriteKind::kDense || a.write.factor!=1 ||
          a.write.layout!=kDmNoIndex || a.write.rows!=kDmNoIndex ||
          gemm.chain.count || gemm.chain.side_count;
    }
    for(auto const& stage:plan.stages)
      extended|=stage.kind>=PlanTaskKind::kDepthwiseConv ||
          stage.conv!=kDmNoIndex || stage.rows_per_batch || stage.binding_producer!=kDmNoIndex ||
          stage.norm_epsilon!=0.0f || stage.chain.count || stage.chain.side_count || stage.partial_rows_per_image;
    for(auto const& stage:plan.stages)extended|=stage.moe.step!=DmMoeStep::kNone;
    if(extended)throw std::invalid_argument("extended descriptors require the DM device ABI");
    return;
  }
  auto buffer=[&](std::uint32_t id,bool required=false) {
    if((id==kDmNoIndex && required) || (id!=kDmNoIndex && id>=plan.buffers.size()))
      throw std::invalid_argument("DM descriptor buffer outside model plan");
  };
  auto write=[&](DmWriteMap const& map) {
    buffer(map.layout,map.kind==DmWriteKind::kNCHW || map.kind==DmWriteKind::kPixelShuffle);
    buffer(map.rows,map.kind==DmWriteKind::kRowScatter);
  };
  for(auto const& conv:plan.convolutions) {
    buffer(conv.input_layout,true); buffer(conv.output_layout,true);
  }
  auto validate_chain=[&](DmEpilogueChain const& chain) {
    if(chain.count>8 || chain.side_count>5)
      throw std::invalid_argument("DM epilogue capacity exceeded");
    for(unsigned i=0;i<chain.count;++i) {
      auto const& op=chain.operations[i];
      if((op.norm_width || op.norm_epsilon!=0) && (!op.norm_width || !(op.norm_epsilon>0) ||
          !std::isfinite(op.norm_epsilon) || (op.kind!=DmEpilogueKind::kDeferredLayerNorm &&
          op.kind!=DmEpilogueKind::kResidualLN && op.kind!=DmEpilogueKind::kDeferredRMSNorm)))
        throw std::invalid_argument("invalid per-operation normalization geometry");
      unsigned required=0;
      switch(op.kind) {
        case DmEpilogueKind::kBias: case DmEpilogueKind::kScale:
        case DmEpilogueKind::kResidual: case DmEpilogueKind::kDeferredRMSNorm: required=1; break;
        case DmEpilogueKind::kDeferredLayerNorm: required=3; break;
        case DmEpilogueKind::kResidualLN: required=4; break;
        case DmEpilogueKind::kActivation: case DmEpilogueKind::kGatePair: break;
      }
      for(unsigned p=0;p<4;++p)buffer(op.parameter[p],p<required);
      write(op.residual_map);
    }
    for(unsigned i=0;i<chain.side_count;++i) {
      buffer(chain.side[i].buffer,true); buffer(chain.side[i].auxiliary);
    }
  };
  for(auto const& gemm:plan.gemms) {
    auto const& a=gemm.access;
    if(a.conv!=kDmNoIndex && a.conv>=plan.convolutions.size())
      throw std::invalid_argument("DM GEMM convolution outside geometry table");
    buffer(a.rows,a.a==DmAAccess::kRowGather);
    buffer(a.binding,a.a==DmAAccess::kRowGather || a.b==DmBAccess::kExpertIndirect);
    buffer(a.a_scale); write(a.write);
    validate_chain(gemm.chain);
  }

  unsigned moe_tokens=0;
  for(unsigned i=0;i<plan.stages.size();++i) {
    auto const& stage=plan.stages[i];
    auto const& moe=stage.moe;
    if(stage.kind==PlanTaskKind::kMoETopK || stage.kind==PlanTaskKind::kMoECombine) {
      unsigned capacity=0;
      unsigned tokens=plan.forward_token_axis?plan.serving_seq:
          (plan.serving && moe.top_k && moe.row_capacity%moe.top_k==0?moe.row_capacity/moe.top_k:0);
      if(!tokens || tokens>4096 || (plan.serving && (plan.serving_seq<=0 ||
          tokens%plan.serving_seq || tokens/plan.serving_seq>64)) ||
          (moe_tokens && moe_tokens!=tokens) || !MoeVirtualCapacity(tokens,moe.top_k,
          moe.experts,moe.block_rows,moe.grouped,&capacity) ||
          moe.experts>128 || moe.top_k>32 || capacity!=moe.binding_capacity ||
          moe.row_capacity!=std::uint64_t(tokens)*moe.top_k ||
          moe.router_gemm>=plan.gemms.size() || !moe.chunk_tokens ||
          moe.step==DmMoeStep::kNone || unsigned(moe.step)>unsigned(DmMoeStep::kCombine))
        throw std::invalid_argument("invalid MoE stage capacity or router contract");
      moe_tokens=tokens;
      bool combine=stage.kind==PlanTaskKind::kMoECombine;
      if(combine!=(moe.step==DmMoeStep::kCombine) || !stage.group ||
          (!combine && (stage.width!=moe.top_k || stage.extent!=moe.experts)) ||
          (moe.step==DmMoeStep::kSelectAndDispatch &&
           (moe.row_capacity>4096 || stage.group<tokens)) ||
          ((moe.step==DmMoeStep::kHistogram || moe.step==DmMoeStep::kPrefix) && !moe.grouped) ||
          (combine && (stage.width<16 || stage.width>256 || stage.width%16 ||
                       stage.group>128 || !stage.extent)))
        throw std::invalid_argument("invalid MoE stage specialization");
      auto typed=[&](unsigned operand,char const* dtype,bool optional=false) {
        auto id=stage.operands[operand];buffer(id,!optional);
        if(id!=kDmNoIndex && plan.buffers[id].dtype!=dtype)
          throw std::invalid_argument("MoE stage operand has incompatible dtype");
      };
      if(combine) {
        for(unsigned j=0;j<4;++j)typed(j,"bf16");typed(4,"f32",true);
      }else {
        typed(0,"f32");typed(1,"i32");typed(2,"i32");typed(3,"bf16");
        for(unsigned j=4;j<9;++j)typed(j,"i32");
      }
    }else if(moe.step!=DmMoeStep::kNone)
      throw std::invalid_argument("MoE configuration is attached to another task kind");
    validate_chain(stage.chain);
    if(stage.kind==PlanTaskKind::kDwPwFused) {
      if(stage.gemm>=plan.gemms.size() || stage.conv>=plan.convolutions.size())
        throw std::invalid_argument("fused dw-pw descriptor outside model tables");
      auto const& conv=plan.convolutions[stage.conv];auto const& g=plan.gemms[stage.gemm];
      buffer(stage.operands[0],true);buffer(stage.operands[1],true);
      if(conv.c!=conv.k || conv.c%8 || stage.width!=conv.c || g.k!=conv.c ||
          stage.rows_per_batch!=conv.p*conv.q || g.access.rows_per_batch!=stage.rows_per_batch ||
          g.access.a!=DmAAccess::kDense || g.access.b!=DmBAccess::kDense ||
          g.access.a_scale!=kDmNoIndex || stage.chain.side_count ||
          stage.chain.store_rounding!=DmRounding::kBF16 ||
          conv.input_layout!=stage.operands[0] || conv.output_layout!=g.a ||
          stage.operands[2]!=kDmNoIndex || stage.operands[3]!=kDmNoIndex)
        throw std::invalid_argument("invalid fused dw-pw ownership or operand contract");
      for(unsigned i=0;i<stage.chain.count;++i) {
        auto kind=stage.chain.operations[i].kind;
        if(kind!=DmEpilogueKind::kBias && kind!=DmEpilogueKind::kScale && kind!=DmEpilogueKind::kActivation)
          throw std::invalid_argument("unsupported fused depthwise epilogue");
      }
    }
    if(stage.partial_rows_per_image && (stage.kind!=PlanTaskKind::kGlobalPoolReduce || !stage.group ||
       stage.partial_rows_per_image!=(std::uint64_t(stage.rows_per_batch)+stage.group-1)/stage.group))
      throw std::invalid_argument("invalid per-image partial count");
    if(stage.kind==PlanTaskKind::kLayerNorm || stage.kind==PlanTaskKind::kEmbeddingSum ||
       stage.kind==PlanTaskKind::kLayoutConvert || stage.kind==PlanTaskKind::kPool ||
       stage.kind==PlanTaskKind::kGlobalPoolReduce || stage.kind==PlanTaskKind::kEncoderAttention ||
       stage.kind==PlanTaskKind::kDepthwiseConv) {
      if(!stage.width || stage.width>4096 || !stage.group || stage.group>1024)
        throw std::invalid_argument("invalid DM scalar task geometry");
      auto typed=[&](unsigned operand,char const* dtype,bool optional=false) {
        auto id=stage.operands[operand];buffer(id,!optional);
        if(id!=kDmNoIndex && plan.buffers[id].dtype!=dtype)
          throw std::invalid_argument("DM scalar operand has incompatible dtype");
      };
      if(stage.kind==PlanTaskKind::kLayerNorm) {
        if(stage.group%4 || !(stage.norm_epsilon>0) || !std::isfinite(stage.norm_epsilon))
          throw std::invalid_argument("invalid LayerNorm geometry or epsilon");
        for(unsigned o=0;o<4;++o)typed(o,"bf16");
        typed(4,"f32",true);
      }else if(stage.kind==PlanTaskKind::kEmbeddingSum) {
        for(unsigned o=0;o<2;++o)typed(o,"i64");
        for(unsigned o=2;o<6;++o)typed(o,"bf16");
        typed(6,"f32",true);
        auto const& types=plan.buffers[stage.operands[3]].layout;
        auto const& positions=plan.buffers[stage.operands[4]].layout;
        if(!stage.extent || types.rank!=2 || positions.rank!=2 ||
           !types.logical[0] || types.logical[1]!=stage.width ||
           positions.logical[1]!=stage.width ||
           (plan.forward && positions.logical[0]<unsigned(plan.serving_seq)))
          throw std::invalid_argument("invalid embedding table extents");
      }else if(stage.kind==PlanTaskKind::kDepthwiseConv) {
        for(unsigned o=0;o<3;++o)typed(o,"bf16");typed(3,"f32",true);
        if((stage.width!=32 && stage.width!=64 && stage.width!=128 && stage.width!=256) ||
           stage.group>128 || !stage.extent || stage.conv>=plan.convolutions.size() ||
           stage.chain.side_count || stage.chain.store_rounding!=DmRounding::kBF16)
          throw std::invalid_argument("invalid depthwise task geometry or chain");
        unsigned gates=0;
        for(unsigned op=0;op<stage.chain.count;++op) {
          auto const& step=stage.chain.operations[op];
          if(step.kind!=DmEpilogueKind::kBias && step.kind!=DmEpilogueKind::kScale &&
             step.kind!=DmEpilogueKind::kActivation && step.kind!=DmEpilogueKind::kGatePair)
            throw std::invalid_argument("unsupported depthwise epilogue");
          if(step.kind==DmEpilogueKind::kGatePair) {
            ++gates;if(step.gate!=DmGatePair::kSimpleGate)
              throw std::invalid_argument("depthwise requires SimpleGate");
          }
        }
        auto const& conv=plan.convolutions[stage.conv];
        if(gates>1 || (gates && conv.c%16) || conv.c!=conv.k ||
           stage.extent*(gates?2:1)!=conv.c ||
           conv.input_layout!=stage.operands[0] || conv.output_layout!=stage.operands[2] ||
           std::uint64_t(conv.p)*conv.q!=stage.rows_per_batch)
          throw std::invalid_argument("depthwise ownership differs from its convolution");
        for(auto id:{stage.operands[0],stage.operands[2]})
          if(plan.buffers[id].layout.kind!=DmLayout::kNHWC)
            throw std::invalid_argument("depthwise requires NHWC buffers");
      }else if(stage.kind==PlanTaskKind::kPool) {
        typed(0,"bf16");typed(1,"bf16");
        if(stage.width<32 || stage.width>256 || stage.width%32 || !stage.extent ||
           stage.conv>=plan.convolutions.size())
          throw std::invalid_argument("invalid pool channel tile or geometry");
        auto const& conv=plan.convolutions[stage.conv];
        if(conv.input_layout!=stage.operands[0] || conv.output_layout!=stage.operands[1] ||
           stage.extent!=conv.c || conv.k!=conv.c ||
           std::uint64_t(conv.p)*conv.q!=stage.rows_per_batch)
          throw std::invalid_argument("pool ownership differs from its window geometry");
        for(auto id:{stage.operands[0],stage.operands[1]})
          if(plan.buffers[id].layout.kind!=DmLayout::kNHWC)
            throw std::invalid_argument("pool requires NHWC buffers");
      }else if(stage.kind==PlanTaskKind::kEncoderAttention) {
        typed(0,"bf16");typed(1,"bf16");typed(2,"i64",true);
        if((stage.width!=128 && stage.width!=384 && stage.width!=512) ||
           (stage.group!=64 && stage.group!=128) || !stage.extent ||
           stage.rows_per_batch!=stage.width || unsigned(plan.serving_seq)!=stage.width)
          throw std::invalid_argument("invalid encoder attention geometry");
      }else if(stage.kind==PlanTaskKind::kGlobalPoolReduce) {
        typed(0,"f32");typed(1,"f32");
        typed(2,"bf16",true);
        auto const& partial=plan.buffers[stage.operands[0]].layout;
        if(stage.width<32 || stage.width>256 || stage.width%32 || !stage.extent ||
           !stage.rows_per_batch || partial.rank!=3 || partial.logical[2]!=stage.extent)
          throw std::invalid_argument("invalid global pool partial or channel tile");
      }else {
        typed(0,"bf16");typed(1,"bf16");
        auto const& layout=plan.buffers[stage.operands[1]].layout;
        if(layout.kind!=DmLayout::kNHWC || layout.rank!=4 || layout.logical[3]!=stage.width)
          throw std::invalid_argument("layout conversion requires NHWC output geometry");
      }
    }
    if(stage.conv!=kDmNoIndex && stage.conv>=plan.convolutions.size())
      throw std::invalid_argument("DM stage convolution outside geometry table");
    if(stage.binding_producer!=kDmNoIndex) {
      if(stage.binding_producer>=plan.stages.size() || stage.binding_producer==i)
        throw std::invalid_argument("DM binding producer outside stage table or self-dependent");
      if(stage.kind!=PlanTaskKind::kGemm || stage.gemm>=plan.gemms.size() ||
         (plan.gemms[stage.gemm].access.a!=DmAAccess::kRowGather &&
          plan.gemms[stage.gemm].access.b!=DmBAccess::kExpertIndirect))
        throw std::invalid_argument("DM binding dependency requires a gathered or expert GEMM");
    }
  }
}

mlir::DenseI64ArrayAttr EncodeDm(mlir::Builder& b, codegen::ConvDesc const& x) {
  Words w;
  for (auto value:{x.n,x.h,x.w,x.c,x.k,x.r,x.s,x.stride_h,x.stride_w,
                  x.pad_h,x.pad_w,x.dilation_h,x.dilation_w,x.p,x.q,
                  x.input_layout,x.output_layout}) Put(w,value);
  auto attr=b.getDenseI64ArrayAttr(w); (void)DecodeDmConv(attr); return attr;
}
codegen::ConvDesc DecodeDmConv(mlir::Attribute attr) {
  Reader r(attr,17); codegen::ConvDesc x;
  x.n=r.U32(); x.h=r.U32(); x.w=r.U32(); x.c=r.U32(); x.k=r.U32();
  x.r=r.U32(); x.s=r.U32(); x.stride_h=r.U32(); x.stride_w=r.U32();
  x.pad_h=r.U32(); x.pad_w=r.U32(); x.dilation_h=r.U32(); x.dilation_w=r.U32();
  x.p=r.U32(); x.q=r.U32(); x.input_layout=r.U32(); x.output_layout=r.U32();
  if (!x.n || !x.h || !x.w || !x.c || !x.k || !x.r || !x.s ||
      !x.stride_h || !x.stride_w || !x.dilation_h || !x.dilation_w)
    throw std::invalid_argument("DM convolution has zero extent or stride");
  auto extent=[](std::uint32_t input,std::uint32_t pad,std::uint32_t filter,
                 std::uint32_t dilation,std::uint32_t stride) {
    auto available=std::uint64_t(input)+2*std::uint64_t(pad);
    auto footprint=std::uint64_t(dilation)*(filter-1)+1;
    return available<footprint ? 0 : (available-footprint)/stride+1;
  };
  if (!x.p || !x.q || extent(x.h,x.pad_h,x.r,x.dilation_h,x.stride_h)!=x.p ||
      extent(x.w,x.pad_w,x.s,x.dilation_w,x.stride_w)!=x.q)
    throw std::invalid_argument("DM convolution output shape disagrees with geometry");
  return x;
}

mlir::DenseI64ArrayAttr EncodeDm(mlir::Builder& b, codegen::DmBufferLayout const& x) {
  Words w; Put(w,x.kind); Put(w,x.rank);
  for(auto v:x.logical)Put(w,v); for(auto v:x.physical)Put(w,v);
  for(auto v:x.strides)Put(w,v);
  for(auto v:{x.halo_top,x.halo_bottom,x.halo_left,x.halo_right})Put(w,v);
  Put(w,x.fill); auto attr=b.getDenseI64ArrayAttr(w);
  (void)DecodeDmLayout(attr); return attr;
}
codegen::DmBufferLayout DecodeDmLayout(mlir::Attribute attr) {
  Reader r(attr,19); codegen::DmBufferLayout x;
  x.kind=r.Enum<DmLayout>(1); x.rank=r.U32();
  for(auto& v:x.logical)v=r.U32(); for(auto& v:x.physical)v=r.U32();
  for(auto& v:x.strides)v=r.U64();
  x.halo_top=r.U32(); x.halo_bottom=r.U32(); x.halo_left=r.U32(); x.halo_right=r.U32();
  x.fill=r.Enum<DmFill>(1);
  if(x.rank>4 || (x.kind==DmLayout::kNHWC && x.rank!=4))
    throw std::invalid_argument("invalid DM buffer layout rank");
  for(unsigned i=0;i<x.rank;++i)
    if(!x.logical[i] || x.physical[i]<x.logical[i] || !x.strides[i])
      throw std::invalid_argument("DM physical layout does not contain logical shape");
  if(x.kind==DmLayout::kNHWC &&
     (std::uint64_t(x.logical[1])+x.halo_top+x.halo_bottom>x.physical[1] ||
      std::uint64_t(x.logical[2])+x.halo_left+x.halo_right>x.physical[2] ||
      (x.physical[3]%8 && !(x.logical[3]<=4 && x.physical[3]==4)) ||
      x.strides[3]!=1 || x.strides[2]%8))
    throw std::invalid_argument("DM NHWC layout has invalid halo or channel alignment");
  for(unsigned i=1;i<x.rank;++i)
    if(x.strides[i]>std::numeric_limits<std::uint64_t>::max()/x.physical[i] ||
       x.strides[i-1]<x.strides[i]*x.physical[i])
      throw std::invalid_argument("DM physical layout has overlapping strides");
  return x;
}

mlir::DenseI64ArrayAttr EncodeDm(mlir::Builder& b, codegen::DmGemmAccess const& x) {
  Words w;
  Put(w,x.a); Put(w,x.b); Put(w,x.rows_per_batch); Put(w,x.a_row_stride);
  Put(w,x.a_row_offset); Put(w,x.conv); Put(w,x.rows); Put(w,x.binding);
  Put(w,x.a_scale); Put(w,x.expert_stride); PutWrite(w,x.write);
  if(x.binding_blocks || x.binding_rows || x.experts || x.block_rows || x.routing_topk) {
    Put(w,x.binding_blocks); Put(w,x.binding_rows); Put(w,x.experts); Put(w,x.block_rows);
    if(x.routing_topk)Put(w,x.routing_topk);
  }
  auto attr=b.getDenseI64ArrayAttr(w); (void)DecodeDmAccess(attr); return attr;
}
codegen::DmGemmAccess DecodeDmAccess(mlir::Attribute attr) {
  auto array=llvm::dyn_cast_or_null<mlir::DenseI64ArrayAttr>(attr);
  Reader r(attr,array && (array.size()==18 || array.size()==19) ? array.size() : 14); codegen::DmGemmAccess x;
  x.a=r.Enum<DmAAccess>(2); x.b=r.Enum<DmBAccess>(1);
  x.rows_per_batch=r.U32(); x.a_row_stride=r.U32(); x.a_row_offset=r.U32();
  x.conv=r.U32(); x.rows=r.U32(); x.binding=r.U32(); x.a_scale=r.U32();
  x.expert_stride=r.U64(); x.write=r.Write();
  if(r.words.size()>=18) {
    x.binding_blocks=r.U32(); x.binding_rows=r.U32(); x.experts=r.U32(); x.block_rows=r.U32();
    if(!x.binding_blocks || !x.binding_rows || !x.experts || !x.block_rows ||
       x.binding_blocks>std::uint32_t(std::numeric_limits<int>::max())/x.block_rows)
      throw std::invalid_argument("invalid DM binding capacity");
    if(r.words.size()==19) {
      x.routing_topk=r.U32();
      if(!x.routing_topk || x.routing_topk>x.experts || x.b!=DmBAccess::kExpertIndirect)
        throw std::invalid_argument("invalid expert row-scatter top-k");
    }
  }
  if((x.a==DmAAccess::kIm2Col && x.conv==kDmNoIndex) ||
     (x.a==DmAAccess::kRowGather && (x.rows==kDmNoIndex || x.binding==kDmNoIndex)) ||
     (x.b==DmBAccess::kExpertIndirect && (x.binding==kDmNoIndex || !x.expert_stride)) ||
     (x.write.kind==DmWriteKind::kRowScatter && x.write.rows==kDmNoIndex))
    throw std::invalid_argument("DM GEMM access lacks its geometry or binding source");
  return x;
}

mlir::DenseI64ArrayAttr EncodeDm(mlir::Builder& b,codegen::DmMoeStage const& x) {
  Words w;Put(w,x.step);
  for(auto v:{x.experts,x.top_k,x.block_rows,x.binding_capacity,x.row_capacity,
      x.router_gemm,x.chunk_tokens})Put(w,v);
  Put(w,x.grouped);auto attr=b.getDenseI64ArrayAttr(w);
  (void)DecodeDmMoeStage(attr);return attr;
}
codegen::DmMoeStage DecodeDmMoeStage(mlir::Attribute attr) {
  Reader r(attr,9);DmMoeStage x;x.step=r.Enum<DmMoeStep>(6);
  x.experts=r.U32();x.top_k=r.U32();x.block_rows=r.U32();
  x.binding_capacity=r.U32();x.row_capacity=r.U32();x.router_gemm=r.U32();x.chunk_tokens=r.U32();
  auto grouped=r.U32();if(grouped>1)throw std::invalid_argument("invalid MoE binding policy");
  x.grouped=grouped;
  if(x.step!=DmMoeStep::kNone && (!x.experts || x.experts>128 || !x.top_k ||
      x.top_k>32 || x.top_k>x.experts || !x.block_rows || !x.binding_capacity ||
      !x.row_capacity || x.router_gemm==kDmNoIndex || !x.chunk_tokens ||
      (!x.grouped && x.block_rows!=1)))
    throw std::invalid_argument("invalid MoE stage descriptor");
  return x;
}

mlir::DictionaryAttr EncodeDm(mlir::Builder& b, codegen::DmEpilogueChain const& x) {
  if(x.count>8 || x.side_count>5) throw std::invalid_argument("DM epilogue capacity exceeded");
  llvm::SmallVector<mlir::Attribute> ops,side;
  for(unsigned i=0;i<x.count;++i) {
    auto const& o=x.operations[i]; Words w; Put(w,o.kind);
    for(auto p:o.parameter)Put(w,p); PutWrite(w,o.residual_map);
    Put(w,o.activation); Put(w,o.gate); Put(w,o.unit);
    Put(w,o.input_rounding); Put(w,o.output_rounding);
    if(o.norm_width || o.norm_epsilon!=0) {
      Put(w,o.norm_width);std::uint32_t bits;
      std::memcpy(&bits,&o.norm_epsilon,sizeof(bits));Put(w,bits);
    }
    ops.push_back(b.getDenseI64ArrayAttr(w));
  }
  for(unsigned i=0;i<x.side_count;++i) {
    auto const& s=x.side[i]; Words w;
    Put(w,s.kind); Put(w,s.buffer); Put(w,s.auxiliary); Put(w,s.count);
    side.push_back(b.getDenseI64ArrayAttr(w));
  }
  auto attr=b.getDictionaryAttr({b.getNamedAttr("operations",b.getArrayAttr(ops)),
      b.getNamedAttr("side",b.getArrayAttr(side)),
      b.getNamedAttr("store_rounding",b.getI64IntegerAttr(unsigned(x.store_rounding)))});
  (void)DecodeDmChain(attr); return attr;
}
codegen::DmEpilogueChain DecodeDmChain(mlir::Attribute attr) {
  auto dict=llvm::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
  auto ops=dict ? dict.getAs<mlir::ArrayAttr>("operations") : mlir::ArrayAttr{};
  auto side=dict ? dict.getAs<mlir::ArrayAttr>("side") : mlir::ArrayAttr{};
  auto rounding=dict ? dict.getAs<mlir::IntegerAttr>("store_rounding") : mlir::IntegerAttr{};
  if(!ops || !side || !rounding || ops.size()>8 || side.size()>5 ||
     rounding.getInt()<0 || rounding.getInt()>1)
    throw std::invalid_argument("invalid DM epilogue chain");
  DmEpilogueChain x; x.count=ops.size(); x.side_count=side.size();
  x.store_rounding=static_cast<DmRounding>(rounding.getInt());
  for(unsigned i=0;i<x.count;++i) {
    auto words=llvm::dyn_cast<mlir::DenseI64ArrayAttr>(ops[i]);
    Reader r(ops[i],words && words.size()==16?16:14);
    auto& o=x.operations[i]; o.kind=r.Enum<DmEpilogueKind>(7);
    for(auto& p:o.parameter)p=r.U32(); o.residual_map=r.Write();
    o.activation=r.Enum<DmActivation>(5); o.gate=r.Enum<DmGatePair>(1); o.unit=r.U32();
    o.input_rounding=r.Enum<DmRounding>(1); o.output_rounding=r.Enum<DmRounding>(1);
    if(r.words.size()==16) {
      o.norm_width=r.U32();auto bits=r.U32();std::memcpy(&o.norm_epsilon,&bits,sizeof(bits));
      if(!o.norm_width || !(o.norm_epsilon>0) || !std::isfinite(o.norm_epsilon) ||
          (o.kind!=DmEpilogueKind::kDeferredLayerNorm && o.kind!=DmEpilogueKind::kResidualLN &&
           o.kind!=DmEpilogueKind::kDeferredRMSNorm))
        throw std::invalid_argument("invalid per-operation normalization geometry");
    }
    if(o.kind==DmEpilogueKind::kGatePair && (!o.unit || (o.unit&(o.unit-1))))
      throw std::invalid_argument("DM gate interleave unit must be a power of two");
  }
  unsigned seen=0;
  for(unsigned i=0;i<x.side_count;++i) {
    Reader r(side[i],4); auto& s=x.side[i]; s.kind=r.Enum<DmSideOutputKind>(4);
    s.buffer=r.U32(); s.auxiliary=r.U32(); s.count=r.U32();
    auto bit=1u<<unsigned(s.kind);
    if(s.buffer==kDmNoIndex || (seen&bit) ||
       (s.kind==DmSideOutputKind::kTopKPartial && !s.count))
      throw std::invalid_argument("invalid or duplicate DM epilogue side output");
    seen|=bit;
  }
  return x;
}

std::string EmitDm(codegen::ConvDesc const& x) {
  std::ostringstream out; out<<'{'; bool first=true;
  for(auto v:{x.n,x.h,x.w,x.c,x.k,x.r,x.s,x.stride_h,x.stride_w,x.pad_h,
              x.pad_w,x.dilation_h,x.dilation_w,x.p,x.q,x.input_layout,x.output_layout}) {
    if(!first)out<<", "; first=false; out<<v<<'u';
  }
  out<<'}'; return out.str();
}
std::string EmitDm(codegen::DmBufferLayout const& x) {
  std::ostringstream out;
  out<<"{static_cast<DmLayout>("<<unsigned(x.kind)<<"u), "<<x.rank<<"u, ";
  ArrayLiteral(out,x.logical); out<<", "; ArrayLiteral(out,x.physical); out<<", ";
  ArrayLiteral(out,x.strides);
  out<<", "<<x.halo_top<<"u, "<<x.halo_bottom<<"u, "<<x.halo_left<<"u, "
     <<x.halo_right<<"u, static_cast<DmFill>("<<unsigned(x.fill)<<"u)}";
  return out.str();
}
std::string EmitDm(codegen::DmGemmAccess const& x) {
  std::ostringstream out;
  out<<"{static_cast<DmAAccess>("<<unsigned(x.a)<<"u), static_cast<DmBAccess>("
     <<unsigned(x.b)<<"u), "<<x.rows_per_batch<<"u, "<<x.a_row_stride<<"u, "
     <<x.a_row_offset<<"u, "<<x.conv<<"u, "<<x.rows<<"u, "<<x.binding<<"u, "
     <<x.a_scale<<"u, "<<x.expert_stride<<"ull, "<<WriteLiteral(x.write);
  if(x.binding_blocks || x.binding_rows || x.experts || x.block_rows || x.routing_topk)
    out<<", "<<x.binding_blocks<<"u, "<<x.binding_rows<<"u, "<<x.experts<<"u, "<<x.block_rows<<'u';
  if(x.routing_topk)out<<", "<<x.routing_topk<<'u';
  out<<'}';
  return out.str();
}
std::string EmitDm(codegen::DmMoeStage const& x) {
  std::ostringstream out;out<<"{static_cast<DmMoeStep>("<<unsigned(x.step)<<"u), ";
  for(auto v:{x.experts,x.top_k,x.block_rows,x.binding_capacity,x.row_capacity,
      x.router_gemm,x.chunk_tokens})out<<v<<"u, ";
  out<<(x.grouped?"true":"false")<<'}';return out.str();
}
std::string EmitDm(codegen::DmEpilogueChain const& x) {
  std::ostringstream out; out<<'{'<<x.count<<"u, {";
  for(unsigned i=0;i<x.count;++i) {
    if(i)out<<", "; auto const& o=x.operations[i];
    out<<"{static_cast<DmEpilogueKind>("<<unsigned(o.kind)<<"u), ";
    ArrayLiteral(out,o.parameter);
    out<<", "<<WriteLiteral(o.residual_map)<<", static_cast<DmActivation>("
       <<unsigned(o.activation)<<"u), static_cast<DmGatePair>("<<unsigned(o.gate)
       <<"u), "<<o.unit<<"u, static_cast<DmRounding>("<<unsigned(o.input_rounding)
       <<"u), static_cast<DmRounding>("<<unsigned(o.output_rounding)<<"u)";
    if(o.norm_width || o.norm_epsilon!=0)
      out<<", "<<o.norm_width<<"u, "<<std::scientific<<std::setprecision(9)<<o.norm_epsilon<<"f";
    out<<'}';
  }
  out<<"}, "<<x.side_count<<"u, {";
  for(unsigned i=0;i<x.side_count;++i) {
    if(i)out<<", "; auto const& s=x.side[i];
    out<<"{static_cast<DmSideOutputKind>("<<unsigned(s.kind)<<"u), "
       <<s.buffer<<"u, "<<s.auxiliary<<"u, "<<s.count<<"u}";
  }
  out<<"}, static_cast<DmRounding>("<<unsigned(x.store_rounding)<<"u)}";
  return out.str();
}
}  // namespace tilemega::frontend
