// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <limits>
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
  if(!plan.dm) {
    bool extended=!plan.convolutions.empty();
    for(auto const& buffer:plan.buffers)extended|=buffer.layout.rank!=0;
    for(auto const& gemm:plan.gemms) {
      auto const& a=gemm.access;
      extended|=a.a!=DmAAccess::kDense || a.b!=DmBAccess::kDense ||
          a.rows_per_batch || a.a_row_stride || a.a_row_offset ||
          a.conv!=kDmNoIndex || a.rows!=kDmNoIndex || a.binding!=kDmNoIndex ||
          a.a_scale!=kDmNoIndex || a.expert_stride ||
          a.write.kind!=DmWriteKind::kDense || a.write.factor!=1 ||
          a.write.layout!=kDmNoIndex || a.write.rows!=kDmNoIndex ||
          gemm.chain.count || gemm.chain.side_count;
    }
    for(auto const& stage:plan.stages)
      extended|=stage.kind>=PlanTaskKind::kDepthwiseConv ||
          stage.conv!=kDmNoIndex || stage.rows_per_batch;
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
  for(auto const& gemm:plan.gemms) {
    auto const& a=gemm.access;
    if(a.conv!=kDmNoIndex && a.conv>=plan.convolutions.size())
      throw std::invalid_argument("DM GEMM convolution outside geometry table");
    buffer(a.rows,a.a==DmAAccess::kRowGather);
    buffer(a.binding,a.a==DmAAccess::kRowGather || a.b==DmBAccess::kExpertIndirect);
    buffer(a.a_scale); write(a.write);
    if(gemm.chain.count>8 || gemm.chain.side_count>5)
      throw std::invalid_argument("DM epilogue capacity exceeded");
    for(unsigned i=0;i<gemm.chain.count;++i) {
      auto const& op=gemm.chain.operations[i];
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
    for(unsigned i=0;i<gemm.chain.side_count;++i) {
      buffer(gemm.chain.side[i].buffer,true); buffer(gemm.chain.side[i].auxiliary);
    }
  }
  for(auto const& stage:plan.stages)
    if(stage.conv!=kDmNoIndex && stage.conv>=plan.convolutions.size())
      throw std::invalid_argument("DM stage convolution outside geometry table");
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
  auto attr=b.getDenseI64ArrayAttr(w); (void)DecodeDmAccess(attr); return attr;
}
codegen::DmGemmAccess DecodeDmAccess(mlir::Attribute attr) {
  Reader r(attr,14); codegen::DmGemmAccess x;
  x.a=r.Enum<DmAAccess>(2); x.b=r.Enum<DmBAccess>(1);
  x.rows_per_batch=r.U32(); x.a_row_stride=r.U32(); x.a_row_offset=r.U32();
  x.conv=r.U32(); x.rows=r.U32(); x.binding=r.U32(); x.a_scale=r.U32();
  x.expert_stride=r.U64(); x.write=r.Write();
  if((x.a==DmAAccess::kIm2Col && x.conv==kDmNoIndex) ||
     (x.a==DmAAccess::kRowGather && (x.rows==kDmNoIndex || x.binding==kDmNoIndex)) ||
     (x.b==DmBAccess::kExpertIndirect && (x.binding==kDmNoIndex || !x.expert_stride)) ||
     (x.write.kind==DmWriteKind::kRowScatter && x.write.rows==kDmNoIndex))
    throw std::invalid_argument("DM GEMM access lacks its geometry or binding source");
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
    Reader r(ops[i],14); auto& o=x.operations[i]; o.kind=r.Enum<DmEpilogueKind>(7);
    for(auto& p:o.parameter)p=r.U32(); o.residual_map=r.Write();
    o.activation=r.Enum<DmActivation>(5); o.gate=r.Enum<DmGatePair>(1); o.unit=r.U32();
    o.input_rounding=r.Enum<DmRounding>(1); o.output_rounding=r.Enum<DmRounding>(1);
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
     <<x.a_scale<<"u, "<<x.expert_stride<<"ull, "<<WriteLiteral(x.write)<<'}';
  return out.str();
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
       <<"u), static_cast<DmRounding>("<<unsigned(o.output_rounding)<<"u)}";
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
