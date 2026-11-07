// SPDX-License-Identifier: BSD-3-Clause
#pragma once
// Emitters shared by both forward lowerings (DNN3 prompt §5.7 item 2): the
// binding and geometry trait structs, the tensor-map table, the convolution
// trait and the epilogue chain. The legacy lowering and the F3 lowering each
// include this header; neither copies it.
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Solver/Sm90ForwardCoordinates.h>
#include <tilemega/Support/Json.h>

#include <mlir/IR/BuiltinAttributes.h>

#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::codegen::forward_emit {

using solver::Sm90ForwardBody;
constexpr unsigned kAbsent=~0u;

mlir::DictionaryAttr Entry(mlir::Attribute value,char const* what) {
  auto dictionary=llvm::dyn_cast<mlir::DictionaryAttr>(value);
  if(!dictionary)throw std::invalid_argument(std::string("native forward ")+what+" is malformed");
  return dictionary;
}
std::int64_t Integer(mlir::DictionaryAttr value,char const* field,std::int64_t fallback=-1) {
  auto attribute=value.getAs<mlir::IntegerAttr>(field);
  if(!attribute) {
    if(fallback>=0)return fallback;
    throw std::invalid_argument(std::string("native forward stage lacks ")+field);
  }
  return attribute.getInt();
}
std::string String(mlir::DictionaryAttr value,char const* field) {
  auto attribute=value.getAs<mlir::StringAttr>(field);
  if(!attribute)throw std::invalid_argument(std::string("native forward stage lacks ")+field);
  return attribute.getValue().str();
}
std::string Quoted(std::string const& text) {
  // A C string literal, escaped as JSON and with the dump's own line break
  // removed so the initializer stays on one line.
  auto value=json::Value(text).Dump(0);
  while(!value.empty() && (value.back()=='\n' || value.back()=='\r'))value.pop_back();
  return value;
}
std::string Float(double value) {
  std::ostringstream out;
  out<<std::scientific<<std::setprecision(9)<<value<<'f';
  return out.str();
}
std::string Identifier(unsigned value) {
  return value==codegen::kDmNoIndex?"kAbsent":std::to_string(value)+"u";
}

// One stage's buffer and map identifiers. Every wrapper reads the same names,
// so an unused one is simply absent.
struct Binding {
  unsigned map=kAbsent,value_map=kAbsent,query_map=kAbsent;
  unsigned input=kAbsent,weight=kAbsent,depthwise=kAbsent,expand=kAbsent,output=kAbsent;
  unsigned partial=kAbsent,counters=kAbsent,input_scale=kAbsent;
  unsigned vector[4]={kAbsent,kAbsent,kAbsent,kAbsent};
  unsigned residual=kAbsent,stats=kAbsent,row_stats=kAbsent,channel_partial=kAbsent;
  unsigned dw_vector[4]={kAbsent,kAbsent,kAbsent,kAbsent};
  unsigned index0=kAbsent,index1=kAbsent,table=kAbsent;
  unsigned row_partials=kAbsent,row_counters=kAbsent;
  std::string Emit(unsigned index) const {
    std::ostringstream out;
    out<<"struct Bind_"<<index<<" {\n  static constexpr unsigned kMap="<<Identifier(map)
       <<",kValueMap="<<Identifier(value_map)<<",kQueryMap="<<Identifier(query_map)
       <<",\n      kInput="<<Identifier(input)<<",kWeight="<<Identifier(weight)
       <<",kDepthwise="<<Identifier(depthwise)<<",kExpand="<<Identifier(expand)
       <<",\n      kOutput="<<Identifier(output)<<",kPartial="<<Identifier(partial)
       <<",kCounters="<<Identifier(counters)<<",kInputScale="<<Identifier(input_scale)
       <<",\n      kVector0="<<Identifier(vector[0])<<",kVector1="<<Identifier(vector[1])
       <<",kVector2="<<Identifier(vector[2])<<",kVector3="<<Identifier(vector[3])
       <<",\n      kResidual="<<Identifier(residual)<<",kStats="<<Identifier(stats)
       <<",kRowStats="<<Identifier(row_stats)
       <<",kChannelPartial="<<Identifier(channel_partial)
       <<",\n      kIndex0="<<Identifier(index0)<<",kIndex1="<<Identifier(index1)
       <<",kTable="<<Identifier(table)
       <<",\n      kRowPartials="<<Identifier(row_partials)
       <<",kRowCounters="<<Identifier(row_counters)
       <<",\n      kDwVector0="<<Identifier(dw_vector[0])
       <<",kDwVector1="<<Identifier(dw_vector[1])
       <<",kDwVector2="<<Identifier(dw_vector[2])
       <<",kDwVector3="<<Identifier(dw_vector[3])<<";\n};\n";
    return out.str();
  }
};

// One stage's scalar geometry. C++17 has no floating-point template
// parameter, so the epsilon and the attention scale travel as a trait class.
struct Geometry {
  int width=0,height=0,channels=0,out_channels=0,output_width=0,output_height=0;
  int stride=1,output_stride=0,residual_stride=0,heads=0,sequence=0,norm_width=0;
  int depthwise_pack=1,padded_channels=0;
  int scale_rows=0,scale_channels=0,scale_stride=0;
  int pool_rows=0,pool_parts=0,pool_channels=0;
  int write_kind=0,shuffle_factor=1,write_width=0,write_height=0,write_channels=0;
  double epsilon=0,scale=0;
  std::string Emit(unsigned index) const {
    std::ostringstream out;
    out<<"struct Shape_"<<index<<" {\n  static constexpr int kWidth="<<width
       <<",kHeight="<<height<<",kChannels="<<channels<<",kOutChannels="<<out_channels
       <<",\n      kOutputWidth="<<output_width<<",kOutputHeight="<<output_height
       <<",kStrideW="<<stride<<",kStrideH="<<stride
       <<",\n      kOutputStride="<<output_stride<<",kResidualStride="<<residual_stride
       <<",kHeads="<<heads<<",kSequence="<<sequence
       <<",kNormWidth="<<norm_width<<",kDepthwisePack="<<depthwise_pack
       <<",kPaddedChannels="<<padded_channels
       <<",kScaleRows="<<scale_rows<<",kScaleChannels="<<scale_channels<<",kScaleStride="<<scale_stride
       <<",kPoolRows="<<pool_rows<<",kPoolParts="<<pool_parts<<",kPoolChannels="<<pool_channels
       <<";\n"
       <<"  static constexpr float kEpsilon="<<Float(epsilon)<<",kScale="<<Float(scale)
       <<";\n};\n";
    return out.str();
  }
};

std::string EmitConv(unsigned index,codegen::ConvDesc const& conv,
                     codegen::DmBufferLayout const& layout) {
  std::ostringstream out;
  // kC is the input's padded channel block: the im2col panels walk it.
  out<<"struct Conv_"<<index<<" {\n  static constexpr int kQ="<<conv.q<<",kP="<<conv.p
     <<",kC="<<layout.physical[3]<<",kS="<<conv.s<<",kR="<<conv.r
     <<",\n      kStrideW="<<conv.stride_w<<",kStrideH="<<conv.stride_h
     <<",kPadW="<<conv.pad_w<<",kPadH="<<conv.pad_h
     <<",\n      kHaloW="<<layout.halo_left<<",kHaloH="<<layout.halo_top
     <<",kDilationW="<<conv.dilation_w<<",kDilationH="<<conv.dilation_h<<";\n};\n";
  return out.str();
}

// The map table: tiled views carry their own extents, an im2col view carries
// the plan's ConvDesc and input layout for the driver to build from.
struct MapSpec {
  unsigned buffer=0,kind=0,rank=2,swizzle=128;
  std::uint64_t offset=0;
  std::uint64_t dims[4]{},strides[3]{};
  unsigned box[4]{};
  codegen::ConvDesc conv{};
  codegen::DmBufferLayout layout{};
  std::string Emit() const {
    std::ostringstream out;
    out<<"  {"<<buffer<<"u,"<<kind<<"u,"<<rank<<"u,"<<swizzle<<"u,"<<offset<<"ull,{";
    for(unsigned i=0;i<4;++i)out<<(i?",":"")<<dims[i]<<"ull";
    out<<"},{";
    for(unsigned i=0;i<3;++i)out<<(i?",":"")<<strides[i]<<"ull";
    out<<"},{";
    for(unsigned i=0;i<4;++i)out<<(i?",":"")<<box[i]<<'u';
    out<<"},"<<frontend::EmitDm(conv)<<','<<frontend::EmitDm(layout)<<"},\n";
    return out.str();
  }
};

struct MapTable {
  std::vector<MapSpec> specs;
  std::map<std::string,unsigned> seen;
  unsigned Add(MapSpec const& spec) {
    auto text=spec.Emit();
    auto found=seen.find(text);
    if(found!=seen.end())return found->second;
    seen.emplace(text,unsigned(specs.size()));
    specs.push_back(spec);
    return unsigned(specs.size()-1);
  }
};

// The epilogue chain of one stage as the compile-time step list of
// Sm90ForwardEpilogueChain. Only the operations the chain body implements are
// accepted; an unknown one stops the build. Chain vectors fill the binding's
// four vector slots in the order the operations name them.
std::string Chain(codegen::DmEpilogueChain const& chain,int rows,int width,Binding& bind,
                  std::function<void(unsigned)> const& fp32,Geometry* shape=nullptr,
                  bool* gated=nullptr) {
  std::ostringstream out;
  out<<"Sm90ForwardEpilogueChain<"<<rows<<','<<width;
  if(shape)
    out<<','<<shape->write_kind<<','<<shape->shuffle_factor<<','<<shape->write_width
       <<','<<shape->write_height<<','<<shape->write_channels;
  else out<<",0,1,0,0,0";
  unsigned slots=0;
  auto slot=[&](unsigned buffer) {
    for(unsigned i=0;i<slots;++i)if(bind.vector[i]==buffer)return i;
    if(slots==4)throw std::invalid_argument("native forward epilogue needs over four vectors");
    fp32(buffer);
    bind.vector[slots]=buffer;
    return slots++;
  };
  // A deferred or reconstructed LayerNorm reads the row sums its producer
  // wrote, over the plan's own normalization width.
  auto statistics=[&](unsigned buffer,unsigned norm_width,float epsilon) {
    if(bind.stats!=kAbsent && bind.stats!=buffer)
      throw std::invalid_argument("native forward epilogue has two statistics sources");
    bind.stats=buffer;
    if(!norm_width)throw std::invalid_argument("a deferred normalization needs its width");
    if(shape) {
      if(shape->norm_width && shape->norm_width!=int(norm_width))
        throw std::invalid_argument("native forward normalization widths differ in one stage");
      shape->norm_width=int(norm_width);
      // The epsilon of a deferred normalization is the operation's own, which
      // the fold recorded; the stage attribute carries an explicit norm's.
      if(!(epsilon>0))throw std::invalid_argument("a deferred normalization needs its epsilon");
      shape->epsilon=epsilon;
    }
  };
  auto residual=[&](unsigned buffer) {
    if(bind.residual!=kAbsent && bind.residual!=buffer)
      throw std::invalid_argument("native forward epilogue has two residual sources");
    bind.residual=buffer;
  };
  for(unsigned i=0;i<chain.count;++i) {
    auto const& op=chain.operations[i];
    using Kind=codegen::DmEpilogueKind;
    out<<",Sm90ForwardEpilogueStep<Sm90ForwardOp::";
    unsigned first=0,second=0;
    switch(op.kind) {
      case Kind::kBias:out<<"kBias";first=slot(op.parameter[0]);break;
      case Kind::kScale:out<<"kScale";first=slot(op.parameter[0]);break;
      case Kind::kActivation:out<<"kActivation";break;
      case Kind::kGatePair:out<<"kGatePair";if(gated)*gated=true;break;
      case Kind::kResidual:
        if(op.residual_map.kind==codegen::DmWriteKind::kPixelShuffle) {
          if(!shape || shape->write_kind!=2 || int(op.residual_map.factor)!=shape->shuffle_factor)
            throw std::invalid_argument("native forward residual shuffle differs from its output mapping");
        }else if(op.residual_map.kind!=codegen::DmWriteKind::kDense)
          throw std::invalid_argument("native forward residual map must be dense or pixel shuffle");
        out<<"kResidual";residual(op.parameter[0]);first=slot(op.parameter[1]);break;
      case Kind::kDeferredLayerNorm:
        out<<"kDeferredLayerNorm";statistics(op.parameter[0],op.norm_width,op.norm_epsilon);
        first=slot(op.parameter[1]);second=slot(op.parameter[2]);break;
      case Kind::kResidualLN:
        out<<"kResidualLN";residual(op.parameter[0]);
        statistics(op.parameter[1],op.norm_width,op.norm_epsilon);
        first=slot(op.parameter[2]);second=slot(op.parameter[3]);break;
      default:
        throw std::invalid_argument("native forward epilogue does not implement operation "+
            std::to_string(unsigned(op.kind)));
    }
    out<<','<<first<<",codegen::DmActivation::";
    switch(op.activation) {
      case codegen::DmActivation::kRelu:out<<"kRelu";break;
      case codegen::DmActivation::kRelu6:out<<"kRelu6";break;
      case codegen::DmActivation::kGeluErf:out<<"kGeluErf";break;
      case codegen::DmActivation::kGeluTanh:out<<"kGeluTanh";break;
      case codegen::DmActivation::kTanh:out<<"kTanh";break;
      case codegen::DmActivation::kSilu:out<<"kSilu";break;
    }
    out<<",codegen::DmGatePair::"
       <<(op.gate==codegen::DmGatePair::kSwiGLU?"kSwiGLU":"kSimpleGate")
       <<','<<(op.unit?op.unit:16u)<<','<<second<<','<<unsigned(op.residual_map.kind)<<'>';
  }
  for(unsigned i=0;i<chain.side_count;++i) {
    auto const& side=chain.side[i];
    if(side.kind==codegen::DmSideOutputKind::kRowStats)bind.row_stats=side.buffer;
    else if(side.kind==codegen::DmSideOutputKind::kChannelPartialSums)
      bind.channel_partial=side.buffer;
    else throw std::invalid_argument("native forward epilogue does not implement a side output");
  }
  out<<'>';
  return out.str();
}


// The fused depthwise's own epilogue, applied while its result is still in
// registers: only the channel-wise operations such a chain carries are
// implemented, and its vectors fill the binding's depthwise vector slots.
std::string ElementChain(codegen::DmEpilogueChain const& chain,Binding& bind,
                         std::function<void(unsigned)> const& fp32) {
  std::ostringstream out;
  out<<"Sm90ForwardElementChain<";
  unsigned slots=0;
  auto slot=[&](unsigned buffer) {
    for(unsigned i=0;i<slots;++i)if(bind.dw_vector[i]==buffer)return i;
    if(slots==4)
      throw std::invalid_argument("a native forward depthwise chain needs over four vectors");
    fp32(buffer);
    bind.dw_vector[slots]=buffer;
    return slots++;
  };
  for(unsigned i=0;i<chain.count;++i) {
    auto const& op=chain.operations[i];
    using Kind=codegen::DmEpilogueKind;
    if(i)out<<',';
    out<<"Sm90ForwardEpilogueStep<Sm90ForwardOp::";
    unsigned vector=0;
    switch(op.kind) {
      case Kind::kScale:out<<"kScale";vector=slot(op.parameter[0]);break;
      case Kind::kBias:out<<"kBias";vector=slot(op.parameter[0]);break;
      case Kind::kActivation:out<<"kActivation";break;
      default:
        throw std::invalid_argument("a native forward depthwise chain implements scale, "
            "bias and activation only");
    }
    out<<','<<vector<<",codegen::DmActivation::";
    switch(op.activation) {
      case codegen::DmActivation::kRelu:out<<"kRelu";break;
      case codegen::DmActivation::kRelu6:out<<"kRelu6";break;
      case codegen::DmActivation::kGeluErf:out<<"kGeluErf";break;
      case codegen::DmActivation::kGeluTanh:out<<"kGeluTanh";break;
      case codegen::DmActivation::kTanh:out<<"kTanh";break;
      case codegen::DmActivation::kSilu:out<<"kSilu";break;
    }
    out<<'>';
  }
  if(chain.side_count)
    throw std::invalid_argument("a native forward depthwise chain has no side output");
  out<<'>';
  return out.str();
}

}  // namespace tilemega::codegen::forward_emit
