// SPDX-License-Identifier: BSD-3-Clause
// Native DNN forward lowering (prompt §10 N-1 item 1): a verified DnnModelPlan
// plus the forward solver's coordinates become one typed Sm90ForwardProgram
// translation unit. Only the stage kinds the native bodies cover are lowered;
// anything else stops the build rather than falling back to the legacy body.
#include <tilemega/Codegen/ForwardCodegen.h>

#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Solver/Sm90ForwardCoordinates.h>
#include <tilemega/Support/Json.h>

#include <mlir/IR/BuiltinAttributes.h>

#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::codegen {
namespace {

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
  return json::Value(text).Dump(0);
}
std::string Identifier(unsigned value) {
  return value==codegen::kDmNoIndex?"kAbsent":std::to_string(value)+"u";
}

// One stage's buffer and map identifiers. Every wrapper reads the same names,
// so an unused one is simply absent.
struct Binding {
  unsigned map=kAbsent,value_map=kAbsent,query_map=kAbsent;
  unsigned input=kAbsent,weight=kAbsent,depthwise=kAbsent,expand=kAbsent,output=kAbsent;
  unsigned partial=kAbsent,counters=kAbsent;
  unsigned vector[4]={kAbsent,kAbsent,kAbsent,kAbsent};
  unsigned residual=kAbsent,stats=kAbsent,row_stats=kAbsent,channel_partial=kAbsent;
  std::string Emit(unsigned index) const {
    std::ostringstream out;
    out<<"struct Bind_"<<index<<" {\n  static constexpr unsigned kMap="<<Identifier(map)
       <<",kValueMap="<<Identifier(value_map)<<",kQueryMap="<<Identifier(query_map)
       <<",\n      kInput="<<Identifier(input)<<",kWeight="<<Identifier(weight)
       <<",kDepthwise="<<Identifier(depthwise)<<",kExpand="<<Identifier(expand)
       <<",\n      kOutput="<<Identifier(output)<<",kPartial="<<Identifier(partial)
       <<",kCounters="<<Identifier(counters)
       <<",\n      kVector0="<<Identifier(vector[0])<<",kVector1="<<Identifier(vector[1])
       <<",kVector2="<<Identifier(vector[2])<<",kVector3="<<Identifier(vector[3])
       <<",\n      kResidual="<<Identifier(residual)<<",kStats="<<Identifier(stats)
       <<",kRowStats="<<Identifier(row_stats)
       <<",kChannelPartial="<<Identifier(channel_partial)<<";\n};\n";
    return out.str();
  }
};

// One stage's scalar geometry. C++17 has no floating-point template
// parameter, so the epsilon and the attention scale travel as a trait class.
struct Geometry {
  int width=0,height=0,channels=0,out_channels=0,output_width=0,output_height=0;
  int stride=1,output_stride=0,residual_stride=0,heads=0,sequence=0;
  double epsilon=0,scale=0;
  std::string Emit(unsigned index) const {
    std::ostringstream out;
    out<<"struct Shape_"<<index<<" {\n  static constexpr int kWidth="<<width
       <<",kHeight="<<height<<",kChannels="<<channels<<",kOutChannels="<<out_channels
       <<",\n      kOutputWidth="<<output_width<<",kOutputHeight="<<output_height
       <<",kStrideW="<<stride<<",kStrideH="<<stride
       <<",\n      kOutputStride="<<output_stride<<",kResidualStride="<<residual_stride
       <<",kHeads="<<heads<<",kSequence="<<sequence<<";\n"
       <<"  static constexpr float kEpsilon="<<epsilon<<"f,kScale="<<scale<<"f;\n};\n";
    return out.str();
  }
};

std::string EmitConv(unsigned index,codegen::ConvDesc const& conv,
                     codegen::DmBufferLayout const& layout) {
  std::ostringstream out;
  out<<"struct Conv_"<<index<<" {\n  static constexpr int kQ="<<conv.q<<",kP="<<conv.p
     <<",kC="<<conv.c<<",kS="<<conv.s<<",kR="<<conv.r
     <<",\n      kStrideW="<<conv.stride_w<<",kStrideH="<<conv.stride_h
     <<",kPadW="<<conv.pad_w<<",kPadH="<<conv.pad_h
     <<",\n      kHaloW="<<layout.halo_left<<",kHaloH="<<layout.halo_top
     <<",kDilationW="<<conv.dilation_w<<",kDilationH="<<conv.dilation_h<<";\n};\n";
  return out.str();
}

// The map table: tiled views carry their own extents, an im2col view carries
// the plan's ConvDesc and input layout for the driver to build from.
struct MapSpec {
  unsigned buffer=0,kind=0,rank=2;
  std::uint64_t offset=0;
  std::uint64_t dims[4]{},strides[3]{};
  unsigned box[4]{};
  codegen::ConvDesc conv{};
  codegen::DmBufferLayout layout{};
  std::string Emit() const {
    std::ostringstream out;
    out<<"  {"<<buffer<<"u,"<<kind<<"u,"<<rank<<"u,"<<offset<<"ull,{";
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
std::string Chain(codegen::DmEpilogueChain const& chain,int rows,int width,Binding& bind) {
  std::ostringstream out;
  out<<"Sm90ForwardEpilogueChain<"<<rows<<','<<width;
  unsigned slots=0;
  auto slot=[&](unsigned buffer) {
    for(unsigned i=0;i<slots;++i)if(bind.vector[i]==buffer)return i;
    if(slots==4)throw std::invalid_argument("native forward epilogue needs over four vectors");
    bind.vector[slots]=buffer;
    return slots++;
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
      case Kind::kGatePair:out<<"kGatePair";break;
      case Kind::kResidual:
        out<<"kResidual";residual(op.parameter[0]);first=slot(op.parameter[1]);break;
      case Kind::kDeferredLayerNorm:
        out<<"kDeferredLayerNorm";
        first=slot(op.parameter[1]);second=slot(op.parameter[2]);break;
      case Kind::kResidualLN:
        out<<"kResidualLN";residual(op.parameter[0]);
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
       <<','<<(op.unit?op.unit:16u)<<','<<second<<'>';
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

}  // namespace

std::string EmitNativeForwardProgram(mlir::ModuleOp module) {
  auto plan=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  auto text=module->getAttrOfType<mlir::StringAttr>("tmexec.sm90_forward");
  if(!plan || !text)
    throw std::invalid_argument("native forward lowering needs a DNN plan and its coordinates");
  if(!plan.getAs<mlir::BoolAttr>("dm") || !plan.getAs<mlir::BoolAttr>("forward"))
    throw std::invalid_argument("native forward lowering needs a DM forward plan");
  auto coordinates=solver::Sm90ForwardCoordinates::FromJson(json::Parse(text.getValue().str()));
  auto stages=plan.getAs<mlir::ArrayAttr>("stages");
  auto buffers=plan.getAs<mlir::ArrayAttr>("buffers");
  auto gemms=plan.getAs<mlir::ArrayAttr>("gemms");
  auto convolutions=plan.getAs<mlir::ArrayAttr>("dm_convolutions");
  if(!stages || !buffers)throw std::invalid_argument("native forward plan lacks stages or buffers");
  if(stages.size()!=coordinates.stages.size())
    throw std::invalid_argument("native forward coordinates do not cover every stage");
  auto batch=module->getAttrOfType<mlir::IntegerAttr>("tmexec.sm90_forward_batch");
  auto arena_attr=module->getAttrOfType<mlir::IntegerAttr>("tmexec.sm90_forward_arena");
  if(!batch || !arena_attr || batch.getInt()<1 || arena_attr.getInt()<1024 ||
     arena_attr.getInt()%1024)
    throw std::invalid_argument("native forward lowering needs its batch and arena budget");
  int arena=int(arena_attr.getInt());

  std::vector<codegen::DmBufferLayout> layouts;
  for(auto value:buffers)
    layouts.push_back(frontend::DecodeDmLayout(Entry(value,"buffer").get("dm_layout")));
  std::vector<codegen::ConvDesc> convs;
  for(unsigned i=0;convolutions && i<convolutions.size();++i)
    convs.push_back(frontend::DecodeDmConv(convolutions[i]));
  auto rows_of=[&](unsigned buffer) {
    auto const& l=layouts.at(buffer);
    return l.rank?int(l.physical[0]):0;
  };
  auto width_of=[&](unsigned buffer) {
    auto const& l=layouts.at(buffer);
    return l.rank?int(l.physical[l.rank-1]):0;
  };

  std::ostringstream out,types;
  out<<"// SPDX-License-Identifier: BSD-3-Clause\n"
     <<"// Generated from a DnnModelPlan and native forward coordinates.\n"
     <<"#define TILEMEGA_STATIC_FORWARD_NATIVE 1\n#define TILEMEGA_ARCH_ID 900\n"
     <<"#include <tilemega/Codegen/tasks/Sm90ForwardProgram.cuh>\n\n"
     <<"namespace tilemega::codegen::native_forward {\n"
     <<"using namespace tilemega::backend;\n"
     <<"constexpr char kNativeForwardIdentity[]="
     <<Quoted(coordinates.ToJson().Dump(0))<<";\n"
     <<"constexpr int kNativeForwardArena="<<arena<<";\n";

  // Buffers keep their plan order: the harness binds external tensors by index.
  out<<"constexpr std::array<tm_buffer_info,"<<buffers.size()<<"> kNativeForwardBuffers={{\n";
  for(auto value:buffers) {
    auto buffer=Entry(value,"buffer");
    auto dtype=buffer.getAs<mlir::StringAttr>("dtype");
    auto role=buffer.getAs<mlir::StringAttr>("role");
    auto pack=buffer.getAs<mlir::StringAttr>("pack_json");
    auto name=dtype?String(buffer,"dtype"):std::string("bf16");
    out<<"  {"<<Quoted(String(buffer,"name"))<<','
       <<(role && role.getValue()=="external"?"TM_BUFFER_EXTERNAL":"TM_BUFFER_INTERNAL")<<','
       <<(name=="f32"?"TM_DTYPE_F32":name=="i32"?"TM_DTYPE_I32":"TM_DTYPE_BF16")<<','
       <<Integer(buffer,"constant",0)<<"ull,"<<Integer(buffer,"per_batch",0)<<"ull,"
       <<Quoted(pack?pack.getValue().str():std::string())<<"},\n";
  }
  out<<"}};\n";

  MapTable maps;
  std::vector<std::string> placements;
  for(unsigned index=0;index<stages.size();++index) {
    auto stage=Entry(stages[index],"stage");
    auto const& c=coordinates.stages[index];
    auto kind=String(stage,"kind");
    auto operands=stage.getAs<mlir::DenseI64ArrayAttr>("operands");
    auto operand=[&](unsigned i) {
      return operands && i<operands.size()?unsigned(operands[i]):kAbsent;
    };
    Binding bind;
    Geometry shape;
    shape.epsilon=0;
    if(auto epsilon=stage.getAs<mlir::FloatAttr>("dm_norm_epsilon"))
      shape.epsilon=epsilon.getValueAsDouble();
    std::string body,wrapper;

    auto conv_index=unsigned(Integer(stage,"dm_conv",codegen::kDmNoIndex));
    bool has_conv=conv_index!=codegen::kDmNoIndex && conv_index<convs.size();
    if(has_conv) {
      auto const& conv=convs[conv_index];
      auto const& input=layouts.at(conv.input_layout);
      types<<EmitConv(index,conv,input);
      shape.width=int(conv.w);shape.height=int(conv.h);shape.channels=int(conv.c);
      shape.out_channels=int(conv.k);shape.output_width=int(conv.q);
      shape.output_height=int(conv.p);shape.stride=int(conv.stride_h);
    }

    switch(c.body) {
      case Sm90ForwardBody::kGemm:
      case Sm90ForwardBody::kConv: {
        if(kind!="kGemm" || !gemms)
          throw std::invalid_argument("native forward GEMM coordinates need a GEMM stage");
        auto gemm=Entry(gemms[unsigned(Integer(stage,"gemm"))],"gemm");
        auto access=frontend::DecodeDmAccess(gemm.get("dm_access"));
        auto chain=frontend::DecodeDmChain(gemm.get("dm_chain"));
        bool im2col=access.a==codegen::DmAAccess::kIm2Col;
        if(im2col!=(c.body==Sm90ForwardBody::kConv))
          throw std::invalid_argument("native forward body disagrees with the plan's A access");
        unsigned a=unsigned(Integer(gemm,"a")),b=unsigned(Integer(gemm,"b"));
        bind.weight=b;bind.output=unsigned(Integer(gemm,"d"));
        int n=int(Integer(gemm,"n")),k=int(Integer(gemm,"k"));
        int rows=0;
        if(im2col) {
          if(!has_conv && access.conv<convs.size()) {
            auto const& conv=convs[access.conv];
            types<<EmitConv(index,conv,layouts.at(conv.input_layout));
            shape.width=int(conv.w);shape.height=int(conv.h);shape.channels=int(conv.c);
            shape.out_channels=int(conv.k);shape.output_width=int(conv.q);
            shape.output_height=int(conv.p);shape.stride=int(conv.stride_h);
            conv_index=access.conv;has_conv=true;
          }
          if(!has_conv)throw std::invalid_argument("native forward conv stage has no ConvDesc");
          auto const& conv=convs[conv_index];
          rows=int(conv.n)*int(conv.p)*int(conv.q);
          MapSpec spec;
          spec.buffer=conv.input_layout;spec.kind=1;spec.rank=4;
          spec.conv=conv;spec.layout=layouts.at(conv.input_layout);
          bind.map=maps.Add(spec);
        }else {
          rows=rows_of(a);
          MapSpec spec;
          auto const& l=layouts.at(a);
          if(l.rank!=2)throw std::invalid_argument("a dense forward GEMM reads a rank-2 A");
          spec.buffer=a;spec.rank=2;
          spec.dims[0]=l.physical[1];spec.dims[1]=l.physical[0];
          spec.strides[0]=l.strides[0]*2;
          spec.box[0]=64;spec.box[1]=64;
          bind.map=maps.Add(spec);
        }
        if(n%c.tile_n || k%(c.split_k*64))
          throw std::invalid_argument("native forward GEMM coordinates do not divide N or K");
        for(unsigned i=0;i<4 && c.split_k>1;++i)if(operand(i)!=kAbsent) {
          // The plan's split-K scratch and counter travel as stage operands.
          if(bind.partial==kAbsent)bind.partial=operand(i);
          else if(bind.counters==kAbsent)bind.counters=operand(i);
        }
        if(c.split_k>1 && (bind.partial==kAbsent || bind.counters==kAbsent))
          throw std::invalid_argument("a split-K forward GEMM needs its partial and counter");
        shape.output_stride=width_of(bind.output);
        auto chain_text=Chain(chain,64,c.tile_n,bind);
        if(bind.residual!=kAbsent)shape.residual_stride=width_of(bind.residual);
        body="Sm90ForwardGemm<64,"+std::to_string(c.tile_n)+",64,"+std::to_string(c.slots)+','+
            std::to_string(c.split_k)+",Sm90ForwardAMode::"+(im2col?"kIm2col":"kDense")+','+
            std::to_string(rows)+','+std::to_string(n)+','+std::to_string(k)+','+chain_text+','+
            (im2col?"Conv_"+std::to_string(index):"Sm90ForwardNoConv")+'>';
        wrapper="ForwardGemmStage";
        break;
      }
      case Sm90ForwardBody::kDepthwise: {
        if(kind!="kDepthwiseConv" || !has_conv)
          throw std::invalid_argument("native forward depthwise coordinates need a conv stage");
        auto chain=frontend::DecodeDmChain(stage.get("dm_chain"));
        auto const& conv=convs[conv_index];
        bind.weight=operand(1);bind.output=operand(2);
        MapSpec spec;
        auto const& l=layouts.at(conv.input_layout);
        spec.buffer=conv.input_layout;spec.rank=4;
        spec.dims[0]=l.physical[3];spec.dims[1]=l.physical[2];
        spec.dims[2]=l.physical[1];spec.dims[3]=l.physical[0];
        spec.strides[0]=l.strides[2]*2;spec.strides[1]=l.strides[1]*2;
        spec.strides[2]=l.strides[0]*2;
        spec.box[0]=64;spec.box[1]=unsigned((c.tile_w-1)*conv.stride_w+conv.s);
        spec.box[2]=unsigned((c.tile_h-1)*conv.stride_h+conv.r);spec.box[3]=1;
        if(c.channel_block!=64)
          throw std::invalid_argument("a depthwise tiled map reads one 64-channel block");
        bind.map=maps.Add(spec);
        auto chain_text=Chain(chain,c.tile_w*c.tile_h,c.channel_block,bind);
        if(bind.residual!=kAbsent)shape.residual_stride=width_of(bind.residual);
        body="Sm90ForwardDepthwise<"+std::to_string(c.tile_w)+','+std::to_string(c.tile_h)+','+
            std::to_string(c.channel_block)+','+std::to_string(conv.stride_h)+','+
            std::to_string(c.slots)+','+chain_text+'>';
        wrapper="ForwardDepthwiseStage";
        break;
      }
      case Sm90ForwardBody::kDwPw: {
        if(kind!="kDwPwFused" || !has_conv || !gemms)
          throw std::invalid_argument("native forward DWPW coordinates need a fused stage");
        auto gemm=Entry(gemms[unsigned(Integer(stage,"gemm"))],"gemm");
        auto chain=frontend::DecodeDmChain(gemm.get("dm_chain"));
        auto const& conv=convs[conv_index];
        bind.depthwise=operand(1);bind.weight=unsigned(Integer(gemm,"b"));
        bind.output=unsigned(Integer(gemm,"d"));
        MapSpec spec;
        auto const& l=layouts.at(conv.input_layout);
        spec.buffer=conv.input_layout;spec.rank=4;
        spec.dims[0]=l.physical[3];spec.dims[1]=l.physical[2];
        spec.dims[2]=l.physical[1];spec.dims[3]=l.physical[0];
        spec.strides[0]=l.strides[2]*2;spec.strides[1]=l.strides[1]*2;
        spec.strides[2]=l.strides[0]*2;
        spec.box[0]=64;spec.box[1]=unsigned((c.tile_w-1)*conv.stride_w+conv.s);
        spec.box[2]=unsigned((c.tile_h-1)*conv.stride_h+conv.r);spec.box[3]=1;
        bind.map=maps.Add(spec);
        shape.output_stride=width_of(bind.output);
        auto chain_text=Chain(chain,64,c.tile_n,bind);
        if(bind.residual!=kAbsent)shape.residual_stride=width_of(bind.residual);
        body="Sm90ForwardDwPw<"+std::to_string(c.tile_w)+','+std::to_string(c.tile_h)+','+
            std::to_string(c.channel_block)+','+std::to_string(c.tile_n)+','+
            std::to_string(conv.stride_h)+','+std::to_string(c.slots)+','+chain_text+'>';
        wrapper="ForwardFusedStage";
        break;
      }
      case Sm90ForwardBody::kEncoderAttention: {
        if(kind!="kEncoderAttention")
          throw std::invalid_argument("native forward attention coordinates need its stage");
        unsigned packed=operand(0);
        bind.output=operand(1);
        if(operand(2)!=kAbsent)
          throw std::invalid_argument("native forward attention does not implement a mask");
        int sequence=int(Integer(stage,"width")),heads=int(Integer(stage,"extent"));
        int head_dim=int(Integer(stage,"group"));
        if(head_dim!=64 || sequence%64)
          throw std::invalid_argument("native forward attention requires D64 and a 64-row sequence");
        auto const& l=layouts.at(packed);
        if(l.rank!=2 || int(l.physical[1])!=3*heads*head_dim)
          throw std::invalid_argument("native forward attention needs the packed QKV buffer");
        // Q, K and V are {D, Heads, Seq} views of the packed rows.
        for(unsigned part=0;part<3;++part) {
          MapSpec spec;
          spec.buffer=packed;spec.rank=3;
          spec.offset=std::uint64_t(part)*heads*head_dim;
          spec.dims[0]=unsigned(head_dim);spec.dims[1]=unsigned(heads);
          spec.dims[2]=unsigned(sequence);
          spec.strides[0]=std::uint64_t(head_dim)*2;spec.strides[1]=l.strides[0]*2;
          spec.box[0]=64;spec.box[1]=1;spec.box[2]=part?64:16;
          auto id=maps.Add(spec);
          if(part==0)bind.query_map=id;
          else if(part==1)bind.map=id;
          else bind.value_map=id;
        }
        shape.heads=heads;shape.sequence=sequence;
        shape.scale=1.0/std::sqrt(double(head_dim));
        body="Sm90ForwardEncoderAttention<"+std::to_string(sequence)+",64,"+
            std::to_string(heads)+",16,"+std::to_string(c.slots)+'>';
        wrapper="ForwardAttentionStage";
        break;
      }
      case Sm90ForwardBody::kLayout:
      case Sm90ForwardBody::kPool:
      case Sm90ForwardBody::kGlobalPool:
      case Sm90ForwardBody::kGemv:
      case Sm90ForwardBody::kLayerNorm: {
        wrapper="ForwardSmallStage";
        if(c.body==Sm90ForwardBody::kGemv) {
          if(kind!="kGemm" || !gemms)
            throw std::invalid_argument("native forward GEMV coordinates need a GEMM stage");
          auto gemm=Entry(gemms[unsigned(Integer(stage,"gemm"))],"gemm");
          auto chain=frontend::DecodeDmChain(gemm.get("dm_chain"));
          bind.input=unsigned(Integer(gemm,"a"));bind.weight=unsigned(Integer(gemm,"b"));
          bind.output=unsigned(Integer(gemm,"d"));
          if(chain.count>1 || (chain.count==1 &&
              chain.operations[0].kind!=codegen::DmEpilogueKind::kBias))
            throw std::invalid_argument("the native forward GEMV epilogue is a bias alone");
          if(chain.count)bind.vector[0]=chain.operations[0].parameter[0];
          shape.channels=int(Integer(gemm,"k"));shape.out_channels=int(Integer(gemm,"n"));
          shape.width=rows_of(bind.input);shape.height=1;
          body="Sm90ForwardGemv<8>";
          break;
        }
        bind.input=operand(0);bind.output=operand(1);
        if(c.body==Sm90ForwardBody::kLayout) {
          if(kind!="kLayoutConvert")
            throw std::invalid_argument("native forward layout coordinates need its stage");
          auto const& l=layouts.at(bind.output);
          shape.channels=int(l.physical[3]);
          shape.width=int(l.logical[2]);shape.height=int(l.logical[1]);
          body="Sm90ForwardLayout<"+std::to_string(shape.channels)+",64>";
        }else if(c.body==Sm90ForwardBody::kPool) {
          if(kind!="kPool")throw std::invalid_argument("native forward pool coordinates need its stage");
          auto const& l=layouts.at(bind.input);
          shape.channels=int(l.physical[3]);
          shape.width=int(l.logical[2]);shape.height=int(l.logical[1]);
          body="Sm90ForwardMaxPool<3,2>";
        }else if(c.body==Sm90ForwardBody::kGlobalPool) {
          if(kind!="kGlobalPoolReduce")
            throw std::invalid_argument("native forward pooling coordinates need its stage");
          auto const& l=layouts.at(bind.input);
          shape.channels=int(l.physical[3]);
          shape.width=int(l.logical[2]);shape.height=int(l.logical[1]);
          bind.partial=operand(2);bind.counters=operand(3);
          if(bind.partial==kAbsent || bind.counters==kAbsent)
            throw std::invalid_argument("native forward pooling needs its partial and counter");
          body="Sm90ForwardGlobalPool";
        }else {
          if(kind!="kLayerNorm")
            throw std::invalid_argument("native forward LayerNorm coordinates need its stage");
          bind.weight=operand(2);bind.vector[0]=operand(3);
          shape.channels=width_of(bind.input);
          shape.width=rows_of(bind.input);shape.height=1;
          body="Sm90ForwardLayerNorm<"+std::to_string(shape.channels)+'>';
        }
        break;
      }
      default:
        throw std::invalid_argument("native forward lowering does not implement the body "+
            std::string(solver::kSm90ForwardBodies[unsigned(c.body)]));
    }

    types<<bind.Emit(index)<<shape.Emit(index)
         <<"using Body_"<<index<<'='<<body<<";\n"
         <<"using Wrap_"<<index<<'='<<wrapper<<"<Body_"<<index<<",Bind_"<<index
         <<",Shape_"<<index<<">;\n"
         <<"constexpr int kOffset_"<<index<<"=Wrap_"<<index
         <<"::kSharedBytes>1024+kNativeForwardArena?0:"<<(index%2)<<"*kNativeForwardArena;\n"
         <<"using Stage_"<<index<<"=Placement<Wrap_"<<index<<",kOffset_"<<index<<">;\n";
    placements.push_back("Stage_"+std::to_string(index));
  }

  out<<types.str()<<"using NativeForwardProgram=Program<"<<batch.getInt()<<','
     <<coordinates.residency<<','<<buffers.size()<<','<<maps.specs.size()
     <<",kNativeForwardArena";
  for(auto const& name:placements)out<<','<<name;
  out<<">;\n";
  out<<"constexpr std::array<ForwardMapSpec,"<<maps.specs.size()<<"> kNativeForwardMaps={{\n";
  for(auto const& spec:maps.specs)out<<spec.Emit();
  out<<"}};\n"
     <<"}  // namespace tilemega::codegen::native_forward\n"
     <<"#include <tilemega/Codegen/tasks/Sm90ForwardRuntime.cuh>\n";
  return out.str();
}

}  // namespace tilemega::codegen
