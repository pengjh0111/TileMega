// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90ForwardPlan.h>

#include <tilemega/Backend/ConvIteration.h>
#include <tilemega/Codegen/ServingPages.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Frontend/DnnModelPlan.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Solver/ForwardGeometry.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>

#include <algorithm>
#include <stdexcept>

namespace tilemega::solver {
namespace {
using frontend::PlanTaskKind;

std::uint64_t Elements(frontend::PlanBuffer const& buffer,int batch) {
  return std::uint64_t(buffer.constant)+std::uint64_t(buffer.per_batch)*unsigned(batch);
}
}  // namespace

frontend::ModelPlan BuildSm90ForwardPlan(std::string const& export_path,int batch,bool fuse,
    Sm90ForwardAbi abi) {
  if(batch<1)throw std::invalid_argument("a native forward plan needs a positive batch");
  auto bridge=frontend::ReadExportBridge(export_path);
  frontend::DnnPlanOptions options;
  options.batch=unsigned(batch);
  // Every native A panel is a whole 64-channel block, including the image,
  // and the TMA maps supply the convolution padding, so no buffer carries a
  // halo region.
  options.nhwc_channel_block=64;
  options.nhwc_halo=false;
  options.global_pool_channels=128;
  options.dwpw_fuse=fuse;
  options.deferred_layernorm=true;
  // F-1, F-3 and F-4: the stem keeps eight channels, an identity adaptive
  // pool is an alias, and an NHWC flatten reorders the Linear weight.
  options.f3=abi==Sm90ForwardAbi::kF3;
  auto plan=frontend::BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  if(!plan.dm || !plan.forward)
    throw std::invalid_argument("the export is not a DNN forward model");
  return plan;
}

Sm90ForwardBody Sm90ForwardBodyOf(frontend::ModelPlan const& plan,std::size_t index,
    Sm90ForwardAbi abi) {
  auto const& stage=plan.stages.at(index);
  switch(stage.kind) {
    case PlanTaskKind::kGemm: {
      auto const& gemm=plan.gemms.at(stage.gemm);
      if(gemm.access.a==codegen::DmAAccess::kIm2Col) {
        // A 1x1 unit-stride convolution reads its input rows unchanged, so it
        // is a dense GEMM over the pixels; only a real window needs im2col.
        auto const& conv=plan.convolutions.at(gemm.access.conv);
        bool window=conv.r>1 || conv.s>1 || conv.stride_h>1 || conv.stride_w>1 ||
            conv.pad_h || conv.pad_w;
        return window?Sm90ForwardBody::kConv:Sm90ForwardBody::kGemm;
      }
      if(gemm.access.a!=codegen::DmAAccess::kDense)
        throw std::invalid_argument("native forward A access must be dense or im2col");
      // A projection whose A is a single pooled row per image is a GEMV: the
      // WGMMA bodies would run one row of a 64-row tile. F3 keeps it a GEMM
      // and runs it in swap-AB mode instead (DNN3 prompt §4 F-6).
      if(abi==Sm90ForwardAbi::kF3)return Sm90ForwardBody::kGemm;
      auto rows=plan.buffers.at(gemm.a).layout.physical[0];
      return rows<16?Sm90ForwardBody::kGemv:Sm90ForwardBody::kGemm;
    }
    case PlanTaskKind::kDepthwiseConv:return Sm90ForwardBody::kDepthwise;
    case PlanTaskKind::kDwPwFused:return Sm90ForwardBody::kDwPw;
    case PlanTaskKind::kEncoderAttention:return Sm90ForwardBody::kEncoderAttention;
    case PlanTaskKind::kLayoutConvert:return Sm90ForwardBody::kLayout;
    case PlanTaskKind::kPool:return Sm90ForwardBody::kPool;
    case PlanTaskKind::kGlobalPoolReduce:return Sm90ForwardBody::kGlobalPool;
    case PlanTaskKind::kLayerNorm:return Sm90ForwardBody::kLayerNorm;
    case PlanTaskKind::kEmbeddingSum:return Sm90ForwardBody::kEmbedding;
    default:
      throw std::invalid_argument("native forward does not implement this plan stage kind");
  }
}

namespace {
int Dnn1PlanBatch(frontend::ModelPlan const& plan) {
  int result=0;
  for(auto const& b:plan.buffers)if(b.role=="external" && b.per_batch && !b.constant) {
    std::uint64_t elements=1;
    for(unsigned axis=0;axis<b.layout.rank;++axis)elements*=b.layout.physical[axis];
    if(!b.layout.rank || elements%b.per_batch || !elements)
      throw std::invalid_argument("DNN external layout does not bind a whole batch");
    int batch=int(elements/b.per_batch);
    if(result && result!=batch)throw std::invalid_argument("DNN external inputs disagree on batch");
    result=batch;
  }
  if(result<=0)throw std::invalid_argument("DNN plan has no bound external batch");
  return result;
}
std::int64_t Dnn1Pitch(frontend::PlanBuffer const& b) {
  if(!b.layout.rank)throw std::invalid_argument("DNN buffer lacks its row layout");
  return b.layout.physical[b.layout.rank-1];
}
json::Array Dnn1Steps(codegen::DmEpilogueChain const& chain,bool& gated,bool& needs_stats,bool& row_stats) {
  json::Array steps;
  for(unsigned i=0;i<chain.count;++i) {
    auto const& op=chain.operations[i];using Kind=codegen::DmEpilogueKind;
    std::string key;
    switch(op.kind) {
      case Kind::kBias:key="bias";break;
      case Kind::kScale:key="scale";break;
      case Kind::kResidual:key=op.parameter[1]==codegen::kDmNoIndex?"residual":"residual_scaled";break;
      case Kind::kDeferredLayerNorm:key="deferred_ln";needs_stats=true;break;
      case Kind::kResidualLN:key="residual_ln";needs_stats=true;break;
      case Kind::kGatePair:
        key=op.gate==codegen::DmGatePair::kSwiGLU?"swiglu":"simple_gate";gated=true;break;
      case Kind::kActivation:
        switch(op.activation) {
          case codegen::DmActivation::kRelu:key="relu";break;
          case codegen::DmActivation::kRelu6:key="relu6";break;
          case codegen::DmActivation::kGeluErf:key="gelu_erf";break;
          case codegen::DmActivation::kGeluTanh:key="gelu_tanh";break;
          case codegen::DmActivation::kTanh:key="tanh";break;
          case codegen::DmActivation::kSilu:key="silu";break;
        }
        break;
      default:throw std::invalid_argument("DNN1 geometry cannot price this epilogue operation");
    }
    steps.emplace_back(key);
  }
  for(unsigned i=0;i<chain.side_count;++i)
    row_stats|=chain.side[i].kind==codegen::DmSideOutputKind::kRowStats;
  return steps;
}
std::size_t Dnn1PoolProducer(frontend::ModelPlan const& plan,std::size_t pool_index) {
  auto partial=plan.stages.at(pool_index).operands[0];
  std::size_t found=plan.stages.size();
  for(std::size_t i=0;i<pool_index;++i) {
    auto const& stage=plan.stages[i];bool writes=false;
    if(stage.kind==PlanTaskKind::kDepthwiseConv)writes=stage.operands[3]==partial;
    if(frontend::IsGemmStage(stage.kind)) {
      auto const& chain=plan.gemms.at(stage.gemm).chain;
      for(unsigned j=0;j<chain.side_count;++j)
        writes|=chain.side[j].kind==codegen::DmSideOutputKind::kChannelPartialSums &&
            chain.side[j].buffer==partial;
    }
    if(writes) {
      if(found!=plan.stages.size())throw std::invalid_argument("pool partial buffer has multiple producers");
      found=i;
    }
  }
  if(found==plan.stages.size())throw std::invalid_argument("pool partial buffer has no earlier producer");
  return found;
}
void Dnn1EncodeGeometry(frontend::ModelPlan const& plan,std::size_t index,int batch,Sm90ForwardShape& shape) {
  auto const& stage=plan.stages.at(index);json::Value g(json::Object{});
  auto number=[&](char const* key,std::int64_t value) {g.Set(key,double(value));};
  number("batch",batch);number("rows",0);number("channels",shape.channels);
  bool gated=false,needs_stats=false,row_stats=false,channel_pool=false;
  g.Set("has_input_scale",false);
  g.Set("epilogue_steps",json::Array{});g.Set("depthwise_epilogue_steps",json::Array{});
  g.Set("store_kind","scalar_2byte");g.Set("pv_kind","tensor_core");
  auto image=[&](codegen::ConvDesc const& c) {
    number("width",c.w);number("height",c.h);number("output_width",c.q);number("output_height",c.p);
    number("padded_channels",plan.buffers.at(c.input_layout).layout.physical[3]);
  };
  auto chain=[&](codegen::DmEpilogueChain const& operations) {
    g.Set("epilogue_steps",Dnn1Steps(operations,gated,needs_stats,row_stats));
    for(unsigned i=0;i<operations.side_count;++i)
      channel_pool|=operations.side[i].kind==codegen::DmSideOutputKind::kChannelPartialSums;
  };
  auto output_geometry=[&](frontend::PlanBuffer const& out) {
    number("output_pitch",Dnn1Pitch(out));
    if(channel_pool) {
      auto const& layout=out.layout;
      if(layout.rank!=4 || layout.kind!=codegen::DmLayout::kNHWC)
        throw std::invalid_argument("channel pool producer needs an NHWC output");
      number("pool_rows",std::int64_t(layout.logical[1])*layout.logical[2]);
      number("pool_channels",layout.logical[3]);
    }
  };
  switch(shape.body) {
    case Sm90ForwardBody::kGemm:case Sm90ForwardBody::kConv:case Sm90ForwardBody::kGemv: {
      auto const& gemm=plan.gemms.at(stage.gemm);auto const& out=plan.buffers.at(gemm.d);
      if(!gemm.access.rows_per_batch)throw std::invalid_argument("DNN GEMM lacks explicit rows_per_batch");
      shape.m=std::int64_t(batch)*gemm.access.rows_per_batch;shape.pixels=shape.m;
      if(gemm.access.a==codegen::DmAAccess::kIm2Col)image(plan.convolutions.at(gemm.access.conv));
      chain(gemm.chain);number("store_columns",gated?gemm.n/2:gemm.n);number("rows",shape.m);
      output_geometry(out);
      if(gemm.access.a_scale!=codegen::kDmNoIndex) {
        auto const& scale=plan.buffers.at(gemm.access.a_scale).layout;
        if(shape.body!=Sm90ForwardBody::kGemm || scale.rank!=2 ||
            scale.logical[0]!=unsigned(batch) || !scale.logical[1] || scale.strides[1]!=1)
          throw std::invalid_argument("SCA prediction requires dense [batch,channels] scale: index="+std::to_string(index)+" mode="+std::to_string(int(gemm.access.a))+" rank="+std::to_string(scale.rank)+" B="+std::to_string(scale.logical[0])+" C="+std::to_string(scale.logical[1])+" stride="+std::to_string(scale.strides[1]));
        g.Set("has_input_scale",true);number("scale_rows",gemm.access.rows_per_batch);
        number("scale_channels",scale.logical[1]);number("scale_stride",scale.strides[0]);
      }
      g.Set("store_kind",gemm.access.write.kind==codegen::DmWriteKind::kDense?"vector_16byte":"scalar_2byte");
      // Coordinate-dependent padded N remains ceil(N/TN)*TN in codegen/cost.
      shape.weight_bytes=2*shape.n*shape.k;shape.activation_bytes=2*shape.m*shape.k;
      shape.output_bytes=2*shape.m*(gated?gemm.n/2:gemm.n);
      break;
    }
    case Sm90ForwardBody::kDepthwise:case Sm90ForwardBody::kDwPw:case Sm90ForwardBody::kInvertedResidual: {
      auto const& c=plan.convolutions.at(stage.conv);image(c);
      auto const& filters=plan.buffers.at(stage.operands[1]);
      auto taps=std::int64_t(c.c)*c.r*c.s;
      if(!taps || !filters.constant || filters.constant%taps)
        throw std::invalid_argument("DNN DW filter packing is not whole KRSC");
      number("depthwise_pack",filters.constant/taps);
      if(shape.body==Sm90ForwardBody::kDepthwise)chain(stage.chain);
      else {
        bool dw_gated=false,dw_stats=false,dw_rows=false;
        g.Set("depthwise_epilogue_steps",Dnn1Steps(stage.chain,dw_gated,dw_stats,dw_rows));
        if(dw_gated || dw_stats || dw_rows)throw std::invalid_argument("fused DW register chain is not channel-wise");
        chain(plan.gemms.at(stage.gemm).chain);
      }
      if(shape.body==Sm90ForwardBody::kDepthwise) {
        channel_pool|=stage.operands[3]!=codegen::kDmNoIndex;
        output_geometry(plan.buffers.at(stage.operands[2]));
      }else output_geometry(plan.buffers.at(plan.gemms.at(stage.gemm).d));
      number("store_columns",gated?shape.out_channels/2:shape.out_channels);
      g.Set("store_kind","vector_16byte");
      shape.output_bytes=2*shape.pixels*(gated?shape.out_channels/2:shape.out_channels);
      break;
    }
    case Sm90ForwardBody::kEncoderAttention:
      number("seq",stage.width);number("heads",stage.extent);number("head_dim",stage.group);
      shape.m=2LL*batch*stage.extent*stage.width;shape.n=stage.width;shape.k=stage.group;
      shape.pixels=std::int64_t(batch)*stage.extent*(stage.width/16);
      shape.activation_bytes=6LL*batch*stage.width*stage.extent*stage.group;
      shape.output_bytes=2LL*batch*stage.width*stage.extent*stage.group;break;
    case Sm90ForwardBody::kLayout: {
      auto const& out=plan.buffers.at(stage.operands[1]).layout;
      if(out.rank!=4 || out.kind!=codegen::DmLayout::kNHWC)
        throw std::invalid_argument("DNN layout conversion needs NHWC output");
      number("width",out.logical[2]);number("height",out.logical[1]);
      number("input_channels",out.logical[3]);number("padded_channels",out.physical[3]);
      shape.m=std::int64_t(batch)*out.logical[1]*out.logical[2];shape.n=shape.channels=out.logical[3];shape.k=1;
      shape.pixels=shape.m;shape.activation_bytes=shape.output_bytes=2*shape.m*shape.channels;break;
    }
    case Sm90ForwardBody::kPool: {
      auto const& c=plan.convolutions.at(stage.conv);image(c);shape.channels=plan.buffers.at(stage.operands[0]).layout.physical[3];
      shape.m=shape.pixels=std::int64_t(batch)*c.p*c.q;shape.n=shape.channels;shape.k=c.r*c.s;
      shape.kernel=c.r;shape.stride=c.stride_h;shape.activation_bytes=2LL*batch*c.h*c.w*shape.channels;
      shape.output_bytes=2*shape.m*shape.channels;break;
    }
    case Sm90ForwardBody::kGlobalPool: {
      auto const& partial=plan.buffers.at(stage.operands[0]).layout;
      if(partial.rank!=3 || partial.physical[0]!=unsigned(batch))
        throw std::invalid_argument("DNN GAP partial layout must be [B,parts,C]");
      number("pool_producer_stage",Dnn1PoolProducer(plan,index));
      number("parts",partial.physical[1]);number("output_pitch",Dnn1Pitch(plan.buffers.at(stage.operands[2])));
      shape.m=batch;shape.channels=shape.n=stage.extent;shape.k=partial.physical[1];shape.pixels=batch;
      shape.activation_bytes=4LL*batch*partial.physical[1]*shape.channels;
      shape.output_bytes=2LL*batch*shape.channels;break;
    }
    case Sm90ForwardBody::kEmbedding:
      shape.m=shape.pixels=std::int64_t(batch)*stage.rows_per_batch;shape.n=shape.channels=stage.width;shape.k=1;
      number("rows",shape.m);number("seq",stage.rows_per_batch);
      shape.activation_bytes=6*shape.m*shape.channels+16*shape.m;shape.output_bytes=2*shape.m*shape.channels;
      row_stats=stage.operands[6]!=codegen::kDmNoIndex;break;
    case Sm90ForwardBody::kLayerNorm: {
      auto const& input=plan.buffers.at(stage.operands[0]);auto pitch=Dnn1Pitch(input);
      if(!pitch || input.per_batch%pitch)throw std::invalid_argument("DNN LN has an incomplete physical row");
      shape.m=shape.pixels=std::int64_t(batch)*input.per_batch/pitch;shape.n=shape.channels=stage.width;shape.k=1;
      number("rows",shape.m);number("output_pitch",pitch);
      shape.weight_bytes=4*shape.channels;shape.activation_bytes=shape.output_bytes=2*shape.m*shape.channels;break;
    }
  }
  number("channels",shape.channels);g.Set("gated",gated);g.Set("needs_stats",needs_stats);g.Set("has_row_stats",row_stats);g.Set("has_channel_pool",channel_pool);
  g.Set("source","DnnModelPlan descriptors; explicit batch and A rows; logical width distinct from physical pitch");
  shape.dnn1_geometry=std::move(g);
}
}  // namespace

Sm90ForwardMode Sm90ForwardModeOf(frontend::ModelPlan const& plan,std::size_t index,
    Sm90ForwardAbi abi) {
  if(abi!=Sm90ForwardAbi::kF3)return Sm90ForwardMode::kNormal;
  auto const& stage=plan.stages.at(index);
  if(stage.kind!=PlanTaskKind::kGemm)return Sm90ForwardMode::kNormal;
  auto const& gemm=plan.gemms.at(stage.gemm);
  // The stem: the only convolution whose input is physically eight channels,
  // which the producer gathers with cp.async instead of an im2col TMA (§6.6).
  if(gemm.access.a==codegen::DmAAccess::kIm2Col) {
    auto const& conv=plan.convolutions.at(gemm.access.conv);
    return plan.buffers.at(conv.input_layout).layout.physical[3]<=8?
        Sm90ForwardMode::kGather:Sm90ForwardMode::kNormal;
  }
  // At most sixteen rows: the weight becomes the WGMMA A operand (§6.7).
  if(!gemm.access.rows_per_batch)
    throw std::invalid_argument("native DNN dense A mapping requires rows_per_batch");
  auto rows=std::int64_t(gemm.access.rows_per_batch)*Dnn1PlanBatch(plan);
  return rows<=16?Sm90ForwardMode::kSwapAB:Sm90ForwardMode::kNormal;
}

std::vector<Sm90ForwardShape> Sm90ForwardShapesFromPlan(frontend::ModelPlan const& plan,
    Sm90ForwardAbi abi) {
  std::vector<Sm90ForwardShape> shapes;
  int batch=Dnn1PlanBatch(plan);
  for(std::size_t index=0;index<plan.stages.size();++index) {
    auto const& stage=plan.stages[index];
    Sm90ForwardShape shape;
    shape.body=Sm90ForwardBodyOf(plan,index,abi);
    shape.mode=Sm90ForwardModeOf(plan,index,abi);
    auto conv=[&]() -> codegen::ConvDesc const& {
      if(stage.conv==codegen::kDmNoIndex || stage.conv>=plan.convolutions.size())
        throw std::invalid_argument("a native forward conv stage has no descriptor");
      return plan.convolutions[stage.conv];
    };
    switch(shape.body) {
      case Sm90ForwardBody::kGemm:
      case Sm90ForwardBody::kConv:
      case Sm90ForwardBody::kGemv: {
        auto const& gemm=plan.gemms.at(stage.gemm);
        shape.n=gemm.n;
        // The native bodies contract whole 64-element k tiles: a conv's K is
        // its issued iteration count, a dense K its padded extent.
        shape.k=(std::int64_t(gemm.k)+63)/64*64;
        if(shape.body==Sm90ForwardBody::kGemm &&
           gemm.access.a==codegen::DmAAccess::kIm2Col) {
          // A 1x1 convolution read as a dense GEMM: its rows are the pixels.
          auto const& c=plan.convolutions.at(gemm.access.conv);
          shape.m=std::int64_t(c.n)*c.p*c.q;
          shape.pixels=shape.m;shape.channels=c.c;shape.out_channels=c.k;
        }else if(shape.body==Sm90ForwardBody::kConv) {
          auto const& c=plan.convolutions.at(gemm.access.conv);
          shape.k=std::int64_t(backend::ConvIterationGeometry::Build(c,
              plan.buffers.at(c.input_layout).layout,64).iterations)*64;
          shape.m=std::int64_t(c.n)*c.p*c.q;
          shape.pixels=shape.m;shape.channels=c.c;shape.out_channels=c.k;
          shape.kernel=int(c.r);shape.stride=int(c.stride_h);
        }else {
          shape.m=plan.buffers.at(gemm.a).layout.physical[0];
          shape.pixels=shape.m;shape.channels=gemm.k;shape.out_channels=gemm.n;
        }
        shape.weight_bytes=2*shape.n*shape.k;
        shape.activation_bytes=2*shape.m*shape.k;
        shape.output_bytes=2*shape.m*shape.n;
        shape.epilogue_ops=int(gemm.chain.count);
        for(unsigned i=0;i<gemm.chain.count;++i) {
          auto kind=gemm.chain.operations[i].kind;
          if(kind==codegen::DmEpilogueKind::kDeferredLayerNorm ||
             kind==codegen::DmEpilogueKind::kResidualLN ||
             kind==codegen::DmEpilogueKind::kDeferredRMSNorm)shape.whole_row=true;
        }
        break;
      }
      case Sm90ForwardBody::kDepthwise: {
        auto const& c=conv();
        shape.pixels=std::int64_t(c.n)*c.p*c.q;shape.channels=c.c;shape.out_channels=c.c;
        shape.kernel=int(c.r);shape.stride=int(c.stride_h);
        shape.m=shape.pixels;shape.n=c.c;shape.k=c.r*c.s;
        shape.weight_bytes=2*std::int64_t(c.r)*c.s*c.c;
        shape.activation_bytes=2*std::int64_t(c.n)*c.h*c.w*c.c;
        shape.output_bytes=2*shape.pixels*c.c;
        shape.epilogue_ops=int(stage.chain.count);
        break;
      }
      case Sm90ForwardBody::kDwPw:
      case Sm90ForwardBody::kInvertedResidual: {
        auto const& c=conv();
        auto const& gemm=plan.gemms.at(stage.gemm);
        shape.pixels=std::int64_t(c.n)*c.p*c.q;shape.channels=c.c;shape.out_channels=gemm.n;
        shape.kernel=int(c.r);shape.stride=int(c.stride_h);
        shape.m=shape.pixels;shape.n=gemm.n;shape.k=gemm.k;
        shape.fusable=true;
        shape.weight_bytes=2*(std::int64_t(gemm.n)*gemm.k+std::int64_t(c.r)*c.s*c.c);
        shape.activation_bytes=2*std::int64_t(c.n)*c.h*c.w*c.c;
        shape.output_bytes=2*shape.pixels*gemm.n;
        shape.epilogue_ops=int(gemm.chain.count);
        break;
      }
      case Sm90ForwardBody::kEncoderAttention: {
        int sequence=int(stage.width),heads=int(stage.extent),dim=int(stage.group);
        shape.pixels=std::int64_t(heads)*(sequence/16);
        shape.m=sequence;shape.n=sequence;shape.k=dim;
        shape.channels=heads*dim;shape.out_channels=heads*dim;
        // Q K^T and P V over every head.
        shape.weight_bytes=0;
        shape.activation_bytes=2*std::int64_t(sequence)*heads*dim*3;
        shape.output_bytes=2*std::int64_t(sequence)*heads*dim;
        break;
      }
      case Sm90ForwardBody::kEmbedding: {
        shape.pixels=stage.rows_per_batch;shape.channels=stage.width;
        shape.m=shape.pixels;shape.n=shape.channels;shape.k=1;
        shape.activation_bytes=2*shape.pixels*shape.channels*3;
        shape.output_bytes=2*shape.pixels*shape.channels;
        break;
      }
      default: {
        auto input=stage.operands[0],output=stage.operands[1];
        auto const& source=plan.buffers.at(input);
        shape.channels=source.layout.rank?source.layout.physical[source.layout.rank-1]:0;
        shape.pixels=source.layout.rank==4?
            std::int64_t(source.layout.logical[0])*source.layout.logical[1]*source.layout.logical[2]:
            std::int64_t(source.layout.physical[0]);
        shape.m=shape.pixels;shape.n=shape.channels;shape.k=1;
        shape.activation_bytes=2*shape.pixels*shape.channels;
        shape.output_bytes=2*std::int64_t(Elements(plan.buffers.at(output),1));
        break;
      }
    }
    if(shape.body==Sm90ForwardBody::kEncoderAttention) {
      // FLOPs of both matmuls, which the cost model reads through m, n, k.
      shape.m=std::int64_t(stage.extent)*stage.width;shape.n=stage.width;shape.k=stage.group;
    }
    Dnn1EncodeGeometry(plan,index,batch,shape);
    shapes.push_back(shape);
  }
  return shapes;
}

mlir::OwningOpRef<mlir::ModuleOp> MaterializeSm90Forward(frontend::ModelPlan& plan,
    Sm90ForwardCoordinates const& coordinates,int batch,TargetSpec const& target,
    mlir::MLIRContext& context,std::string const& origin) {
  if(target.arch_tag!="sm_90" || target.NvccArch()!="sm_90a" || target.res.num_sms<=0 ||
     target.res.max_dynamic_smem_per_cta<=0 || (origin!="fixed" && origin!="solver"))
    throw std::invalid_argument("native forward materialization requires a complete sm_90a target");
  if(plan.stages.size()!=coordinates.stages.size())
    throw std::invalid_argument("native forward coordinates do not cover every stage");
  for(std::size_t i=0;i<plan.stages.size();++i) {
    coordinates.stages[i].Validate(coordinates.abi);
    if(coordinates.stages[i].body!=Sm90ForwardBodyOf(plan,i,coordinates.abi))
      throw std::invalid_argument("native forward coordinates name another body for a stage");
    if(coordinates.stages[i].mode!=Sm90ForwardModeOf(plan,i,coordinates.abi))
      throw std::invalid_argument("native forward coordinates name another operand mode");
  }
  auto producer_shapes=Sm90ForwardShapesFromPlan(plan,coordinates.abi);
  for(std::size_t i=0;i<plan.stages.size();++i) {
    auto& stage=plan.stages[i];
    if(stage.kind!=PlanTaskKind::kGlobalPoolReduce)continue;
    auto producer=Dnn1PoolProducer(plan,i);
    int parts=Sm90ForwardPoolParts(producer_shapes.at(producer),
        coordinates.stages.at(producer),coordinates.abi);
    auto& partial=plan.buffers.at(stage.operands[0]);auto& layout=partial.layout;
    auto channels=layout.physical[2];
    if(layout.rank!=3 || layout.physical[0]!=unsigned(batch) || !channels ||
        std::uint64_t(parts)*channels>std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("native pool partial allocation exceeds its descriptor");
    layout.logical[1]=layout.physical[1]=unsigned(parts);
    layout.strides[2]=1;layout.strides[1]=channels;layout.strides[0]=std::uint64_t(parts)*channels;
    partial.per_batch=std::uint32_t(layout.strides[0]);
    // The legacy field describes contiguous scalar row bands, whereas this
    // native plan binds tiled ownership through the validated rank-3 layout.
    stage.partial_rows_per_image=0;
  }
  int arena=Sm90ForwardArenaBytes(target,coordinates.residency);

  mlir::OpBuilder b(&context);
  auto module=mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  std::vector<std::uint8_t> written(plan.buffers.size(),1);
  (*module)->setAttr("tilemega.model_plan",frontend::EncodeModelPlan(b,plan,written));
  // The weight packer reads one record per GEMM: the native B layout is the
  // K-major tile pages of that GEMM's own coordinate.
  llvm::SmallVector<mlir::Attribute> records(plan.gemms.size());
  std::vector<bool> bound(plan.gemms.size(),false);
  for(std::size_t i=0;i<plan.stages.size();++i) {
    auto const& stage=plan.stages[i];
    if(!frontend::IsGemmStage(stage.kind))continue;
    auto const& c=coordinates.stages[i];
    int tile_n=c.tile_n;
    records[stage.gemm]=b.getDictionaryAttr({
        b.getNamedAttr("tile_m",b.getI64IntegerAttr(c.tile_m)),
        b.getNamedAttr("tile_n",b.getI64IntegerAttr(tile_n)),
        b.getNamedAttr("tile_k",b.getI64IntegerAttr(64)),
        b.getNamedAttr("stages",b.getI64IntegerAttr(c.slots)),
        b.getNamedAttr("split_k",b.getI64IntegerAttr(c.split_k)),
        b.getNamedAttr("impl",b.getI64IntegerAttr(0))});
    bound[stage.gemm]=true;
  }
  for(std::size_t i=0;i<records.size();++i)
    if(!bound[i])throw std::invalid_argument("a native forward GEMM has no stage");
  (*module)->setAttr("tilemega.gemm_runtime",b.getArrayAttr(records));
  (*module)->setAttr("tmexec.nonpaged_weight_layout_tiled",b.getBoolAttr(true));
  codegen::ResolveServingWeightPacking(*module);
  (*module)->setAttr("tmexec.sm90_forward",b.getStringAttr(coordinates.ToJson().Dump(0)));
  (*module)->setAttr("tmexec.sm90_forward_batch",b.getI64IntegerAttr(batch));
  (*module)->setAttr("tmexec.sm90_forward_arena",b.getI64IntegerAttr(arena));
  (*module)->setAttr("tmexec.sm90_forward_origin",b.getStringAttr(origin));
  (*module)->setAttr("tmexec.sm90_forward_target",b.getStringAttr(target.ToJson()));
  (*module)->setAttr("tmexec.solved_arch",b.getStringAttr("sm_90"));
  (*module)->setAttr("tmexec.solved_residency",b.getI64IntegerAttr(coordinates.residency));
  (*module)->setAttr("tmexec.sm90_forward_grid",
      b.getI64IntegerAttr(target.res.num_sms*coordinates.residency));
  if(mlir::failed(mlir::verify(*module)))
    throw std::invalid_argument("native forward module verification failed");
  return module;
}

}  // namespace tilemega::solver
