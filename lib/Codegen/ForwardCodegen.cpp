// SPDX-License-Identifier: BSD-3-Clause
// Native DNN forward lowering (prompt §10 N-1 item 1): a verified DnnModelPlan
// plus the forward solver's coordinates become one typed Sm90ForwardProgram
// translation unit. Only the stage kinds the native bodies cover are lowered;
// anything else stops the build rather than falling back to the legacy body.
#include <tilemega/Codegen/ForwardCodegen.h>

#include "ForwardEmitShared.h"

#include <tilemega/Backend/ConvIteration.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Solver/Sm90ForwardCoordinates.h>
#include <tilemega/Support/Json.h>

#include <mlir/IR/BuiltinAttributes.h>

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::codegen {
namespace {

using namespace tilemega::codegen::forward_emit;
}  // namespace

std::string EmitNativeForwardProgram(mlir::ModuleOp module) {
  int trace_level=0;
  if(auto* setting=std::getenv("TILEMEGA_TRACE_STAGE")) {
    std::string value(setting);
    if(value!="0" && value!="1" && value!="2")
      throw std::invalid_argument("TILEMEGA_TRACE_STAGE must be 0, 1 or 2");
    trace_level=value[0]-'0';
  }
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
  // A buffer's rows as a GEMM or a row-wise body sees them: a row-major
  // matrix's first extent, or an image's pixels (N x H x W).
  auto rows_of=[&](unsigned buffer) {
    auto const& l=layouts.at(buffer);
    if(l.rank==4 && l.kind==codegen::DmLayout::kNHWC)
      return int(std::uint64_t(l.physical[0])*l.physical[1]*l.physical[2]);
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
     // The DNN plans bind 64-bit token identifiers; the serving ABI spells
     // that dtype as the DM buffer table's fourth code.
     <<"#define TM_DTYPE_I64 3\n"
     // Native compile argv is fixed: bind the build environment into the
     // source so artifact identity sees the same trace macro nvcc compiles.
     <<"#define TILEMEGA_TRACE_STAGE "<<trace_level<<"\n"
     <<"#if defined(TILEMEGA_TRACE_STAGE) && TILEMEGA_TRACE_STAGE\n"
     <<"#define TILEMEGA_FORWARD_TRACE_STAGES "<<stages.size()<<"\n"
     <<"#define TILEMEGA_FORWARD_TRACE_GRID "
     <<module->getAttrOfType<mlir::IntegerAttr>("tmexec.sm90_forward_grid").getInt()<<"\n"
     <<"#endif\n"
     <<"#include <tilemega/Codegen/tasks/Sm90ForwardProgram.cuh>\n\n"
     <<"namespace tilemega::codegen::native_forward {\n"
     <<"using namespace tilemega::backend;\n"
     <<"constexpr char kNativeForwardIdentity[]="
     <<Quoted(coordinates.ToJson().Dump(0))<<";\n"
     <<"constexpr int kNativeForwardArena="<<arena<<";\n";

  // Depthwise SimpleGate has natural half-channel inputs. Its producer
  // gathers the two halves; pack filters and pre-gate affine vectors into
  // the same 16-pair order used by the shared epilogue.
  std::map<unsigned,unsigned> depthwise_gate_pack;
  for(auto value:stages) {
    auto stage=Entry(value,"stage");
    if(String(stage,"kind")!="kDepthwiseConv")continue;
    auto chain=frontend::DecodeDmChain(stage.get("dm_chain"));
    unsigned unit=0;
    for(unsigned i=0;i<chain.count;++i)
      if(chain.operations[i].kind==codegen::DmEpilogueKind::kGatePair)unit=chain.operations[i].unit;
    if(!unit)continue;
    auto operands=stage.getAs<mlir::DenseI64ArrayAttr>("operands");
    if(!operands || operands.size()<2)throw std::invalid_argument("depthwise gate lacks its filter operand");
    depthwise_gate_pack[unsigned(operands[1])]=unit;
    for(unsigned i=0;i<chain.count;++i) {
      auto const& op=chain.operations[i];
      if(op.kind==codegen::DmEpilogueKind::kGatePair)break;
      if(op.kind==codegen::DmEpilogueKind::kBias || op.kind==codegen::DmEpilogueKind::kScale)
        depthwise_gate_pack[op.parameter[0]]=unit;
    }
  }
  // Buffers keep their plan order: the harness binds external tensors by
  // index. The table is written once the stages have named their scratch.
  std::ostringstream table;
  unsigned packed_index=0;
  for(auto value:buffers) {
    auto buffer=Entry(value,"buffer");
    auto dtype=buffer.getAs<mlir::StringAttr>("dtype");
    auto role=buffer.getAs<mlir::StringAttr>("role");
    auto pack=buffer.getAs<mlir::StringAttr>("pack_json");
    auto name=dtype?String(buffer,"dtype"):std::string("bf16");
    auto external=buffer.getAs<mlir::StringAttr>("external_name");
    auto bound=external && !external.getValue().empty()?external.getValue().str():
        String(buffer,"name");
    std::string recipe=pack?pack.getValue().str():std::string();
    if(auto gate=depthwise_gate_pack.find(packed_index);gate!=depthwise_gate_pack.end()) {
      if(recipe.empty())throw std::invalid_argument("depthwise gate parameter lacks a packing recipe");
      recipe=json::Value(json::Object{{"kind","gate_pair_interleave"},{"u",int(gate->second)},
          {"source",json::Parse(recipe)}}).Dump(0);
    }
    ++packed_index;
    table<<"  {"<<Quoted(bound)<<','
         <<(role && role.getValue()=="external"?"TM_BUFFER_EXTERNAL":"TM_BUFFER_INTERNAL")<<','
         <<(name=="f32"?"TM_DTYPE_F32":name=="i32"?"TM_DTYPE_I32":
            name=="i64"?"TM_DTYPE_I64":"TM_DTYPE_BF16")<<','
         <<Integer(buffer,"constant",0)<<"ull,"<<Integer(buffer,"per_batch",0)<<"ull,"
         <<Quoted(recipe)<<"},\n";
  }

  MapTable maps;
  // Side outputs the epilogue accumulates into: a launch must start from zero.
  std::map<unsigned,std::uint64_t> zeroed;
  auto accumulated=[&](unsigned id) {
    if(id==kAbsent || id>=buffers.size())return;
    auto buffer=Entry(buffers[id],"buffer");
    zeroed[id]=std::uint64_t(Integer(buffer,"constant",0))+
        std::uint64_t(Integer(buffer,"per_batch",0))*std::uint64_t(batch.getInt());
  };
  // Split-K partials and their per-tile counters are the native program's own
  // state: they extend the plan's buffer table as internal allocations.
  struct Scratch {std::string name,dtype;std::uint64_t elements;};
  std::vector<Scratch> scratch;
  auto add_scratch=[&](std::string name,char const* dtype,std::uint64_t elements) {
    scratch.push_back({std::move(name),dtype,elements});
    return unsigned(buffers.size()+scratch.size()-1);
  };
  std::vector<std::string> placements;
  std::vector<unsigned> aligned_outputs;
  for(unsigned index=0;index<stages.size();++index) {
    auto stage=Entry(stages[index],"stage");
    auto const& c=coordinates.stages[index];
    auto kind=String(stage,"kind");
    auto operands=stage.getAs<mlir::DenseI64ArrayAttr>("operands");
    auto operand=[&](unsigned i) {
      return operands && i<operands.size()?unsigned(operands[i]):kAbsent;
    };
    Binding bind;
    auto require_fp32=[&](unsigned id) {
      if(id==kAbsent || id==codegen::kDmNoIndex || id>=buffers.size())return;
      auto dtype=Entry(buffers[id],"buffer").getAs<mlir::StringAttr>("dtype");
      if(!dtype || dtype.getValue()!="f32")
        throw std::invalid_argument("a native forward per-channel vector must be FP32: "+
            String(Entry(buffers[id],"buffer"),"name"));
    };
    Geometry shape;
    auto pool_chain=[&](std::string chain) {
      if(bind.channel_partial==kAbsent)return chain;
      auto const& partial=layouts.at(bind.channel_partial);
      auto const& output=layouts.at(bind.output);
      if(partial.rank!=3 || output.rank!=4 || output.kind!=codegen::DmLayout::kNHWC ||
          shape.write_kind!=0 || partial.physical[0]!=unsigned(batch.getInt()) ||
          !partial.physical[1] || !partial.physical[2] ||
          partial.physical[2]!=output.logical[3])
        throw std::invalid_argument("native forward channel partials require [B,parts,C] and a dense NHWC producer");
      shape.pool_rows=int(output.logical[1])*int(output.logical[2]);
      shape.pool_parts=int(partial.physical[1]);shape.pool_channels=int(partial.physical[2]);
      if(shape.pool_rows<=0)throw std::invalid_argument("native forward pooling has no pixels");
      bool spatial=c.body==Sm90ForwardBody::kDepthwise || c.body==Sm90ForwardBody::kDwPw ||
          c.body==Sm90ForwardBody::kInvertedResidual;
      return "Sm90ForwardPoolEpilogueChain<"+std::to_string(shape.pool_rows)+','+
          std::to_string(shape.pool_parts)+','+std::to_string(shape.pool_channels)+','+chain+','+
          std::to_string(spatial?c.tile_w:0)+','+std::to_string(spatial?c.tile_h:0)+','+
          std::to_string(spatial?int(output.logical[2]):0)+'>';
    };
    shape.epsilon=0;
    if(auto epsilon=stage.getAs<mlir::FloatAttr>("dm_norm_epsilon"))
      shape.epsilon=epsilon.getValueAsDouble();
    std::string body,wrapper;
    int stored_width=0;

    auto conv_index=unsigned(Integer(stage,"dm_conv",codegen::kDmNoIndex));
    bool has_conv=conv_index!=codegen::kDmNoIndex && conv_index<convs.size();
    if(has_conv) {
      auto const& conv=convs[conv_index];
      auto const& input=layouts.at(conv.input_layout);
      types<<EmitConv(index,conv,input);
      shape.width=int(conv.w);shape.height=int(conv.h);shape.channels=int(conv.c);
      shape.out_channels=int(conv.k);shape.output_width=int(conv.q);
      shape.output_height=int(conv.p);shape.stride=int(conv.stride_h);
      shape.padded_channels=int(input.physical[3]);
    }

    switch(c.body) {
      case Sm90ForwardBody::kGemm:
      case Sm90ForwardBody::kConv: {
        if(kind!="kGemm" || !gemms)
          throw std::invalid_argument("native forward GEMM coordinates need a GEMM stage");
        auto gemm=Entry(gemms[unsigned(Integer(stage,"gemm"))],"gemm");
        auto access=frontend::DecodeDmAccess(gemm.get("dm_access"));
        auto chain=frontend::DecodeDmChain(gemm.get("dm_chain"));
        // A 1x1 unit-stride convolution's A is its input rows, so the body is
        // a dense GEMM over a rank-2 view of that image.
        bool im2col=c.body==Sm90ForwardBody::kConv;
        if(im2col && access.a!=codegen::DmAAccess::kIm2Col)
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
          // The stage computes one output row per A row of its own mapping:
          // the plan may start at another row (BERT's pooler reads row zero).
          if(!access.rows_per_batch)
            throw std::invalid_argument("native DNN dense A mapping requires rows_per_batch");
          rows=int(access.rows_per_batch)*int(batch.getInt());
          MapSpec spec;
          auto l=layouts.at(a);
          if(l.rank==4 && l.kind==codegen::DmLayout::kNHWC) {
            // An image read row by row: its pixels are the GEMM's rows, which
            // a 1x1 convolution and the small bodies both want.
            if(l.halo_top || l.halo_left)
              throw std::invalid_argument("a dense A view needs an image without a halo");
            auto pixels=std::uint64_t(l.physical[0])*l.physical[1]*l.physical[2];
            l.rank=2;l.physical[0]=unsigned(pixels);l.physical[1]=l.physical[3];
            l.strides[0]=l.physical[1];
          }
          if(l.rank!=2)throw std::invalid_argument("a dense forward GEMM reads a rank-2 A");
          // The plan's row mapping (offset, stride) becomes the view's base and
          // its row stride: BERT's pooler reads one row per sequence.
          std::uint64_t stride=access.a_row_stride?access.a_row_stride:1;
          if(std::uint64_t(access.a_row_offset)+(std::uint64_t(rows)-1)*stride>=l.physical[0])
            throw std::invalid_argument("native forward A row mapping leaves its buffer");
          spec.buffer=a;spec.rank=2;
          spec.offset=std::uint64_t(access.a_row_offset)*l.strides[0];
          spec.dims[0]=l.physical[1];
          spec.dims[1]=(l.physical[0]-access.a_row_offset+stride-1)/stride;
          spec.strides[0]=l.strides[0]*2*stride;
          spec.box[0]=64;spec.box[1]=unsigned(c.tile_m);
          bind.map=maps.Add(spec);
        }
        if(access.a_scale!=codegen::kDmNoIndex) {
          if(im2col || access.a_scale>=layouts.size())
            throw std::invalid_argument("native SCA input scaling requires a dense pointwise view");
          auto const& scale=layouts.at(access.a_scale);
          auto dtype=Entry(buffers[access.a_scale],"buffer").getAs<mlir::StringAttr>("dtype");
          if(scale.rank!=2 || scale.logical[0]!=unsigned(batch.getInt()) ||
              !scale.logical[1] || scale.logical[1]!=unsigned(k) ||
              scale.physical[1]<scale.logical[1] || scale.strides[1]!=1 ||
              !dtype || dtype.getValue()!="bf16" || !access.rows_per_batch)
            throw std::invalid_argument("native SCA scale must be BF16 [batch,logical K] with a contiguous channel axis");
          bind.input_scale=access.a_scale;shape.scale_rows=int(access.rows_per_batch);
          shape.scale_channels=int(scale.logical[1]);shape.scale_stride=int(scale.strides[0]);
        }
        // The weight is packed into whole TN x 64 pages, so the body contracts
        // the padded extents and stores only the model's own columns.
        int padded_n=(n+c.tile_n-1)/c.tile_n*c.tile_n;
        int padded_k=im2col?int(backend::ConvIterationGeometry::Build(convs[conv_index],
            layouts.at(convs[conv_index].input_layout),64).iterations)*64:(k+63)/64*64;
        if(padded_k%(c.split_k*64))
          throw std::invalid_argument("native forward split-K does not divide the contraction");
        if(c.split_k>1) {
          auto tiles=std::uint64_t((rows+c.tile_m-1)/c.tile_m)*((padded_n)/c.tile_n);
          bind.partial=add_scratch("native_forward.partial."+std::to_string(index),"f32",
              tiles*unsigned(c.split_k)*unsigned(c.tile_m)*unsigned(c.tile_n));
          // One 64-bit arrival counter per output tile, held as two i32 slots.
          bind.counters=add_scratch("native_forward.counter."+std::to_string(index),"i32",
              2*tiles);
        }
        // A PixelShuffle writeback scatters one input pixel's channel groups
        // over an f x f block of output pixels (NAFNet's upsample).
        auto const& out=layouts.at(bind.output);
        if(access.write.kind==codegen::DmWriteKind::kPixelShuffle) {
          if(!access.write.factor || out.rank!=4)
            throw std::invalid_argument("a shuffled writeback needs its factor and NHWC output");
          shape.write_kind=2;shape.shuffle_factor=int(access.write.factor);
          shape.write_channels=int(out.logical[3]);
          shape.write_width=int(out.logical[2])/shape.shuffle_factor;
          shape.write_height=int(out.logical[1])/shape.shuffle_factor;
          if(shape.write_width*shape.shuffle_factor!=int(out.logical[2]) ||
             shape.write_height*shape.shuffle_factor!=int(out.logical[1]))
            throw std::invalid_argument("a shuffled output extent is not a whole number of blocks");
        }else if(access.write.kind==codegen::DmWriteKind::kNCHW) {
          // The model's own output, which the harness reads as NCHW.
          if(out.rank!=4)throw std::invalid_argument("an NCHW writeback needs a rank-4 output");
          shape.write_kind=1;
          shape.write_channels=int(out.logical[1]);
          shape.write_height=int(out.logical[2]);shape.write_width=int(out.logical[3]);
        }else if(access.write.kind!=codegen::DmWriteKind::kDense)
          throw std::invalid_argument("native forward implements the dense, NCHW and shuffled writebacks");
        shape.output_stride=width_of(bind.output);
        {
          auto buffer=Entry(buffers[bind.output],"buffer");
          auto elements=std::uint64_t(Integer(buffer,"constant",0))+
              std::uint64_t(Integer(buffer,"per_batch",0))*std::uint64_t(batch.getInt());
          auto written=shape.write_kind==2?
              std::uint64_t(rows)*std::uint64_t(shape.shuffle_factor)*shape.shuffle_factor*
                  std::uint64_t(shape.output_stride):
              shape.write_kind==1?std::uint64_t(rows)*std::uint64_t(shape.write_channels):
              std::uint64_t(rows)*std::uint64_t(shape.output_stride);
          if(written>elements)
            throw std::invalid_argument("a native forward stage writes past its output buffer: "+
                String(buffer,"name"));
        }
        bool gated=false;
        auto chain_text=pool_chain(Chain(chain,c.tile_m,c.tile_n,bind,require_fp32,&shape,&gated));
        // A row-statistics side output of a tiled stage reduces its tiles in a
        // fixed order: the shares and one arrival counter per M tile are the
        // program's own state (the chain has just named the side outputs).
        if(bind.row_stats!=kAbsent && padded_n/c.tile_n>1) {
          auto tiles=std::uint64_t(padded_n/c.tile_n);
          bind.row_partials=add_scratch("native_forward.row_partial."+std::to_string(index),
              "f32",std::uint64_t(rows)*tiles*2);
          bind.row_counters=add_scratch("native_forward.row_counter."+std::to_string(index),
              "i32",2*((std::uint64_t(rows)+c.tile_m-1)/c.tile_m));
        }
        if(bind.residual!=kAbsent)shape.residual_stride=width_of(bind.residual);
        int store=gated?n/2:n;stored_width=store;
        bool whole_row=false;
        for(unsigned i=0;i<chain.count;++i) {
          auto operation=chain.operations[i].kind;
          whole_row|=operation==codegen::DmEpilogueKind::kDeferredLayerNorm ||
              operation==codegen::DmEpilogueKind::kResidualLN;
        }
        if(whole_row && c.split_k>1)
          throw std::invalid_argument("a deferred normalization stage cannot split its K");
        if(gated && (n%2 || c.tile_n%2))
          throw std::invalid_argument("a gated native forward GEMM needs an even N and TN");
        body="Sm90ForwardGemm<"+std::to_string(c.tile_m)+','+std::to_string(c.tile_n)+",64,"+std::to_string(c.slots)+','+
            std::to_string(c.split_k)+",Sm90ForwardAMode::"+(im2col?"kIm2col":"kDense")+','+
            std::to_string(rows)+','+std::to_string(padded_n)+','+std::to_string(padded_k)+','+
            std::to_string(store)+','+chain_text+','+
            (im2col?"Conv_"+std::to_string(index):"Sm90ForwardNoConv")+
            (bind.input_scale!=kAbsent?",true,Sm90ForwardInputScale<"+std::to_string(shape.scale_rows)+
                ","+std::to_string(shape.scale_channels)+","+std::to_string(shape.scale_stride)+">":"")+'>';
        wrapper="ForwardGemmStage";
        break;
      }
      case Sm90ForwardBody::kDepthwise: {
        if(kind!="kDepthwiseConv" || !has_conv)
          throw std::invalid_argument("native forward depthwise coordinates need a conv stage");
        auto chain=frontend::DecodeDmChain(stage.get("dm_chain"));
        auto const& conv=convs[conv_index];
        bind.weight=operand(1);bind.output=operand(2);bind.channel_partial=operand(3);
        shape.output_stride=width_of(bind.output);
        {
          auto filter=Entry(buffers[bind.weight],"buffer");
          auto elements=std::uint64_t(Integer(filter,"constant",0));
          auto taps=std::uint64_t(conv.c)*conv.r*conv.s;
          if(conv.r!=3 || conv.s!=3 || !taps || !elements || elements%taps ||
             elements/taps>8)
            throw std::invalid_argument("a native forward depthwise filter is not 3x3 KRSC packed");
          shape.depthwise_pack=int(elements/taps);
        }
        MapSpec spec;
        auto const& l=layouts.at(conv.input_layout);
        spec.buffer=conv.input_layout;spec.rank=4;
        spec.dims[0]=l.physical[3];spec.dims[1]=l.physical[2];
        spec.dims[2]=l.physical[1];spec.dims[3]=l.physical[0];
        spec.strides[0]=l.strides[2]*2;spec.strides[1]=l.strides[1]*2;
        spec.strides[2]=l.strides[0]*2;
        spec.swizzle=0;
        bool input_gated=false;
        for(unsigned i=0;i<chain.count;++i)
          input_gated|=chain.operations[i].kind==codegen::DmEpilogueKind::kGatePair;
        spec.box[0]=input_gated?32:64;spec.box[1]=unsigned((c.tile_w-1)*conv.stride_w+conv.s);
        spec.box[2]=unsigned((c.tile_h-1)*conv.stride_h+conv.r);spec.box[3]=1;
        if(c.channel_block!=64)
          throw std::invalid_argument("a depthwise tiled map reads one 64-channel block");
        bind.map=maps.Add(spec);
        bool gated=false;
        auto chain_text=pool_chain(Chain(chain,c.tile_w*c.tile_h,c.channel_block,bind,require_fp32,&shape,&gated));
        stored_width=gated?int(conv.c)/2:int(conv.c);
        if(bind.residual!=kAbsent)shape.residual_stride=width_of(bind.residual);
        body="Sm90ForwardDepthwise<"+std::to_string(c.tile_w)+','+std::to_string(c.tile_h)+','+
            std::to_string(c.channel_block)+','+std::to_string(conv.stride_h)+','+
            std::to_string(c.slots)+','+chain_text+','+std::to_string(shape.depthwise_pack)+','+std::to_string(batch.getInt())+'>';
        wrapper="ForwardDepthwiseStage";
        break;
      }
      case Sm90ForwardBody::kDwPw: {
        if(kind!="kDwPwFused" || !has_conv || !gemms)
          throw std::invalid_argument("native forward DWPW coordinates need a fused stage");
        auto gemm=Entry(gemms[unsigned(Integer(stage,"gemm"))],"gemm");
        auto chain=frontend::DecodeDmChain(gemm.get("dm_chain"));
        auto const& conv=convs[conv_index];
        // The fused stage's N is the pointwise projection's, not the
        // depthwise ConvDesc's channel count.
        shape.out_channels=int(Integer(gemm,"n"));
        bind.depthwise=operand(1);bind.weight=unsigned(Integer(gemm,"b"));
        // The depthwise filters are packed KRSC: C x 3 x 3 x pack, and the
        // buffer's own extent fixes that inner padding.
        {
          auto filter=Entry(buffers[bind.depthwise],"buffer");
          auto elements=std::uint64_t(Integer(filter,"constant",0));
          auto taps=std::uint64_t(conv.c)*conv.r*conv.s;
          if(!taps || !elements || elements%taps)
            throw std::invalid_argument("a native forward depthwise filter is not KRSC packed");
          shape.depthwise_pack=int(elements/taps);
          if(shape.depthwise_pack>8)
            throw std::invalid_argument("native forward depthwise packing exceeds eight");
        }
        bind.output=unsigned(Integer(gemm,"d"));
        MapSpec spec;
        auto const& l=layouts.at(conv.input_layout);
        spec.buffer=conv.input_layout;spec.rank=4;
        spec.dims[0]=l.physical[3];spec.dims[1]=l.physical[2];
        spec.dims[2]=l.physical[1];spec.dims[3]=l.physical[0];
        spec.strides[0]=l.strides[2]*2;spec.strides[1]=l.strides[1]*2;
        spec.strides[2]=l.strides[0]*2;
        spec.swizzle=0;
        spec.box[0]=64;spec.box[1]=unsigned((c.tile_w-1)*conv.stride_w+conv.s);
        spec.box[2]=unsigned((c.tile_h-1)*conv.stride_h+conv.r);spec.box[3]=1;
        bind.map=maps.Add(spec);
        shape.output_stride=width_of(bind.output);
        bool gated=false;
        auto chain_text=pool_chain(Chain(chain,64,c.tile_n,bind,require_fp32,&shape,&gated));
        stored_width=gated?shape.out_channels/2:shape.out_channels;
        if(bind.residual!=kAbsent)shape.residual_stride=width_of(bind.residual);
        // The stage's own chain is the depthwise's epilogue; the GEMM's is the
        // pointwise's.
        auto depthwise_chain=ElementChain(
            frontend::DecodeDmChain(stage.get("dm_chain")),bind,require_fp32);
        body="Sm90ForwardDwPw<"+std::to_string(c.tile_w)+','+std::to_string(c.tile_h)+','+
            std::to_string(c.channel_block)+','+std::to_string(c.tile_n)+','+
            std::to_string(conv.stride_h)+','+std::to_string(c.slots)+','+chain_text+','+
            std::to_string(shape.depthwise_pack)+','+depthwise_chain+','+std::to_string(batch.getInt())+'>';
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
        int query_rows=c.tile_m,model_batch=int(batch.getInt());
        int head_dim=int(Integer(stage,"group"));
        if(head_dim!=64 || sequence%64 || sequence%query_rows)
          throw std::invalid_argument("native forward attention requires D64 and a 64-row sequence");
        auto const& l=layouts.at(packed);
        if(l.rank!=2 || int(l.physical[1])!=3*heads*head_dim)
          throw std::invalid_argument("native forward attention needs the packed QKV buffer");
        // The packed rows are head-major with q, k and v interleaved inside
        // each head (`qkv_concat_bias`), so each part is a {D, Heads, Seq}
        // view whose head stride spans all three projections.
        for(unsigned part=0;part<3;++part) {
          MapSpec spec;
          spec.buffer=packed;spec.rank=3;
          spec.offset=std::uint64_t(part)*head_dim;
          spec.dims[0]=unsigned(head_dim);spec.dims[1]=unsigned(heads);
          spec.dims[2]=unsigned(model_batch*sequence);
          spec.strides[0]=std::uint64_t(head_dim)*3*2;spec.strides[1]=l.strides[0]*2;
          spec.box[0]=64;spec.box[1]=1;spec.box[2]=part?64:unsigned(query_rows);
          auto id=maps.Add(spec);
          if(part==0)bind.query_map=id;
          else if(part==1)bind.map=id;
          else bind.value_map=id;
        }
        shape.heads=heads;shape.sequence=sequence;
        shape.scale=1.0/std::sqrt(double(head_dim));
        body="Sm90ForwardEncoderAttention<"+std::to_string(model_batch)+','+
            std::to_string(sequence)+",64,"+std::to_string(heads)+','+
            std::to_string(query_rows)+','+std::to_string(c.slots)+'>';
        wrapper="ForwardAttentionStage";
        break;
      }
      case Sm90ForwardBody::kEmbedding: {
        if(kind!="kEmbeddingSum")
          throw std::invalid_argument("native forward embedding coordinates need its stage");
        // ids, token types, the three tables and the summed rows.
        bind.index0=operand(0);bind.index1=operand(1);
        bind.input=operand(2);bind.weight=operand(3);bind.table=operand(4);
        bind.output=operand(5);bind.row_stats=operand(6);
        shape.channels=int(Integer(stage,"width"));
        shape.width=int(Integer(stage,"dm_rows_per_batch",0))*int(batch.getInt());
        shape.height=int(Integer(stage,"dm_rows_per_batch",0));
        if(!shape.channels || !shape.height)
          throw std::invalid_argument("native forward embedding needs its width and rows");
        body="Sm90ForwardEmbeddingSum<"+std::to_string(shape.channels)+'>';
        wrapper="ForwardSmallStage";
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
          if(chain.count) {
            bind.vector[0]=chain.operations[0].parameter[0];require_fp32(bind.vector[0]);
          }
          shape.channels=int(Integer(gemm,"k"));shape.out_channels=int(Integer(gemm,"n"));
          shape.width=rows_of(bind.output);shape.height=1;
          body="Sm90ForwardGemv<"+std::to_string(c.tile_n)+','+std::to_string(shape.channels)+'>';
          break;
        }
        bind.input=operand(0);bind.output=operand(1);
        if(c.body==Sm90ForwardBody::kLayout) {
          if(kind!="kLayoutConvert")
            throw std::invalid_argument("native forward layout coordinates need its stage");
          auto const& l=layouts.at(bind.output);
          // The destination's padded block and the image's own channels.
          shape.channels=int(l.physical[3]);shape.out_channels=int(l.logical[3]);
          shape.width=int(l.logical[2]);shape.height=int(l.logical[1]);
          if(l.halo_top || l.halo_left)
            throw std::invalid_argument("a native forward image carries no halo region");
          body="Sm90ForwardLayout<"+std::to_string(shape.channels)+",128,"+std::to_string(batch.getInt())+'>';
        }else if(c.body==Sm90ForwardBody::kPool) {
          if(kind!="kPool")throw std::invalid_argument("native forward pool coordinates need its stage");
          auto const& l=layouts.at(bind.input);
          shape.channels=int(l.physical[3]);
          shape.width=int(l.logical[2]);shape.height=int(l.logical[1]);
          body="Sm90ForwardMaxPool<3,2,"+std::to_string(batch.getInt())+",128>";
        }else if(c.body==Sm90ForwardBody::kGlobalPool) {
          if(kind!="kGlobalPoolReduce")
            throw std::invalid_argument("native forward pooling coordinates need its stage");
          // The producing convolution wrote the channel partials; this stage
          // reduces them over the parts and divides by the pixel count.
          bind.partial=operand(0);bind.output=operand(2);bind.input=kAbsent;
          auto const& l=layouts.at(bind.partial);
          if(l.rank!=3)throw std::invalid_argument("native forward pooling needs its partial sums");
          shape.channels=int(Integer(stage,"extent"));
          shape.out_channels=width_of(bind.output);
          shape.width=int(l.physical[1]);
          shape.height=int(Integer(stage,"dm_rows_per_batch",0));
          if(shape.height<=0)
            throw std::invalid_argument("native forward pooling needs its pixel count");
          body="Sm90ForwardGlobalPool<"+std::to_string(batch.getInt())+'>';
        }else {
          if(kind!="kLayerNorm")
            throw std::invalid_argument("native forward LayerNorm coordinates need its stage");
          // The stage's operands are {input, gamma, beta, output}, and the
          // affine pair are the exported tables in model precision.
          bind.weight=operand(1);bind.table=operand(2);bind.output=operand(3);
          shape.channels=int(Integer(stage,"width"));
          shape.output_stride=width_of(bind.output);
          if(shape.channels<=0 || shape.channels>width_of(bind.input) ||
              shape.output_stride!=width_of(bind.input))
            throw std::invalid_argument("native forward LayerNorm width/pitch is inconsistent");
          shape.width=rows_of(bind.input);shape.height=1;
          body="Sm90ForwardLayerNorm<"+std::to_string(shape.channels)+','+
              std::to_string(shape.output_stride)+'>';
        }
        break;
      }
      default:
        throw std::invalid_argument("native forward lowering does not implement the body "+
            std::string(solver::kSm90ForwardBodies[unsigned(c.body)]));
    }

    if(shape.write_kind==0 && stored_width>=8) {
      if(shape.output_stride%8)
        throw std::invalid_argument("native forward dense vector stores require an eight-element row pitch");
      auto output=Entry(buffers[bind.output],"buffer");
      auto role=output.getAs<mlir::StringAttr>("role");
      if(role && role.getValue()=="external")aligned_outputs.push_back(bind.output);
    }
    if(kind!="kEmbeddingSum")accumulated(bind.row_stats);
    accumulated(bind.channel_partial);
    types<<bind.Emit(index)<<shape.Emit(index)
         <<"using Body_"<<index<<'='<<body<<";\n"
         <<"using Wrap_"<<index<<'='<<wrapper<<"<Body_"<<index<<",Bind_"<<index
         <<",Shape_"<<index<<">;\n"
         <<"constexpr int kOffset_"<<index<<"=Wrap_"<<index
         <<"::kSharedBytes>1024+kNativeForwardArena?0:"<<(index%2)<<"*kNativeForwardArena;\n"
         <<"using Stage_"<<index<<"=Placement<Wrap_"<<index<<",kOffset_"<<index<<">;\n";
    placements.push_back("Stage_"+std::to_string(index));
  }

  out<<"constexpr std::array<tm_buffer_info,"<<(buffers.size()+scratch.size())
     <<"> kNativeForwardBuffers={{\n"<<table.str();
  for(auto const& item:scratch)
    out<<"  {"<<Quoted(item.name)<<",TM_BUFFER_INTERNAL,"
       <<(item.dtype=="f32"?"TM_DTYPE_F32":"TM_DTYPE_I32")<<','<<item.elements
       <<"ull,0ull,\"\"},\n";
  out<<"}};\n";
  out<<"constexpr std::array<ForwardZeroed,"<<zeroed.size()<<"> kNativeForwardZeroed={{\n";
  for(auto const& [id,elements]:zeroed)out<<"  {"<<id<<"u,"<<elements<<"ull},\n";
  out<<"}};\n";
  out<<types.str()<<"using NativeForwardProgram=Program<"<<batch.getInt()<<','
     <<coordinates.residency<<','<<(buffers.size()+scratch.size())<<','<<maps.specs.size()
     <<",kNativeForwardArena";
  for(auto const& name:placements)out<<','<<name;
  out<<">;\n";
  out<<"constexpr std::array<ForwardMapSpec,"<<maps.specs.size()<<"> kNativeForwardMaps={{\n";
  for(auto const& spec:maps.specs)out<<spec.Emit();
  out<<"}};\n"
     <<"}  // namespace tilemega::codegen::native_forward\n"
     <<"#define tm_plan_create tm_forward_create_aligned_storage\n"
     <<"#include <tilemega/Codegen/tasks/Sm90ForwardRuntime.cuh>\n"
     <<"#undef tm_plan_create\n";
  out<<"extern \"C\" void* tm_plan_create(int batch,void* const* external,int device) {\n"
     <<"  if(!external)return nullptr;\n";
  for(unsigned id:aligned_outputs)
    out<<"  if(reinterpret_cast<std::uintptr_t>(external["<<id<<"])%16) {\n"
       <<"    std::fprintf(stderr,\"native forward output binding "<<id
       <<" requires 16-byte alignment\\n\");return nullptr;\n  }\n";
  out<<"  return tm_forward_create_aligned_storage(batch,external,device);\n}\n";
  // The trace ABI belongs to generated forward artifacts; the shared host
  // runtime stays independent of the attribution-only record layout.
  out<<R"FWDTRACE(
#if TILEMEGA_TRACE_STAGE
extern "C" unsigned tm_forward_trace_schema() {return 1;}
extern "C" unsigned tm_forward_trace_stages() {return TILEMEGA_FORWARD_TRACE_STAGES;}
extern "C" unsigned tm_forward_trace_grid() {return TILEMEGA_FORWARD_TRACE_GRID;}
extern "C" unsigned tm_forward_trace_segments() {
  return tilemega::backend::kSm90ForwardTraceSegments;
}
extern "C" unsigned tm_forward_trace_record_bytes() {
  return sizeof(tilemega::backend::Sm90ForwardTraceRecord);
}
extern "C" unsigned long long tm_forward_trace_bytes() {
  return sizeof(tilemega::backend::sm90_forward_trace_records);
}
extern "C" char const* tm_forward_trace_segment_name(unsigned index) {
  using namespace tilemega::backend;
  return index<kSm90ForwardTraceSegments?kSm90ForwardTraceNames[index]:nullptr;
}
extern "C" int tm_forward_trace_reset() {
  using namespace tilemega::backend;
  auto status=cudaDeviceSynchronize();
  if(status!=cudaSuccess)return int(status);
  void* pointer=nullptr;
  status=cudaGetSymbolAddress(&pointer,sm90_forward_trace_records);
  if(status==cudaSuccess)status=cudaMemset(pointer,0,sizeof(sm90_forward_trace_records));
  if(status==cudaSuccess)status=cudaDeviceSynchronize();
  return int(status);
}
extern "C" int tm_forward_trace_copy(void* output,unsigned long long bytes) {
  using namespace tilemega::backend;
  if(!output || bytes!=sizeof(sm90_forward_trace_records))return -1;
  auto status=cudaDeviceSynchronize();
  if(status==cudaSuccess)
    status=cudaMemcpyFromSymbol(output,sm90_forward_trace_records,sizeof(sm90_forward_trace_records));
  return int(status);
}
#endif
)FWDTRACE";
  return out.str();
}

}  // namespace tilemega::codegen
