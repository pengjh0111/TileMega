// SPDX-License-Identifier: BSD-3-Clause
// The F3 forward lowering (DNN3 prompt §5.7 item 2): a verified DnnModelPlan
// plus schema-3 coordinates whose `abi` is "f3" become one typed
// Sm90F3Program translation unit. The binding, geometry, map-table and
// convolution emitters come from `ForwardEmitShared.h`; only the pieces whose
// output types differ from the legacy skeleton are written here.
//
// F3 has no fallback: a stage kind it does not implement throws (§1.3).
#include <tilemega/Codegen/ForwardCodegen.h>

#include "ForwardEmitShared.h"

#include <tilemega/Backend/ConvIteration.h>
#include <tilemega/Backend/Sm90F3Ring.cuh>
#include <tilemega/Solver/ForwardGeometry.h>

#include <cmath>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::codegen {
namespace {

using namespace tilemega::codegen::forward_emit;
using solver::Sm90ForwardAbi;
using solver::Sm90ForwardBody;
using solver::Sm90ForwardMode;
using solver::Sm90F3ChainClass;

// One stage's binding, in the field names the F3 wrappers read. The legacy
// `Binding` carries operands F3 has no body for (depthwise filters, expand
// weights, per-image input scales), so F3 emits its own trait.
struct F3Binding {
  unsigned map=kAbsent,residual_map=kAbsent,query_map=kAbsent,value_map=kAbsent,output_map=kAbsent;
  unsigned input=kAbsent,weight=kAbsent,output=kAbsent;
  unsigned partial=kAbsent,counters=kAbsent;
  unsigned vector[4]={kAbsent,kAbsent,kAbsent,kAbsent};
  unsigned residual=kAbsent,stats=kAbsent,row_stats=kAbsent;
  unsigned row_partials=kAbsent,row_counters=kAbsent,channel_partial=kAbsent;
  unsigned index0=kAbsent,index1=kAbsent,table=kAbsent;
  std::string Emit(unsigned index) const {
    std::ostringstream out;
    out<<"struct Bind_"<<index<<" {\n  static constexpr unsigned kMap="<<Identifier(map)
       <<",kResidualMap="<<Identifier(residual_map)
       <<",kQueryMap="<<Identifier(query_map)<<",kValueMap="<<Identifier(value_map)
       <<",kOutputMap="<<Identifier(output_map)
       <<",\n      kInput="<<Identifier(input)<<",kWeight="<<Identifier(weight)
       <<",kOutput="<<Identifier(output)
       <<",\n      kPartial="<<Identifier(partial)<<",kCounters="<<Identifier(counters)
       <<",\n      kVector0="<<Identifier(vector[0])<<",kVector1="<<Identifier(vector[1])
       <<",kVector2="<<Identifier(vector[2])<<",kVector3="<<Identifier(vector[3])
       <<",\n      kResidual="<<Identifier(residual)<<",kStats="<<Identifier(stats)
       <<",kRowStats="<<Identifier(row_stats)
       <<",\n      kRowPartials="<<Identifier(row_partials)
       <<",kRowCounters="<<Identifier(row_counters)
       <<",kChannelPartial="<<Identifier(channel_partial)
       <<",\n      kIndex0="<<Identifier(index0)<<",kIndex1="<<Identifier(index1)
       <<",kTable="<<Identifier(table)<<";\n};\n";
    return out.str();
  }
};

// One stage's scalar geometry, again in the F3 wrappers' own field names.
struct F3Geometry {
  int width=0,height=0,channels=0,out_channels=0;
  int output_stride=0,residual_stride=0,norm_width=0,pool_parts=0,rows=0;
  double epsilon=0;
  std::string Emit(unsigned index) const {
    std::ostringstream out;
    out<<"struct Shape_"<<index<<" {\n  static constexpr int kWidth="<<width
       <<",kHeight="<<height<<",kChannels="<<channels
       <<",kOutChannels="<<out_channels
       <<",\n      kOutputStride="<<output_stride<<",kResidualStride="<<residual_stride
       <<",kNormWidth="<<norm_width<<",kPoolParts="<<pool_parts<<",kRows="<<rows
       <<";\n  static constexpr float kEpsilon="<<Float(epsilon)<<";\n};\n";
    return out.str();
  }
};

// The chain as an `Sm90F3Chain` instantiation. The vector slots are allocated
// in the order the operations name them, exactly as the legacy emitter does,
// and the class the register map and the cost model key on comes out with it.
struct F3Chain {
  std::string text;
  Sm90F3ChainClass chain_class=Sm90F3ChainClass::kPlain;
  int vectors=0;
  bool residual=false,stats=false;
};

F3Chain EmitF3Chain(codegen::DmEpilogueChain const& chain,int tile_m,int tile_n,
                    F3Binding& bind,std::function<void(unsigned)> const& fp32,
                    F3Geometry* shape,bool swap_ab,
                    std::vector<codegen::DmBufferLayout> const& layouts) {
  F3Chain result;
  unsigned stat_parts=1;
  unsigned slots=0;
  auto slot=[&](unsigned buffer) {
    for(unsigned i=0;i<slots;++i)if(bind.vector[i]==buffer)return i;
    if(slots==4)throw std::invalid_argument("an F3 epilogue needs over four vectors");
    fp32(buffer);
    bind.vector[slots]=buffer;
    return slots++;
  };
  auto statistics=[&](unsigned buffer,unsigned norm_width,float epsilon) {
    if(bind.stats!=kAbsent && bind.stats!=buffer)
      throw std::invalid_argument("an F3 epilogue has two statistics sources");
    bind.stats=buffer;
    auto const& layout=layouts.at(buffer);
    stat_parts=layout.rank==3?layout.physical[0]:1;
    if(!norm_width)throw std::invalid_argument("a deferred normalization needs its width");
    if(shape) {
      if(shape->norm_width && shape->norm_width!=int(norm_width))
        throw std::invalid_argument("F3 normalization widths differ in one stage");
      shape->norm_width=int(norm_width);
      if(!(epsilon>0))throw std::invalid_argument("a deferred normalization needs its epsilon");
      shape->epsilon=epsilon;
    }
  };
  auto residual=[&](unsigned buffer) {
    if(bind.residual!=kAbsent && bind.residual!=buffer)
      throw std::invalid_argument("an F3 epilogue has two residual sources");
    bind.residual=buffer;
  };
  std::ostringstream steps;
  bool deferred=false,residual_norm=false,activation=false;
  for(unsigned i=0;i<chain.count;++i) {
    auto const& op=chain.operations[i];
    using Kind=codegen::DmEpilogueKind;
    if(i)steps<<',';
    steps<<"Sm90F3Step<Sm90ForwardOp::";
    unsigned first=0,second=0;
    switch(op.kind) {
      case Kind::kBias:steps<<"kBias";first=slot(op.parameter[0]);break;
      case Kind::kScale:steps<<"kScale";first=slot(op.parameter[0]);break;
      case Kind::kActivation:steps<<"kActivation";activation=true;break;
      case Kind::kResidual:
        if(op.residual_map.kind!=codegen::DmWriteKind::kDense)
          throw std::invalid_argument("an F3 residual takes the dense mapping only");
        steps<<"kResidual";residual(op.parameter[0]);
        result.residual=true;
        // A scaled residual rounds its multiply before the add; an unscaled
        // one has no vector, and the body checks the pointer for that.
        first=op.parameter[1]==codegen::kDmNoIndex?kAbsent:slot(op.parameter[1]);
        break;
      case Kind::kDeferredLayerNorm:
        steps<<"kDeferredLayerNorm";deferred=true;result.stats=true;
        statistics(op.parameter[0],op.norm_width,op.norm_epsilon);
        first=slot(op.parameter[1]);second=slot(op.parameter[2]);break;
      case Kind::kResidualLN:
        steps<<"kResidualLN";residual_norm=true;result.stats=true;result.residual=true;
        residual(op.parameter[0]);
        statistics(op.parameter[1],op.norm_width,op.norm_epsilon);
        first=slot(op.parameter[2]);second=slot(op.parameter[3]);break;
      default:
        throw std::invalid_argument("the F3 epilogue does not implement operation "+
            std::to_string(unsigned(op.kind)));
    }
    steps<<','<<first<<",codegen::DmActivation::";
    switch(op.activation) {
      case codegen::DmActivation::kRelu:steps<<"kRelu";break;
      case codegen::DmActivation::kRelu6:steps<<"kRelu6";break;
      case codegen::DmActivation::kGeluErf:steps<<"kGeluErf";break;
      case codegen::DmActivation::kGeluTanh:steps<<"kGeluTanh";break;
      case codegen::DmActivation::kTanh:steps<<"kTanh";break;
      case codegen::DmActivation::kSilu:steps<<"kSilu";break;
    }
    steps<<','<<second<<'>';
  }
  bool row_stats=false;
  for(unsigned i=0;i<chain.side_count;++i) {
    auto const& side=chain.side[i];
    if(side.kind==codegen::DmSideOutputKind::kRowStats) {
      bind.row_stats=side.buffer;row_stats=true;
    }else if(side.kind==codegen::DmSideOutputKind::kChannelPartialSums)
      bind.channel_partial=side.buffer;
    else throw std::invalid_argument("the F3 epilogue does not implement this side output");
  }
  result.vectors=int(slots);
  result.chain_class=solver::Sm90F3ClassOf(result.residual,deferred,residual_norm,
      row_stats,activation,swap_ab);
  std::ostringstream out;
  out<<"Sm90F3Chain<Sm90F3SideLayout<"<<tile_m<<','<<tile_n<<','
     <<(result.residual?"true":"false")<<','<<slots<<','
     <<(result.stats?"true":"false")<<','<<stat_parts<<'>';
  if(chain.count)out<<','<<steps.str();
  out<<'>';
  result.text=row_stats?"Sm90F3WriteRowStats<"+out.str()+'>':out.str();
  return result;
}

}  // namespace

std::string EmitF3ForwardProgram(mlir::ModuleOp module) {
  // DT-STEM is retained as an explicit diagnostic until it wins a same-card
  // model comparison. VGG16 B16 currently regresses4.56% in P-PAIR.
  bool enable_tma_stem=false;
  if(auto* setting=std::getenv("TILEMEGA_F3_TMA_STEM")) {
    std::string value=setting;
    if(value!="0" && value!="1")throw std::invalid_argument("TILEMEGA_F3_TMA_STEM must be0 or1");
    enable_tma_stem=value=="1";
  }
  int trace_level=0;
  if(auto* setting=std::getenv("TILEMEGA_TRACE_STAGE")) {
    std::string value(setting);
    if(value!="0" && value!="1" && value!="2" && value!="3")
      throw std::invalid_argument("TILEMEGA_TRACE_STAGE must be 0, 1, 2 or 3");
    trace_level=value[0]-'0';
  }
  auto plan=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  auto text=module->getAttrOfType<mlir::StringAttr>("tmexec.sm90_forward");
  if(!plan || !text)
    throw std::invalid_argument("the F3 forward lowering needs a DNN plan and its coordinates");
  if(!plan.getAs<mlir::BoolAttr>("dm") || !plan.getAs<mlir::BoolAttr>("forward"))
    throw std::invalid_argument("the F3 forward lowering needs a DM forward plan");
  auto coordinates=solver::Sm90ForwardCoordinates::FromJson(json::Parse(text.getValue().str()));
  if(coordinates.abi!=Sm90ForwardAbi::kF3)
    throw std::invalid_argument("the F3 forward lowering needs coordinates whose abi is f3");
  auto stages=plan.getAs<mlir::ArrayAttr>("stages");
  auto buffers=plan.getAs<mlir::ArrayAttr>("buffers");
  auto gemms=plan.getAs<mlir::ArrayAttr>("gemms");
  auto convolutions=plan.getAs<mlir::ArrayAttr>("dm_convolutions");
  if(!stages || !buffers)
    throw std::invalid_argument("the F3 forward plan lacks stages or buffers");
  if(stages.size()!=coordinates.stages.size())
    throw std::invalid_argument("F3 forward coordinates do not cover every stage");
  auto batch=module->getAttrOfType<mlir::IntegerAttr>("tmexec.sm90_forward_batch");
  if(!batch || batch.getInt()<1)
    throw std::invalid_argument("the F3 forward lowering needs its batch");
  int model_batch=int(batch.getInt());

  std::vector<codegen::DmBufferLayout> layouts;
  for(auto value:buffers)
    layouts.push_back(frontend::DecodeDmLayout(Entry(value,"buffer").get("dm_layout")));
  std::vector<codegen::ConvDesc> convs;
  for(unsigned i=0;convolutions && i<convolutions.size();++i)
    convs.push_back(frontend::DecodeDmConv(convolutions[i]));
  auto width_of=[&](unsigned buffer) {
    auto const& l=layouts.at(buffer);
    return l.rank?int(l.physical[l.rank-1]):0;
  };
  auto rows_of=[&](unsigned buffer) {
    auto const& l=layouts.at(buffer);
    if(l.rank==4 && l.kind==codegen::DmLayout::kNHWC)
      return int(std::uint64_t(l.physical[0])*l.physical[1]*l.physical[2]);
    return l.rank?int(l.physical[0]):0;
  };

  std::ostringstream out,types;
  out<<"// SPDX-License-Identifier: BSD-3-Clause\n"
     <<"// Generated from a DnnModelPlan and F3 forward coordinates.\n"
     <<"#define TILEMEGA_STATIC_FORWARD_F3 1\n#define TILEMEGA_ARCH_ID 900\n"
     <<"#define TM_DTYPE_I64 3\n"
     <<"#define TILEMEGA_TRACE_STAGE "<<trace_level<<"\n"
     <<"#if TILEMEGA_TRACE_STAGE\n"
     <<"#define TILEMEGA_FORWARD_TRACE_STAGES "<<stages.size()<<"\n"
     <<"#define TILEMEGA_FORWARD_TRACE_GRID "
     <<module->getAttrOfType<mlir::IntegerAttr>("tmexec.sm90_forward_grid").getInt()<<"\n"
     <<"#endif\n"
     <<"#include <tilemega/Codegen/tasks/Sm90F3Program.cuh>\n\n"
     <<"namespace tilemega::codegen::forward_f3 {\n"
     <<"using namespace tilemega::backend;\n"
     <<"constexpr char kF3ForwardIdentity[]="
     <<Quoted(coordinates.ToJson().Dump(0))<<";\n";

  // The buffer table keeps the plan's order: the harness binds external
  // tensors by index.
  std::ostringstream table;
  for(auto value:buffers) {
    auto buffer=Entry(value,"buffer");
    auto dtype=buffer.getAs<mlir::StringAttr>("dtype");
    auto role=buffer.getAs<mlir::StringAttr>("role");
    auto pack=buffer.getAs<mlir::StringAttr>("pack_json");
    auto name=dtype?String(buffer,"dtype"):std::string("bf16");
    auto external=buffer.getAs<mlir::StringAttr>("external_name");
    auto bound=external && !external.getValue().empty()?external.getValue().str():
        String(buffer,"name");
    table<<"  {"<<Quoted(bound)<<','
         <<(role && role.getValue()=="external"?"TM_BUFFER_EXTERNAL":"TM_BUFFER_INTERNAL")<<','
         <<(name=="f32"?"TM_DTYPE_F32":name=="i32"?"TM_DTYPE_I32":
            name=="i64"?"TM_DTYPE_I64":"TM_DTYPE_BF16")<<','
         <<Integer(buffer,"constant",0)<<"ull,"<<Integer(buffer,"per_batch",0)<<"ull,"
         <<Quoted(pack?pack.getValue().str():std::string())<<"},\n";
  }

  MapTable maps;
  struct Scratch {std::string name,dtype;std::uint64_t elements;};
  std::vector<Scratch> scratch;
  auto add_scratch=[&](std::string name,char const* dtype,std::uint64_t elements) {
    scratch.push_back({std::move(name),dtype,elements});
    return unsigned(buffers.size()+scratch.size()-1);
  };
  std::vector<std::string> placements;

  for(unsigned index=0;index<stages.size();++index) {
    auto stage=Entry(stages[index],"stage");
    auto const& c=coordinates.stages[index];
    auto kind=String(stage,"kind");
    auto operands=stage.getAs<mlir::DenseI64ArrayAttr>("operands");
    auto operand=[&](unsigned i) {
      return operands && i<operands.size()?unsigned(operands[i]):kAbsent;
    };
    F3Binding bind;
    F3Geometry shape;
    auto require_fp32=[&](unsigned id) {
      if(id==kAbsent || id==codegen::kDmNoIndex || id>=buffers.size())return;
      auto dtype=Entry(buffers[id],"buffer").getAs<mlir::StringAttr>("dtype");
      if(!dtype || dtype.getValue()!="f32")
        throw std::invalid_argument("an F3 per-channel vector must be FP32: "+
            String(Entry(buffers[id],"buffer"),"name"));
    };
    if(auto epsilon=stage.getAs<mlir::FloatAttr>("dm_norm_epsilon"))
      shape.epsilon=epsilon.getValueAsDouble();
    std::string body,wrapper;

    auto conv_index=unsigned(Integer(stage,"dm_conv",codegen::kDmNoIndex));
    bool has_conv=conv_index!=codegen::kDmNoIndex && conv_index<convs.size();

    switch(c.body) {
      case Sm90ForwardBody::kGemm:
      case Sm90ForwardBody::kConv: {
        if(kind!="kGemm" || !gemms)
          throw std::invalid_argument("F3 GEMM coordinates need a GEMM stage");
        auto gemm=Entry(gemms[unsigned(Integer(stage,"gemm"))],"gemm");
        auto access=frontend::DecodeDmAccess(gemm.get("dm_access"));
        auto chain=frontend::DecodeDmChain(gemm.get("dm_chain"));
        bool im2col=c.body==Sm90ForwardBody::kConv;
        bool gather=c.mode==Sm90ForwardMode::kGather;
        bool tma_stem=gather && enable_tma_stem;
        bool swap=c.mode==Sm90ForwardMode::kSwapAB;
        unsigned a=unsigned(Integer(gemm,"a"));
        bind.weight=unsigned(Integer(gemm,"b"));
        bind.output=unsigned(Integer(gemm,"d"));
        int n=int(Integer(gemm,"n")),k=int(Integer(gemm,"k"));
        int rows=0,padded_k=0;
        if(im2col || gather) {
          if(!has_conv && access.conv<convs.size()) {conv_index=access.conv;has_conv=true;}
          if(!has_conv)throw std::invalid_argument("an F3 conv stage has no ConvDesc");
          auto const& conv=convs[conv_index];
          auto const& input=layouts.at(conv.input_layout);
          rows=int(conv.n)*int(conv.p)*int(conv.q);
          shape.channels=int(conv.c);shape.out_channels=int(conv.k);
          types<<EmitConv(index,conv,input);
          padded_k=int(backend::ConvIterationGeometry::Build(conv,input,64).iterations)*64;
          if(gather) {
            if(input.physical[3]>8)
              throw std::invalid_argument("F3 gather mode needs an eight-channel input");
            bind.input=conv.input_layout;
          }
          if(!gather || tma_stem) {
            MapSpec spec;
            spec.buffer=conv.input_layout;spec.kind=1;spec.rank=4;
            spec.conv=conv;spec.layout=input;
            if(gather) {spec.box[0]=8;spec.swizzle=0;}
            bind.map=maps.Add(spec);
          }
        }else {
          if(!access.rows_per_batch)
            throw std::invalid_argument("an F3 dense A mapping requires rows_per_batch");
          rows=int(access.rows_per_batch)*model_batch;
          padded_k=(k+63)/64*64;
          auto l=layouts.at(a);
          if(l.rank==4 && l.kind==codegen::DmLayout::kNHWC) {
            if(l.halo_top || l.halo_left)
              throw std::invalid_argument("a dense A view needs an image without a halo");
            auto image_elements=std::uint64_t(l.physical[1])*l.physical[2]*l.physical[3];
            // VGG FC1 consumes a whole flattened NHWC image; its weight recipe
            // already permutes NCHW columns, so the TMA row must span all K.
            auto row_width=image_elements==std::uint64_t(k)?unsigned(k):l.physical[3];
            auto dense_rows=std::uint64_t(l.physical[0])*image_elements/row_width;
            l.rank=2;l.physical[0]=unsigned(dense_rows);l.physical[1]=row_width;
            l.strides[0]=l.physical[1];
          }
          if(l.rank!=2)throw std::invalid_argument("a dense F3 GEMM reads a rank-2 A");
          std::uint64_t stride=access.a_row_stride?access.a_row_stride:1;
          if(std::uint64_t(access.a_row_offset)+(std::uint64_t(rows)-1)*stride>=l.physical[0])
            throw std::invalid_argument("an F3 A row mapping leaves its buffer");
          MapSpec spec;
          spec.buffer=a;spec.rank=2;
          spec.offset=std::uint64_t(access.a_row_offset)*l.strides[0];
          spec.dims[0]=l.physical[1];
          spec.dims[1]=(l.physical[0]-access.a_row_offset+stride-1)/stride;
          spec.strides[0]=l.strides[0]*2*stride;
          if(std::uint64_t(k)>spec.dims[0])
            throw std::invalid_argument("an F3 dense A row is narrower than its contraction");
          // swap-AB loads the activation as the WGMMA's B operand, so the box
          // is the stage's own rows; normal mode takes the M tile.
          spec.box[0]=64;spec.box[1]=unsigned(swap?(rows<=8?8:16):c.tile_m);
          bind.map=maps.Add(spec);
          shape.channels=k;shape.out_channels=n;
        }
        if(access.a_scale!=codegen::kDmNoIndex)
          throw std::invalid_argument("F3 does not implement a per-image input scale");
        if(access.write.kind!=codegen::DmWriteKind::kDense)
          throw std::invalid_argument("F3 implements the dense writeback only");
        int tile_n=swap?128:c.tile_n;
        if((tile_n==96 || tile_n==192) && n%tile_n)
          throw std::invalid_argument("F3 TN96/192 must divide the output width");
        int padded_n=(n+tile_n-1)/tile_n*tile_n;
        if(padded_k%(c.split_k*c.tk))
          throw std::invalid_argument("F3 split-K does not divide the contraction");
        shape.output_stride=width_of(bind.output);
        auto f3chain=EmitF3Chain(chain,swap?(rows<=8?8:16):c.tile_m,tile_n,bind,
                                 require_fp32,&shape,swap,layouts);
        if(bind.residual!=kAbsent) {
          shape.residual_stride=width_of(bind.residual);
          // The producer prefetches the residual tile through its own SW128
          // 2D map, one 64-column box per request (§6.4 step 1).
          auto const& l=layouts.at(bind.residual);
          MapSpec spec;
          spec.buffer=bind.residual;spec.rank=2;
          spec.dims[0]=unsigned(shape.residual_stride);
          spec.dims[1]=unsigned(rows_of(bind.residual));
          spec.strides[0]=std::uint64_t(shape.residual_stride)*2;
          spec.box[0]=64;spec.box[1]=unsigned(c.tile_m);
          bind.residual_map=maps.Add(spec);
        }
        if(!swap && !solver::Sm90F3TileFeasible(c.tile_m,c.tile_n,f3chain.chain_class))
          throw std::invalid_argument("DT-SPILL excluded this F3 tile and chain at stage "+
              std::to_string(index)+": "+
              std::to_string(c.tile_m)+"x"+std::to_string(c.tile_n)+" "+
              solver::kSm90F3ChainClasses[unsigned(f3chain.chain_class)]);
        if(c.split_k>1) {
          auto tiles=swap?std::uint64_t(padded_n/128):
              std::uint64_t((rows+c.tile_m-1)/c.tile_m)*(padded_n/c.tile_n);
          auto per_tile=swap?std::uint64_t((rows<=8?8:16))*128:
              std::uint64_t(c.tile_m)*c.tile_n;
          bind.partial=add_scratch("f3.partial."+std::to_string(index),"f32",
              tiles*unsigned(c.split_k)*per_tile);
          bind.counters=add_scratch("f3.counter."+std::to_string(index),"i32",2*tiles);
        }
        if(bind.row_stats!=kAbsent) {
          if(tile_n%64 || (c.tile_m==64 && tile_n==192))
            throw std::invalid_argument("F3 statistics require whole WG64-column blocks");
          bind.row_partials=bind.row_stats;bind.row_stats=kAbsent;
        }
        // A GAP producer also writes the per-(image, 64-row block) channel
        // partials the later global-average stage reduces (F-7, §6.4 step 4).
        std::string pool="Sm90F3NoPool";
        if(bind.channel_partial!=kAbsent) {
          auto const& partial=layouts.at(bind.channel_partial);
          auto const& o=layouts.at(bind.output);
          if(partial.rank!=3 || o.rank!=4 || o.kind!=codegen::DmLayout::kNHWC ||
              partial.physical[0]!=unsigned(model_batch) || !partial.physical[1] ||
              partial.physical[2]!=o.logical[3])
            throw std::invalid_argument(
                "F3 channel partials need [B,parts,C] and a dense NHWC producer");
          int area=int(o.logical[1])*int(o.logical[2]);
          if(area<=0)throw std::invalid_argument("an F3 pooled producer has no pixels");
          pool="Sm90F3Pool<"+std::to_string(area)+','+
              std::to_string(partial.physical[1])+','+
              std::to_string(partial.physical[2])+'>';
          if(swap)throw std::invalid_argument("a swap-AB stage does not produce GAP partials");
        }
        shape.rows=rows;
        bool tma_store=c.store==solver::Sm90ForwardStore::kTma;
        if(tma_store) {
          if(swap || c.tile_m<128 || tile_n%64)
            throw std::invalid_argument("F3 TMA store needs a full64-column WG tile");
          MapSpec spec;spec.buffer=bind.output;spec.rank=2;
          spec.dims[0]=n;spec.dims[1]=rows;spec.strides[0]=shape.output_stride*2;
          spec.box[0]=64;spec.box[1]=64;
          bind.output_map=maps.Add(spec);
        }
        if(swap) {
          body="Sm90F3GemmSwapAB<"+std::to_string(c.slots)+','+std::to_string(c.split_k)+','+
              std::to_string(rows)+','+std::to_string(padded_n)+','+
              std::to_string(padded_k)+','+std::to_string(n)+','+f3chain.text+','+
              std::to_string(c.tk)+'>';
        }else {
          char const* mode=gather?(tma_stem?"kTmaStem":"kGather"):"kNormal";
          body="Sm90F3Gemm<"+std::to_string(c.tile_m)+','+std::to_string(c.tile_n)+','+
              std::to_string(c.slots)+','+std::to_string(c.split_k)+
              ",Sm90F3Mode::"+mode+','+std::to_string(rows)+','+std::to_string(padded_n)+','+
              std::to_string(padded_k)+','+std::to_string(n)+','+f3chain.text+','+
              ((im2col||gather)?"Conv_"+std::to_string(index):"Sm90F3NoConv")+','+pool+','+
              std::to_string(c.tk)+','+(tma_store?"true":"false")+'>';
        }
        wrapper="GemmStage";
        break;
      }
      case Sm90ForwardBody::kLayout: {
        if(kind!="kLayoutConvert")
          throw std::invalid_argument("F3 layout coordinates need their stage");
        bind.input=operand(0);bind.output=operand(1);
        auto const& l=layouts.at(bind.output);
        if(l.physical[3]!=8)
          throw std::invalid_argument("an F3 layout stage fills one eight-channel block");
        shape.channels=int(l.logical[3]);shape.out_channels=int(l.physical[3]);
        shape.width=int(l.logical[2]);shape.height=int(l.logical[1]);
        shape.output_stride=int(l.physical[3]);
        body="Sm90F3Layout<"+std::to_string(shape.channels)+','+
            std::to_string(model_batch)+'>';
        wrapper="SmallStage";
        break;
      }
      case Sm90ForwardBody::kPool: {
        if(kind!="kPool" || !has_conv)
          throw std::invalid_argument("F3 pool coordinates need their stage");
        bind.input=operand(0);bind.output=operand(1);
        auto const& conv=convs[conv_index];
        if(conv.dilation_h!=1 || conv.dilation_w!=1)
          throw std::invalid_argument("an F3 max pool has no dilation");
        if(conv.pad_h!=conv.pad_w)
          throw std::invalid_argument("an F3 max pool pads both axes equally");
        auto const& l=layouts.at(bind.input);
        shape.channels=int(l.physical[3]);
        shape.width=int(conv.w);shape.height=int(conv.h);
        shape.output_stride=width_of(bind.output);
        // The generalized body takes the plan's own window, stride and pad, so
        // VGG16's 2x2/s2/p0 and ResNet's 3x3/s2/p1 share it (§4 F-2).
        body="Sm90F3MaxPool<"+std::to_string(conv.r)+','+std::to_string(conv.s)+','+
            std::to_string(conv.stride_h)+','+std::to_string(conv.pad_h)+','+
            std::to_string(shape.channels)+','+std::to_string(model_batch)+','+
            std::to_string(shape.width)+','+std::to_string(shape.height)+'>';
        wrapper="SmallStage";
        break;
      }
      case Sm90ForwardBody::kGlobalPool: {
        if(kind!="kGlobalPoolReduce")
          throw std::invalid_argument("F3 pooling coordinates need their stage");
        bind.partial=operand(0);bind.output=operand(2);
        auto const& l=layouts.at(bind.partial);
        if(l.rank!=3)throw std::invalid_argument("F3 pooling needs its partial sums");
        shape.channels=int(Integer(stage,"extent"));
        shape.out_channels=width_of(bind.output);
        shape.pool_parts=int(l.physical[1]);
        shape.rows=int(Integer(stage,"dm_rows_per_batch",0));
        if(shape.rows<=0)throw std::invalid_argument("F3 pooling needs its pixel count");
        body="Sm90F3GlobalPool<"+std::to_string(model_batch)+'>';
        wrapper="SmallStage";
        break;
      }
      case Sm90ForwardBody::kLayerNorm: {
        if(kind!="kLayerNorm")
          throw std::invalid_argument("F3 LayerNorm coordinates need their stage");
        bind.input=operand(0);bind.weight=operand(1);bind.table=operand(2);
        bind.output=operand(3);
        shape.channels=int(Integer(stage,"width"));
        shape.output_stride=width_of(bind.output);
        if(shape.channels<=0 || shape.channels>width_of(bind.input) ||
            shape.output_stride!=width_of(bind.input))
          throw std::invalid_argument("F3 LayerNorm width or pitch is inconsistent");
        shape.rows=rows_of(bind.input);
        body="Sm90F3LayerNorm<"+std::to_string(shape.channels)+','+
            std::to_string(shape.output_stride)+'>';
        wrapper="SmallStage";
        break;
      }
      case Sm90ForwardBody::kEmbedding: {
        if(kind!="kEmbeddingSum")
          throw std::invalid_argument("F3 embedding coordinates need their stage");
        bind.index0=operand(0);bind.index1=operand(1);
        bind.input=operand(2);bind.weight=operand(3);bind.table=operand(4);
        bind.output=operand(5);bind.row_stats=operand(6);
        shape.channels=int(Integer(stage,"width"));
        shape.height=int(Integer(stage,"dm_rows_per_batch",0));
        shape.rows=shape.height*model_batch;
        if(!shape.channels || !shape.height)
          throw std::invalid_argument("F3 embedding needs its width and rows");
        shape.output_stride=shape.channels;
        body="Sm90F3EmbeddingSum<"+std::to_string(shape.channels)+'>';
        wrapper="SmallStage";
        break;
      }
      case Sm90ForwardBody::kEncoderAttention: {
        if(kind!="kEncoderAttention" || operand(2)!=kAbsent)
          throw std::invalid_argument("F3 attention requires an unmasked encoder stage");
        unsigned packed=operand(0);
        bind.output=operand(1);
        int sequence=int(Integer(stage,"width")),heads=int(Integer(stage,"extent"));
        int head_dim=int(Integer(stage,"group"));
        auto const& l=layouts.at(packed);
        if(sequence!=128 || heads!=12 || head_dim!=64 || l.rank!=2 ||
            l.physical[0]!=unsigned(model_batch*sequence) || l.physical[1]!=2304)
          throw std::invalid_argument("F3 attention requires packed Bx128x12x3x64 QKV");
        for(unsigned part=0;part<3;++part) {
          MapSpec spec;
          spec.buffer=packed;spec.rank=3;spec.offset=std::uint64_t(part)*head_dim;
          spec.dims[0]=64;spec.dims[1]=12;spec.dims[2]=unsigned(model_batch*sequence);
          spec.strides[0]=std::uint64_t(head_dim)*3*2;spec.strides[1]=l.strides[0]*2;
          spec.box[0]=64;spec.box[1]=1;
          spec.box[2]=part==0?std::min(128,16*model_batch):128;
          auto id=maps.Add(spec);
          if(part==0)bind.query_map=id;
          else if(part==1)bind.map=id;
          else bind.value_map=id;
        }
        body="Sm90F3Attention<"+std::to_string(model_batch)+','+
            std::to_string(std::min(128,16*model_batch))+'>';
        wrapper="AttentionStage";
        break;
      }
      default:
        throw std::invalid_argument("the F3 forward lowering does not implement the body "+
            std::string(solver::kSm90ForwardBodies[unsigned(c.body)]));
    }

    types<<bind.Emit(index)<<shape.Emit(index)
         <<"using Body_"<<index<<'='<<body<<";\n"
         <<"using Wrap_"<<index<<'='<<wrapper<<"<Body_"<<index<<",Bind_"<<index
         <<",Shape_"<<index<<">;\n"
         // A stage whose payload exceeds one arena spans both and starts at
         // the global header (§5.2 item 3).
         <<"constexpr int kOffset_"<<index<<"=Sm90F3Budget::Offset("<<index
         <<",Wrap_"<<index<<"::kSharedBytes);\n"
         <<"using Stage_"<<index<<"=Placement<Wrap_"<<index<<",kOffset_"<<index<<">;\n";
    placements.push_back("Stage_"+std::to_string(index));
  }

  out<<"constexpr std::array<tm_buffer_info,"<<(buffers.size()+scratch.size())
     <<"> kF3ForwardBuffers={{\n"<<table.str();
  for(auto const& item:scratch)
    out<<"  {"<<Quoted(item.name)<<",TM_BUFFER_INTERNAL,"
       <<(item.dtype=="f32"?"TM_DTYPE_F32":"TM_DTYPE_I32")<<','<<item.elements
       <<"ull,0ull,\"\"},\n";
  out<<"}};\n"<<types.str()
     <<"using F3ForwardProgram=Program<"<<model_batch<<','
     <<(buffers.size()+scratch.size())<<','<<maps.specs.size();
  for(auto const& name:placements)out<<','<<name;
  out<<">;\n"
     <<"constexpr std::array<ForwardMapSpec,"<<maps.specs.size()<<"> kF3ForwardMaps={{\n";
  for(auto const& spec:maps.specs)out<<spec.Emit();
  out<<"}};\n"
     <<"}  // namespace tilemega::codegen::forward_f3\n"
     <<"#include <tilemega/Codegen/tasks/Sm90F3Runtime.cuh>\n";
  return out.str();
}

}  // namespace tilemega::codegen
