// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/SemanticLifting.h>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace tilemega::frontend {
namespace {
using namespace analysis;
constexpr auto missing=codegen::kDmNoIndex;
ClosedForm C(long value) {return ClosedForm::Constant(value);}
IndexResult I(char const* dim,long scale=1,long group=1) {
  return IndexResult::Dim(dim,C(scale),C(group));
}
IndexResult Add(std::initializer_list<IndexResult> parts,long offset=0) {
  auto result=IndexResult::Affine({},C(offset));
  for(auto const& part:parts)
    result.terms.insert(result.terms.end(),part.terms.begin(),part.terms.end());
  return result;
}
IterationDim D(char const* name,ClosedForm extent,bool reduction=false) {
  IterationDim result;result.name=name;result.extent=std::move(extent);
  if(reduction)result.type=IteratorType::kReduction;
  return result;
}
TensorSpace T(std::string name,std::vector<TensorAxis> axes) {
  TensorSpace result;result.name=std::move(name);result.axes=std::move(axes);return result;
}
struct Builder {
  ModelPlan const& plan;
  ClosedForm batch;
  std::unordered_map<unsigned,std::string> writer;
  std::unordered_map<unsigned,TensorSpace> written;
  LiftedModel result;

  explicit Builder(ModelPlan const& p,ClosedForm b):plan(p),batch(std::move(b)) {
    result.has_plan=true;result.written.assign(plan.buffers.size(),0);
  }
  PlanBuffer const& Buffer(unsigned id) const {return plan.buffers.at(id);}
  std::string Producer(unsigned id) const {
    auto found=writer.find(id);return found==writer.end()?std::string():found->second;
  }
  TensorSpace Space(unsigned id,ClosedForm rows,unsigned width) const {
    auto found=written.find(id);if(found!=written.end())return found->second;
    auto const& buffer=Buffer(id);auto const& layout=buffer.layout;
    if(!layout.rank)return T(buffer.name,{{"row",rows},{"channel",C(width)}});
    std::vector<TensorAxis> axes;
    for(unsigned axis=0;axis<layout.rank;++axis) {
      auto extent=C(layout.physical[axis]);
      if(axis==0 && layout.kind==codegen::DmLayout::kNHWC)extent=batch;
      // LayoutConvert writes alignment padding through the entire pixel pitch.
      if(axis==3 && layout.kind==codegen::DmLayout::kNHWC)extent=C(layout.strides[2]);
      axes.push_back({"d"+std::to_string(axis),extent});
    }
    return T(buffer.name,std::move(axes));
  }
  std::vector<IndexResult> Rows(unsigned id,char const* column) const {
    auto const& l=Buffer(id).layout;
    if(!l.rank)return {I("m"),I(column)};
    std::vector<IndexResult> map(l.rank);map.back()=I(column);
    long trailing=1;
    for(unsigned axis=l.rank-1;axis-->0;) {
      auto index=I("m",1,trailing);
      if(axis)index=Add({index,I("m",-long(l.logical[axis]),trailing*l.logical[axis])});
      if(l.kind==codegen::DmLayout::kNHWC) {
        if(axis==1)index.offset=C(l.halo_top);
        if(axis==2)index.offset=C(l.halo_left);
      }
      map[axis]=std::move(index);trailing*=l.logical[axis];
    }
    return map;
  }
  std::pair<TensorSpace,std::vector<IndexResult>> Mapped(unsigned id,
      codegen::DmWriteMap const& mapping,ClosedForm rows,unsigned columns) const {
    if(mapping.layout!=missing && mapping.layout!=id)
      throw std::invalid_argument("DNN map layout must describe its tensor");
    auto const& l=Buffer(id).layout;
    if(l.rank && mapping.layout==missing)
      throw std::invalid_argument("DNN structured output requires its device layout map");
    if(mapping.kind==codegen::DmWriteKind::kDense)
      return {Space(id,rows,columns),Rows(id,"n")};
    if(l.rank!=4)throw std::invalid_argument("DNN image map requires a rank-four layout");
    auto space=Space(id,rows,columns);space.axes[0].extent=batch;
    if(mapping.kind==codegen::DmWriteKind::kNCHW) {
      if(l.kind!=codegen::DmLayout::kRowMajor || l.logical[1]!=columns)
        throw std::invalid_argument("DNN NCHW map disagrees with its channels");
      auto h=l.logical[2],w=l.logical[3];
      if(!h || !w)throw std::invalid_argument("empty NCHW image map");
      return {space,{I("m",1,h*w),I("n"),
          Add({I("m",1,w),I("m",-long(h),h*w)}),
          Add({I("m"),I("m",-long(w),w)})}};
    }
    if(mapping.kind==codegen::DmWriteKind::kPixelShuffle) {
      auto r=mapping.factor;
      if(!r || l.kind!=codegen::DmLayout::kNHWC || l.logical[1]%r ||
         l.logical[2]%r || std::uint64_t(l.logical[3])*r*r!=columns)
        throw std::invalid_argument("DNN pixel shuffle map disagrees with its image geometry");
      auto h=l.logical[1]/r,w=l.logical[2]/r;
      if(!h || !w)throw std::invalid_argument("empty pixel shuffle image map");
      return {space,{I("m",1,h*w),
          Add({I("m",r,w),I("m",-long(r)*h,h*w),I("n",1,r),I("n",-long(r),r*r)},l.halo_top),
          Add({I("m",r),I("m",-long(r)*w,w),I("n"),I("n",-long(r),r)},l.halo_left),
          I("n",1,r*r)}};
    }
    throw std::invalid_argument("DNN write map requires explicit bound scatter semantics");
  }
  void Chain(SemanticOp& op,PlanGemm const& g,ClosedForm rows) const {
    if(g.chain.count>8 || g.chain.side_count>5)
      throw std::invalid_argument("invalid DNN epilogue descriptor length");
    for(unsigned i=0;i<g.chain.count;++i) {
      auto const& step=g.chain.operations[i];
      using K=codegen::DmEpilogueKind;
      if(step.kind==K::kBias || step.kind==K::kScale) {
        auto id=step.parameter[0];
        op.epilogue_operands.push_back(Read(id,
            T(Buffer(id).name,{{"channel",C(g.n)}}),{I("n")}));
      }else if(step.kind==K::kResidual) {
        auto id=step.parameter[0];auto mapped=Mapped(id,step.residual_map,rows,g.n);
        op.epilogue_operands.push_back(Read(id,std::move(mapped.first),std::move(mapped.second)));
        if(step.parameter[1]!=missing) {
          auto scale=step.parameter[1];
          op.epilogue_operands.push_back(Read(scale,
              T(Buffer(scale).name,{{"channel",C(g.n)}}),{I("n")}));
        }
      }else if(step.kind!=K::kActivation) {
        throw std::invalid_argument("DNN GEMM pairing/norm requires explicit reduction semantics");
      }
    }
  }
  void GemmSideStores(SemanticOp& op,PlanGemm const& g,ClosedForm rows) {
    std::set<unsigned> seen;
    for(unsigned i=0;i<g.chain.side_count;++i) {
      auto const& side=g.chain.side[i];auto id=side.buffer;
      if(id==missing || !seen.insert(id).second || Buffer(id).dtype!="f32")
        throw std::invalid_argument("DNN GEMM side output requires distinct FP32 storage");
      TensorSpace space;std::vector<IndexResult> map;
      if(side.kind==codegen::DmSideOutputKind::kRowStats) {
        space=T(Buffer(id).name,{{"row",rows},{"stat",C(2)}});
        for(unsigned stat=0;stat<2;++stat) {
          ElementWrite write;write.tensor=space;write.effect.kind=EffectKind::kWrite;
          write.map.results={I("m"),IndexResult::Affine({},C(stat))};
          op.additional_writes.push_back(std::move(write));
        }
        op.tile_storage.push_back({space.name,"n",1});
      }else if(side.kind==codegen::DmSideOutputKind::kChannelPartialSums) {
        if(!g.access.rows_per_batch)
          throw std::invalid_argument("DNN channel partials require per-image rows");
        space=T(Buffer(id).name,{{"image",batch},{"channel",C(g.n)}});
        ElementWrite write;write.tensor=space;write.effect.kind=EffectKind::kWrite;
        write.map.results={I("m",1,g.access.rows_per_batch),I("n")};
        op.additional_writes.push_back(std::move(write));
        op.tile_storage.push_back({space.name,"m",1});
      }else {
        throw std::invalid_argument("DNN GEMM selection/partial side stores require their reduction semantics");
      }
      writer[id]=op.name;written[id]=space;result.written[id]=1;
    }
  }
  SemanticOperand Read(unsigned id,TensorSpace space,std::vector<IndexResult> map) const {
    SemanticOperand read;read.producer=Producer(id);read.tensor=std::move(space);
    read.map.results=std::move(map);return read;
  }
  void Own(SemanticOp& op,ClosedForm rows,unsigned columns) const {
    op.exact_task_access=true;
    op.task_space=T(op.name+".owners",{{"m",rows},{"n",C(columns)}});
    op.task_map.results={I("m"),I("n")};
  }
  void Stats(SemanticOp& op,unsigned id,ClosedForm rows) {
    if(id==missing)return;
    // Both statistics are written after all channel lanes finish. Their
    // two coordinates share the same row owner; duplicate element writes
    // collapse under relation union rather than adding another task space.
    auto space=T(Buffer(id).name,{{"row",rows},{"stat",C(2)}});
    for(unsigned stat=0;stat<2;++stat) {
      ElementWrite write;write.tensor=space;
      write.map.results={I("m"),IndexResult::Affine({},C(stat))};
      write.effect.kind=EffectKind::kWrite;
      op.additional_writes.push_back(std::move(write));
    }
    writer[id]=op.name;written[id]=space;result.written[id]=1;
  }
  void Record(unsigned stage,SemanticOp op,OpRole role,unsigned output) {
    op.dtype=Buffer(output).dtype=="f32"?ScalarType::kF32:ScalarType::kBF16;
    op.result_effect.kind=EffectKind::kWrite;
    writer[output]=op.name;written[output]=op.result;result.written[output]=1;
    result.ops.push_back({op.name,role,OwnershipKind::kTilePerBlock,int(stage),0,
                          plan.stages[stage].representative});
    result.sem.ops.push_back(std::move(op));
  }
};
}

LiftedModel LiftDnnSemantics(ModelPlan const& plan,LiftOptions const& options) {
  if(!plan.dm || !plan.forward || !options.forward || options.batch_symbol.empty() ||
     plan.forward_token_axis || options.static_seq<=0)
    throw std::invalid_argument("DNN semantics require a symbolic batch and fixed sequence");
  Builder b(plan,ClosedForm::Symbol(options.batch_symbol));
  for(unsigned index=0;index<plan.stages.size();++index) {
    auto const& stage=plan.stages[index];
    auto rows=b.batch*C(stage.rows_per_batch?stage.rows_per_batch:options.static_seq);
    SemanticOp op;op.name="dnn.s"+std::to_string(index);
    if(stage.kind==PlanTaskKind::kLayoutConvert) {
      auto input=stage.operands[0],output=stage.operands[1];
      auto const& l=b.Buffer(output).layout;
      if(l.kind!=codegen::DmLayout::kNHWC || l.rank!=4 ||
         stage.rows_per_batch!=l.logical[1]*l.logical[2])
        throw std::invalid_argument("layout conversion ownership differs from its image geometry");
      auto h=l.logical[1],w=l.logical[2],c=l.logical[3];
      auto input_space=T(b.Buffer(input).name,
          {{"image",b.batch},{"channel",C(c)},{"height",C(h)},{"width",C(w)}});
      op.kind=OperatorKind::kTranspose;op.arithmetic="layout_convert";
      op.domain={D("m",rows),D("n",C(l.strides[2]))};
      op.result=b.Space(output,rows,l.strides[2]);op.result_map.results=b.Rows(output,"n");
      auto read=b.Read(input,input_space,{I("m",1,h*w),I("n"),
          Add({I("m",1,w),I("m",-long(h),h*w)}),Add({I("m"),I("m",-long(w),w)})});
      op.operands={read};
      op.element_reads={{read.tensor,read.map,{Add({I("n",-1)},c-1)}}};
      b.Own(op,rows,l.strides[2]);b.Record(index,std::move(op),OpRole::kLayoutConvert,output);
    }else if(stage.kind==PlanTaskKind::kLayerNorm) {
      auto x=stage.operands[0],gamma=stage.operands[1],beta=stage.operands[2];
      auto out=stage.operands[3],width=stage.width;
      op.kind=OperatorKind::kReduction;op.arithmetic="layernorm";
      op.domain={D("m",rows),D("n",C(width)),D("c",C(width),true)};
      op.result=b.Space(out,rows,width);op.result_map.results=b.Rows(out,"n");
      op.operands={b.Read(x,b.Space(x,rows,width),b.Rows(x,"c")),
          b.Read(gamma,T(b.Buffer(gamma).name,{{"channel",C(width)}}),{I("n")}),
          b.Read(beta,T(b.Buffer(beta).name,{{"channel",C(width)}}),{I("n")})};
      b.Own(op,rows,width);b.Stats(op,stage.operands[4],rows);
      b.Record(index,std::move(op),OpRole::kLayerNorm,out);
    }else if(stage.kind==PlanTaskKind::kEmbeddingSum) {
      auto width=stage.width,out=stage.operands[5];
      op.kind=OperatorKind::kGather;op.arithmetic="embedding_sum";
      // The three additions' inputs form a semantic reduction of length
      // three. The device still observes the ordered BF16 rounding points.
      op.domain={D("m",rows),D("n",C(width)),D("table",C(3),true)};
      op.result=b.Space(out,rows,width);op.result_map.results=b.Rows(out,"n");
      for(unsigned id:{stage.operands[0],stage.operands[1]})
        op.operands.push_back(b.Read(id,T(b.Buffer(id).name,{{"row",rows}}),{I("m")}));
      for(unsigned table=0;table<3;++table) {
        auto id=stage.operands[2+table];auto const& l=b.Buffer(id).layout;
        auto map=table==2?Add({I("m"),I("m",-options.static_seq,options.static_seq)}):
            IndexResult::DataDependent(b.Buffer(stage.operands[table]).name,{"m"});
        op.operands.push_back(b.Read(id,
            T(b.Buffer(id).name,{{"index",C(l.logical[0])},{"channel",C(width)}}),
            {map,I("n")}));
      }
      b.Own(op,rows,width);b.Stats(op,stage.operands[6],rows);
      b.Record(index,std::move(op),OpRole::kEmbeddingSum,out);
    }else if(stage.kind==PlanTaskKind::kEncoderAttention) {
      auto packed=stage.operands[0],output=stage.operands[1],mask=stage.operands[2];
      auto sequence=stage.width,heads=stage.extent;
      if((sequence!=128 && sequence!=384 && sequence!=512) || !heads ||
         (stage.group!=64 && stage.group!=128) || stage.rows_per_batch!=sequence)
        throw std::invalid_argument("encoder attention ownership differs from packed geometry");
      auto packed_space=T(b.Buffer(packed).name,{{"row",b.batch*C(sequence)},
          {"channel",C(heads*192)}});
      auto row=Add({I("m"),I("m",-long(sequence),sequence),I("m",sequence,heads*sequence)});
      auto head=Add({I("m",192,sequence),I("m",-long(heads)*192,heads*sequence)});
      auto channel=[&](char const* dim,long offset) {
        auto value=head;value.terms.push_back(I(dim).terms.front());value.offset=C(offset);return value;
      };
      auto kv_row=Add({I("m",sequence,heads*sequence),I("j")});
      op.kind=OperatorKind::kReduction;op.arithmetic="encoder_attention";
      op.domain={D("m",b.batch*C(heads*sequence)),D("n",C(64)),
          D("j",C(sequence),true),D("d",C(64),true)};
      op.result=T(b.Buffer(output).name,{{"row",b.batch*C(sequence)},{"channel",C(heads*64)}});
      op.result_map.results={row,Add({I("m",64,sequence),I("m",-long(heads)*64,heads*sequence),I("n")})};
      op.operands={b.Read(packed,packed_space,{row,channel("d",0)}),
          b.Read(packed,packed_space,{kv_row,channel("d",64)}),
          b.Read(packed,packed_space,{kv_row,channel("n",128)})};
      if(mask!=missing)op.operands.push_back(b.Read(mask,
          T(b.Buffer(mask).name,{{"image",b.batch},{"token",C(sequence)}}),
          {I("m",1,heads*sequence),I("j")}));
      b.Own(op,b.batch*C(heads*sequence),64);
      b.Record(index,std::move(op),OpRole::kEncoderAttention,output);
    }else if(stage.kind==PlanTaskKind::kDepthwiseConv) {
      auto input=stage.operands[0],weight=stage.operands[1],output=stage.operands[2];
      auto const& conv=plan.convolutions.at(stage.conv);
      auto const& layout=b.Buffer(input).layout;
      unsigned gates=0;
      for(unsigned i=0;i<stage.chain.count;++i)
        gates+=stage.chain.operations[i].kind==codegen::DmEpilogueKind::kGatePair;
      auto channels=conv.c/(gates?2:1),bands=(conv.p+stage.group-1)/stage.group;
      if(gates>1 || !channels || channels!=stage.extent || conv.c!=conv.k ||
         conv.input_layout!=input || conv.output_layout!=output ||
         stage.rows_per_batch!=conv.p*conv.q)
        throw std::invalid_argument("depthwise ownership differs from its convolution");
      auto channel=I("n");
      if(gates)channel=Add({I("n"),I("pair",channels)});
      op.kind=OperatorKind::kReduction;
      op.arithmetic=gates?"depthwise_simple_gate":"depthwise_conv";
      op.domain={D("m",rows),D("n",C(channels)),D("r",C(conv.r),true),D("s",C(conv.s),true)};
      if(gates)op.domain.push_back(D("pair",C(2),true));
      op.result=b.Space(output,rows,channels);op.result_map.results=b.Rows(output,"n");
      auto input_map=std::vector<IndexResult>{I("m",1,conv.p*conv.q),
          Add({I("m",conv.stride_h,conv.q),I("m",-long(conv.stride_h)*conv.p,conv.p*conv.q),
               I("r",conv.dilation_h)},long(layout.halo_top)-conv.pad_h),
          Add({I("m",conv.stride_w),I("m",-long(conv.stride_w)*conv.q,conv.q),
               I("s",conv.dilation_w)},long(layout.halo_left)-conv.pad_w),channel};
      op.operands={b.Read(input,b.Space(input,b.batch*C(conv.h*conv.w),conv.c),input_map),
          b.Read(weight,T(b.Buffer(weight).name,{{"channel",C(conv.c)},
              {"r",C(conv.r)},{"s",C(conv.s)},{"padding",C(8)}}),
              {channel,I("r"),I("s"),IndexResult::Affine({},C(0))})};
      bool contracted=false;
      for(unsigned i=0;i<stage.chain.count;++i) {
        auto const& step=stage.chain.operations[i];
        if(step.kind==codegen::DmEpilogueKind::kGatePair)contracted=true;
        if(step.kind==codegen::DmEpilogueKind::kBias || step.kind==codegen::DmEpilogueKind::kScale) {
          auto id=step.parameter[0];
          op.operands.push_back(b.Read(id,T(b.Buffer(id).name,
              {{"channel",C(contracted?channels:conv.c)}}),{contracted?I("n"):channel}));
        }
      }
      b.Own(op,b.batch*C(bands*stage.group*conv.q),channels);
      // Pad each image's ownership range independently: a final short row
      // band must never take ownership of the next image's first rows.
      op.task_map.results[0]=Add({I("m"),I("m",long(bands*stage.group-conv.p)*conv.q,conv.p*conv.q)});
      if(stage.operands[3]!=missing) {
        auto id=stage.operands[3];auto space=T(b.Buffer(id).name,
            {{"image",b.batch},{"band",C(bands)},{"channel",C(channels)}});
        ElementWrite write;write.tensor=space;write.effect.kind=EffectKind::kWrite;
        auto local_band=Add({I("m"),I("m",-long(conv.p)*conv.q,conv.p*conv.q)});
        local_band.outer_divisor=C(stage.group*conv.q);
        write.map.results={I("m",1,conv.p*conv.q),local_band,I("n")};
        op.additional_writes.push_back(std::move(write));
        b.writer[id]=op.name;b.written[id]=space;b.result.written[id]=1;
      }
      b.Record(index,std::move(op),OpRole::kDepthwiseConv,output);
    }else if(stage.kind==PlanTaskKind::kPool) {
      auto input=stage.operands[0],output=stage.operands[1];
      auto const& conv=plan.convolutions.at(stage.conv);
      auto const& layout=b.Buffer(input).layout;
      if(conv.input_layout!=input || conv.output_layout!=output || conv.c!=stage.extent ||
         conv.k!=conv.c || stage.rows_per_batch!=conv.p*conv.q)
        throw std::invalid_argument("pool ownership differs from its window geometry");
      op.kind=OperatorKind::kReduction;op.arithmetic="pool";
      op.domain={D("m",rows),D("n",C(conv.c)),D("r",C(conv.r),true),D("s",C(conv.s),true)};
      op.result=b.Space(output,rows,conv.c);op.result_map.results=b.Rows(output,"n");
      auto y=Add({I("m",conv.stride_h,conv.q),
          I("m",-long(conv.stride_h)*conv.p,conv.p*conv.q),I("r",conv.dilation_h)},-long(conv.pad_h));
      auto x=Add({I("m",conv.stride_w),I("m",-long(conv.stride_w)*conv.q,conv.q),
          I("s",conv.dilation_w)},-long(conv.pad_w));
      auto negate=[](IndexResult value,long maximum) {
        value.offset=C(maximum)+C(-1)*value.offset;
        for(auto& term:value.terms)term.coefficient=C(-1)*term.coefficient;
        return value;
      };
      auto py=y,px=x;py.offset=py.offset+C(layout.halo_top);px.offset=px.offset+C(layout.halo_left);
      auto read=b.Read(input,b.Space(input,b.batch*C(conv.h*conv.w),conv.c),
          {I("m",1,conv.p*conv.q),py,px,I("n")});
      op.operands={read};
      op.element_reads={{read.tensor,read.map,{y,negate(y,conv.h-1),x,negate(x,conv.w-1)}}};
      b.Own(op,rows,conv.c);b.Record(index,std::move(op),OpRole::kPool,output);
    }else if(stage.kind==PlanTaskKind::kGlobalPoolReduce) {
      auto input=stage.operands[0],output=stage.operands[1];
      auto area=stage.rows_per_batch,tile=stage.group,channels=stage.extent;
      if(!area || !tile || !channels)throw std::invalid_argument("empty global pool reduction");
      auto parts=stage.partial_rows_per_image?C(stage.partial_rows_per_image):
          (b.batch*C(area)).CeilDiv(C(tile));
      op.kind=OperatorKind::kReduction;op.arithmetic="global_pool_reduce";
      bool partitioned=false;
      for(auto const& prior:b.result.sem.ops)for(auto const& store:prior.tile_storage)
        partitioned|=store.tensor==b.Buffer(input).name;
      op.domain={D("m",b.batch),D("n",C(channels)),D("r",partitioned?C(area):parts,true)};
      op.domain_nonnegative={Add({I("r",tile),I("m",-long(area))},tile-1),
          Add({I("m",area),I("r",-long(tile))},area-1)};
      if(stage.partial_rows_per_image)op.domain_nonnegative.clear();
      op.result=b.Space(output,b.batch,channels);op.result_map.results=b.Rows(output,"n");
      op.operands={b.Read(input,T(b.Buffer(input).name,
          {{"image",b.batch},{"part",parts},{"channel",C(channels)}}),{I("m"),I("r"),I("n")})};
      if(partitioned) {
        op.domain_nonnegative.clear();
        op.operands={b.Read(input,b.written.at(input),{I("m"),I("n")})};
        op.tile_storage_reads.push_back({b.Buffer(input).name,"r","m",C(area)});
      }
      b.Own(op,b.batch,channels);b.Record(index,std::move(op),OpRole::kGlobalPoolReduce,output);
    }else if(stage.kind==PlanTaskKind::kGemm) {
      auto const& g=plan.gemms.at(stage.gemm);
      if(g.beta!=0 || g.epilogue!=PlanGemm::Epilogue::kStore || g.access.a_scale!=missing ||
         g.access.b!=codegen::DmBAccess::kDense)
        throw std::invalid_argument("DNN GEMM requires explicit semantics for its epilogue/access recipe");
      rows=b.batch*C(g.access.rows_per_batch?g.access.rows_per_batch:options.static_seq);
      op.kind=OperatorKind::kMatmul;op.arithmetic="gemm";
      auto output=b.Mapped(g.d,g.access.write,rows,g.n);
      op.result=std::move(output.first);op.result_map.results=std::move(output.second);
      op.reduction.splittable=true;op.reduction.reduction_operator="add";
      op.reduction.partial_tensor=op.name+".partial";op.reduction.combiner=op.name+".combine";
      op.reduction.ownership={"m","n"};
      if(g.access.a==codegen::DmAAccess::kIm2Col) {
        auto const& conv=plan.convolutions.at(g.access.conv);
        auto const& l=b.Buffer(conv.input_layout).layout;
        if(conv.input_layout!=g.a || g.access.rows_per_batch!=conv.p*conv.q)
          throw std::invalid_argument("convolution row or input map differs from its descriptor");
        op.domain={D("m",rows),D("n",C(g.n)),D("c",C(conv.c),true),
            D("r",C(conv.r),true),D("s",C(conv.s),true)};
        op.operands={b.Read(g.a,b.Space(g.a,b.batch*C(conv.h*conv.w),conv.c),
            {I("m",1,conv.p*conv.q),
             Add({I("m",conv.stride_h,conv.q),I("m",-long(conv.stride_h)*conv.p,conv.p*conv.q),
                  I("r",conv.dilation_h)},long(l.halo_top)-conv.pad_h),
             Add({I("m",conv.stride_w),I("m",-long(conv.stride_w)*conv.q,conv.q),
                  I("s",conv.dilation_w)},long(l.halo_left)-conv.pad_w),I("c")}),
            b.Read(g.b,T(b.Buffer(g.b).name,{{"output",C(g.n)},{"r",C(conv.r)},
                {"s",C(conv.s)},{"channel",C(l.physical[3])}}),{I("n"),I("r"),I("s"),I("c")})};
        op.reduction.dim="c";
      }else if(g.access.a==codegen::DmAAccess::kDense) {
        auto const& layout=b.Buffer(g.a).layout;
        if(layout.rank) {
          auto pitch=std::uint64_t(g.k);
          if(layout.logical[layout.rank-1]!=g.k || layout.strides[layout.rank-1]!=1 ||
             layout.halo_top || layout.halo_bottom || layout.halo_left || layout.halo_right)
            throw std::invalid_argument("DNN dense A requires contiguous rows without halo");
          for(unsigned axis=layout.rank-1;axis-->0;) {
            if(layout.strides[axis]!=pitch)
              throw std::invalid_argument("DNN dense A physical row pitch differs from GEMM K");
            pitch*=layout.logical[axis];
          }
        }
        op.domain={D("m",rows),D("n",C(g.n)),D("k",C(g.k),true)};
        auto map=b.Rows(g.a,"k");
        if(g.access.a_row_offset || g.access.a_row_stride>1) {
          if(b.Buffer(g.a).layout.rank)
            throw std::invalid_argument("strided dense row map requires flattened storage");
          map[0]=Add({I("m",std::max(1u,g.access.a_row_stride))},g.access.a_row_offset);
        }
        auto source_rows=rows*C(std::max(1u,g.access.a_row_stride))+C(g.access.a_row_offset);
        op.operands={b.Read(g.a,b.Space(g.a,source_rows,g.k),map),
            b.Read(g.b,T(b.Buffer(g.b).name,{{"output",C(g.n)},{"channel",C(g.k)}}),{I("n"),I("k")})};
        op.reduction.dim="k";
      }else throw std::invalid_argument("DNN GEMM has an unsupported A map");
      b.Chain(op,g,rows);
      b.Own(op,rows,g.n);b.GemmSideStores(op,g,rows);
      b.Record(index,std::move(op),OpRole::kProjection,g.d);
    }else throw std::invalid_argument("DNN L-sem has no rule for plan stage "+std::to_string(index));
  }
  return std::move(b.result);
}
} // namespace tilemega::frontend
