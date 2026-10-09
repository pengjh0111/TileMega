// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DnnModelPlan.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/FormatVariadic.h>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace tilemega::frontend {
namespace {
using namespace codegen;
constexpr auto missing=kDmNoIndex;
std::string Op(FxNodeRecord const& n) {
  if(n.target=="<built-in function getitem>")return "getitem";
  if(n.target.rfind("aten.",0)!=0)return n.target;
  return n.target.substr(5,n.target.find('.',5)-5);
}
unsigned U(std::string const& s) {
  if(s.empty() || s.find_first_not_of("0123456789")!=std::string::npos)
    throw std::invalid_argument("DNN requires a bound non-batch extent: "+s);
  auto value=std::stoull(s);
  if(!value || value>std::numeric_limits<unsigned>::max())throw std::invalid_argument("invalid DNN extent");
  return value;
}
unsigned Checked(std::uint64_t value) {
  if(value>std::numeric_limits<unsigned>::max())throw std::invalid_argument("DNN buffer exceeds ABI capacity");
  return value;
}
FxArgument const& Arg(FxNodeRecord const& n,unsigned i) {
  static FxArgument none;
  return i<n.args.size()?n.args[i]:none;
}
std::string Ref(FxNodeRecord const& n,unsigned i) {
  auto const& a=Arg(n,i);
  if(a.kind!=FxArgument::Kind::kNode)throw std::invalid_argument("DNN argument must reference a tensor: "+n.name);
  return a.text;
}
long Int(FxArgument const& a,long fallback) {
  if(a.kind==FxArgument::Kind::kNone)return fallback;
  if(a.kind!=FxArgument::Kind::kInt)throw std::invalid_argument("DNN expected an integer argument");
  return a.integer;
}
double Real(FxArgument const& a,double fallback) {
  if(a.kind==FxArgument::Kind::kNone)return fallback;
  if(a.kind==FxArgument::Kind::kFloat)return a.real;
  if(a.kind==FxArgument::Kind::kInt)return a.integer;
  throw std::invalid_argument("DNN expected a numeric argument");
}
std::pair<unsigned,unsigned> Pair(FxArgument const& a,unsigned fallback,bool zero=false) {
  long x=fallback,y=fallback;
  if(a.kind==FxArgument::Kind::kInt)x=y=a.integer;
  else if(a.kind==FxArgument::Kind::kList && !a.items.empty()) {
    if(a.items.size()>2)throw std::invalid_argument("DNN window must have one or two coordinates");
    x=Int(a.items[0],fallback);y=Int(a.items.back(),fallback);
  }else if(a.kind!=FxArgument::Kind::kNone && a.kind!=FxArgument::Kind::kList)
    throw std::invalid_argument("DNN expected a window argument");
  if(x<(zero?0:1) || y<(zero?0:1) || x>UINT32_MAX || y>UINT32_MAX)
    throw std::invalid_argument("DNN window outside descriptor range");
  return {unsigned(x),unsigned(y)};
}
std::string Json(llvm::json::Object value) {return llvm::formatv("{0}",llvm::json::Value(std::move(value))).str();}
struct Builder {
  std::vector<FxNodeRecord> const& nodes;
  DnnPlanOptions options;
  ModelPlan p;
  std::map<std::string,FxNodeRecord const*> records;
  std::map<std::string,std::string> parameters,parameter_root;
  std::map<unsigned,unsigned> owner;
  std::set<unsigned> consumed;
  std::set<std::string> shape_values;
  std::set<std::string> externally_observed;
  std::map<std::string,unsigned> users;
  Builder(std::vector<FxNodeRecord> const& n,std::vector<SignatureInput> const& inputs,
          std::vector<std::string> const& outputs,DnnPlanOptions o)
      :nodes(n),options(o) {
    if(!o.batch || o.batch>64 || (o.small_input_channels!=4 && o.small_input_channels!=8))
      throw std::invalid_argument("invalid DNN batch or small-channel layout");
    p.dm=p.forward=true;p.dtype="bf16";p.serving_seq=1;
    externally_observed.insert(outputs.begin(),outputs.end());
    for(auto const& x:nodes) {records[x.name]=&x;for(auto const& input:x.inputs)++users[input];}
    for(auto const& x:inputs)if(x.kind=="PARAMETER" || x.kind=="BUFFER") {
      parameters[x.name]=x.target;parameter_root[x.name]=x.name;
    }
    for(auto const& x:inputs)if(x.kind=="USER_INPUT")Input(*records.at(x.name));
  }
  unsigned Add(PlanBuffer b) {auto id=p.buffers.size();p.buffers.push_back(std::move(b));return id;}
  unsigned Value(std::string const& name) const {
    auto found=p.node_buffer.find(name);
    if(found==p.node_buffer.end())throw std::invalid_argument("DNN tensor has no executable producer: "+name);
    return found->second;
  }
  void Alias(FxNodeRecord const& n,std::string const& source) {
    if(parameters.count(source)) {parameters[n.name]=parameters.at(source);parameter_root[n.name]=parameter_root.at(source);}
    else p.node_buffer[n.name]=Value(source);
  }
  DmBufferLayout Image(unsigned h,unsigned w,unsigned c,bool nchw=false) const {
    DmBufferLayout l;l.rank=4;l.kind=nchw?DmLayout::kRowMajor:DmLayout::kNHWC;
    unsigned logical[]={options.batch,nchw?c:h,nchw?h:w,nchw?w:c};
    std::copy(logical,logical+4,l.logical);std::copy(logical,logical+4,l.physical);
    if(!nchw)l.physical[3]=c<=4?options.small_input_channels:(c+7)/8*8;
    Strides(l);return l;
  }
  static void Strides(DmBufferLayout& l) {
    l.strides[l.rank-1]=1;
    for(unsigned axis=l.rank-1;axis-->0;)l.strides[axis]=l.strides[axis+1]*l.physical[axis+1];
  }
  unsigned ImageBuffer(std::string name,unsigned h,unsigned w,unsigned c) {
    PlanBuffer b;b.name=std::move(name);b.layout=Image(h,w,c);b.per_batch=Checked(b.layout.strides[0]);return Add(std::move(b));
  }
  unsigned RowsBuffer(std::string name,unsigned rows,unsigned channels,char const* dtype="bf16") {
    PlanBuffer b;b.name=std::move(name);b.dtype=dtype;b.per_batch=Checked(std::uint64_t(rows)*channels);
    b.layout.rank=2;b.layout.logical[0]=b.layout.physical[0]=Checked(std::uint64_t(options.batch)*rows);
    b.layout.logical[1]=b.layout.physical[1]=channels;b.layout.strides[1]=1;b.layout.strides[0]=channels;
    return Add(std::move(b));
  }
  unsigned Weight(std::string name,llvm::json::Object recipe,unsigned elements,char const* dtype="bf16") {
    PlanBuffer b;b.name=std::move(name);b.source=PlanBuffer::Source::kWeight;b.constant=elements;
    b.dtype=dtype;b.role="external";b.external_name=b.name;b.pack_json=Json(std::move(recipe));return Add(std::move(b));
  }
  std::string Fqn(std::string const& ref) const {
    auto it=parameters.find(ref);if(it==parameters.end())throw std::invalid_argument("DNN weight is not a parameter: "+ref);
    return it->second;
  }
  FxNodeRecord const& Parameter(std::string const& ref) const {return *records.at(parameter_root.at(ref));}
  unsigned Bias(std::string const& ref) {
    auto const& node=Parameter(ref);std::uint64_t elements=1;for(auto const& x:node.shape)elements*=U(x);
    return Weight(Fqn(ref)+".bias_f32",llvm::json::Object{{"kind","linear_bias"},{"source",Fqn(ref)}},Checked(elements),"f32");
  }
  unsigned Stage(PlanTaskKind kind,FxNodeRecord const& n,unsigned output,
                 std::initializer_list<unsigned> operands={}) {
    PlanStage stage;stage.kind=kind;stage.operands.fill(missing);
    std::copy(operands.begin(),operands.end(),stage.operands.begin());
    stage.representative=n.name;stage.representative_index=n.index;
    unsigned id=p.stages.size();p.stages.push_back(stage);owner[output]=id;p.node_buffer[n.name]=output;return id;
  }
  DmEpilogueChain& Chain(unsigned buffer) {
    if(consumed.count(buffer) || !owner.count(buffer))throw std::invalid_argument("DNN epilogue cannot change an already consumed tensor");
    auto& stage=p.stages[owner.at(buffer)];
    if(stage.kind==PlanTaskKind::kGemm)return p.gemms[stage.gemm].chain;
    if(stage.kind==PlanTaskKind::kDepthwiseConv)return stage.chain;
    throw std::invalid_argument("DNN epilogue has no compatible anchor");
  }
  void Append(unsigned buffer,DmEpilogueOp op) {
    auto& chain=Chain(buffer);if(chain.count==8)throw std::invalid_argument("DNN epilogue exceeds eight operations");
    chain.operations[chain.count++]=op;
  }
  void RequireExclusive(std::string const& ref) const {
    if(externally_observed.count(ref) || (users.count(ref) && users.at(ref)>1))
      throw std::invalid_argument("DNN epilogue input has another observable use: "+ref);
  }
  void Input(FxNodeRecord const& n) {
    if(n.dtype!="torch.bfloat16" || n.shape.size()!=4)
      throw std::invalid_argument("DNN input needs the image or encoder structural entry: "+n.name);
    auto c=U(n.shape[1]),h=U(n.shape[2]),w=U(n.shape[3]);
    PlanBuffer b;b.name=n.name;b.layout=Image(h,w,c,true);b.per_batch=Checked(b.layout.strides[0]);
    b.role="external";b.external_name=n.name;unsigned external=Add(std::move(b));
    unsigned internal=ImageBuffer(n.name+".nhwc",h,w,c);
    auto stage=Stage(PlanTaskKind::kLayoutConvert,n,internal,{external,internal});
    p.stages[stage].width=c;p.stages[stage].group=128;p.stages[stage].rows_per_batch=h*w;
  }
  void Halo(unsigned id,unsigned h,unsigned w) {
    auto& b=p.buffers[id];auto& l=b.layout;
    if(l.kind!=DmLayout::kNHWC)throw std::invalid_argument("convolution requires NHWC input");
    l.halo_top=l.halo_bottom=std::max(l.halo_top,h);l.halo_left=l.halo_right=std::max(l.halo_left,w);
    l.physical[1]=l.logical[1]+l.halo_top+l.halo_bottom;
    l.physical[2]=l.logical[2]+l.halo_left+l.halo_right;Strides(l);b.per_batch=Checked(l.strides[0]);
  }
  void Conv(FxNodeRecord const& node) {
    auto a=Value(Ref(node,0));auto weight=Ref(node,1);auto const& shape=Parameter(weight).shape;
    if(shape.size()!=4)throw std::invalid_argument("convolution weight must be OIHW");
    unsigned out_c=U(shape[0]),weight_c=U(shape[1]),r=U(shape[2]),s=U(shape[3]);
    auto in=p.buffers[a].layout;auto stride=Pair(Arg(node,3),1),pad=Pair(Arg(node,4),0,true),dilation=Pair(Arg(node,5),1);
    bool core=Op(node)=="convolution";
    if(core && ((Arg(node,6).kind==FxArgument::Kind::kBool && Arg(node,6).boolean) || Pair(Arg(node,7),0,true)!=std::make_pair(0u,0u)))
      throw std::invalid_argument("transposed convolution is outside the DNN contract");
    unsigned groups=Int(Arg(node,core?8:6),1),channels=in.logical[3];
    bool depthwise=groups==channels && out_c==channels && weight_c==1;
    if(!depthwise && (groups!=1 || weight_c!=channels))throw std::invalid_argument("unsupported grouped convolution");
    long ph=(long(in.logical[1])+2*pad.first-dilation.first*(r-1)-1)/stride.first+1;
    long pw=(long(in.logical[2])+2*pad.second-dilation.second*(s-1)-1)/stride.second+1;
    if(ph<=0 || pw<=0)throw std::invalid_argument("empty convolution output");
    unsigned output=ImageBuffer(node.name,ph,pw,out_c);Halo(a,pad.first,pad.second);consumed.insert(a);
    ConvDesc c;c.n=options.batch;c.h=in.logical[1];c.w=in.logical[2];c.c=channels;c.k=out_c;c.r=r;c.s=s;
    c.p=ph;c.q=pw;c.pad_h=pad.first;c.pad_w=pad.second;c.stride_h=stride.first;c.stride_w=stride.second;
    c.dilation_h=dilation.first;c.dilation_w=dilation.second;c.input_layout=a;c.output_layout=output;
    unsigned cp=depthwise?8:in.physical[3];
    auto b=Weight(Fqn(weight)+".krsc",llvm::json::Object{{"kind","conv_krsc"},{"source",Fqn(weight)},{"padded_channels",cp}},
        Checked(std::uint64_t(out_c)*r*s*cp));
    unsigned conv=p.convolutions.size();p.convolutions.push_back(c);
    if(depthwise) {
      auto stage=Stage(PlanTaskKind::kDepthwiseConv,node,output,{a,b,output});
      auto& t=p.stages[stage];t.width=options.depthwise_channels;t.group=options.depthwise_rows;
      t.extent=out_c;t.conv=conv;t.rows_per_batch=ph*pw;
    }else {
      PlanGemm g;g.a=a;g.b=b;g.c=g.d=output;g.n=out_c;g.k=Checked(std::uint64_t(r)*s*channels);
      g.access.a=DmAAccess::kIm2Col;g.access.conv=conv;g.access.rows_per_batch=ph*pw;
      g.access.write.layout=output;unsigned gemm=p.gemms.size();p.gemms.push_back(g);
      auto stage=Stage(PlanTaskKind::kGemm,node,output);p.stages[stage].gemm=gemm;
    }
    if(Arg(node,2).kind!=FxArgument::Kind::kNone) {
      DmEpilogueOp bias;bias.parameter[0]=Bias(Ref(node,2));Append(output,bias);
    }
  }
  void BatchNorm(FxNodeRecord const& node) {
    RequireExclusive(Ref(node,0));
    auto input=Value(Ref(node,0));auto stage=p.stages.at(owner.at(input));
    if(stage.kind!=PlanTaskKind::kGemm && stage.kind!=PlanTaskKind::kDepthwiseConv)
      throw std::invalid_argument("batch norm requires a convolution anchor");
    bool core=Op(node)=="_native_batch_norm_legit_no_training";
    if(!core && (Arg(node,5).kind!=FxArgument::Kind::kBool || Arg(node,5).boolean))
      throw std::invalid_argument("DNN batch norm must be inference-only");
    unsigned conv=stage.kind==PlanTaskKind::kGemm?p.gemms[stage.gemm].access.conv:stage.conv;
    if(conv==missing)throw std::invalid_argument("batch norm anchor is not a convolution");
    auto const& c=p.convolutions[conv];
    auto& weight=p.buffers[stage.kind==PlanTaskKind::kGemm?p.gemms[stage.gemm].b:stage.operands[1]];
    auto parsed=llvm::json::parse(weight.pack_json);auto const* recipe=parsed?parsed->getAsObject():nullptr;
    if(!recipe || recipe->getString("kind")!="conv_krsc")throw std::invalid_argument("batch norm weight already folded");
    auto original=recipe->getString("source")->str();auto cp=recipe->getInteger("padded_channels").value();
    double epsilon=Real(Arg(node,core?6:7),0);
    llvm::json::Object common{{"gamma",Fqn(Ref(node,1))},{"beta",Fqn(Ref(node,2))},
        {"mean",Fqn(Ref(node,3))},{"variance",Fqn(Ref(node,4))},{"epsilon",epsilon}};
    auto& chain=Chain(input);
    if(chain.count>1 || (chain.count && chain.operations[0].kind!=DmEpilogueKind::kBias))
      throw std::invalid_argument("batch norm must immediately follow convolution/bias");
    if(chain.count) {
      auto const& old=p.buffers[chain.operations[0].parameter[0]];
      auto prior=llvm::json::parse(old.pack_json);common["bias"]=prior->getAsObject()->getString("source")->str();
    }
    auto folded=llvm::json::Object{{"kind","conv_bn_fold"},{"source",original},{"padded_channels",cp},
        {"gamma",Fqn(Ref(node,1))},{"mean",Fqn(Ref(node,3))},{"variance",Fqn(Ref(node,4))},{"epsilon",epsilon}};
    weight.pack_json=Json(std::move(folded));common["kind"]="bn_bias";
    unsigned bias=Weight(node.name+".bn_bias",std::move(common),c.k,"f32");
    chain.count=1;chain.operations[0]=DmEpilogueOp{};chain.operations[0].parameter[0]=bias;
    p.node_buffer[node.name]=input;
  }
  void Activation(FxNodeRecord const& node) {
    RequireExclusive(Ref(node,0));
    auto input=Value(Ref(node,0));DmEpilogueOp step;step.kind=DmEpilogueKind::kActivation;
    auto kind=Op(node);
    if(kind=="relu" || kind=="relu_")step.activation=DmActivation::kRelu;
    else if(kind=="hardtanh" || kind=="hardtanh_") {
      if(Real(Arg(node,1),-1)!=0 || Real(Arg(node,2),1)!=6)
        throw std::invalid_argument("DNN hardtanh requires the ReLU6 range");
      step.activation=DmActivation::kRelu6;
    }else if(kind=="tanh")step.activation=DmActivation::kTanh;
    else if(kind=="silu")step.activation=DmActivation::kSilu;
    else {
      auto approximation=node.kwargs.find("approximate");
      auto text=approximation==node.kwargs.end()?Arg(node,1).text:approximation->second.text;
      if(!text.empty() && text!="none" && text!="tanh")throw std::invalid_argument("unknown GELU approximation");
      step.activation=text=="tanh"?DmActivation::kGeluTanh:DmActivation::kGeluErf;
    }
    Append(input,step);p.node_buffer[node.name]=input;
  }
  void Residual(FxNodeRecord const& node) {
    unsigned a=Value(Ref(node,0)),b=Value(Ref(node,1));
    if(Real(Arg(node,2),1)!=1)throw std::invalid_argument("DNN residual alpha must be one");
    auto mutable_anchor=[&](unsigned id) {return owner.count(id) && !consumed.count(id) &&
        (p.stages[owner.at(id)].kind==PlanTaskKind::kGemm || p.stages[owner.at(id)].kind==PlanTaskKind::kDepthwiseConv);};
    if(!mutable_anchor(a) || (mutable_anchor(b) && owner.at(b)>owner.at(a)))std::swap(a,b);
    if(!mutable_anchor(a) || p.stages[owner.at(a)].kind!=PlanTaskKind::kGemm)
      throw std::invalid_argument("DNN residual needs a mutable GEMM epilogue anchor");
    RequireExclusive(Ref(node,Value(Ref(node,0))==a?0:1));
    auto const& x=p.buffers[a].layout;auto const& y=p.buffers[b].layout;
    if(x.rank!=y.rank || !std::equal(x.logical,x.logical+x.rank,y.logical))
      throw std::invalid_argument("DNN residual shapes disagree");
    DmEpilogueOp step;step.kind=DmEpilogueKind::kResidual;step.parameter[0]=b;step.residual_map.layout=b;
    Append(a,step);consumed.insert(b);p.node_buffer[node.name]=a;
  }
  void Pool(FxNodeRecord const& node) {
    auto input=Value(Ref(node,0));auto in=p.buffers[input].layout;
    if(in.kind!=DmLayout::kNHWC)throw std::invalid_argument("DNN max pool requires an image");
    auto kernel=Pair(Arg(node,1),1),stride=Pair(Arg(node,2),kernel.first),pad=Pair(Arg(node,3),0,true);
    auto dilation=Pair(Arg(node,4),1);
    if(Arg(node,5).kind==FxArgument::Kind::kBool && Arg(node,5).boolean)
      throw std::invalid_argument("DNN max pool ceil_mode is unsupported");
    ConvDesc c;c.n=options.batch;c.h=in.logical[1];c.w=in.logical[2];c.c=c.k=in.logical[3];
    c.r=kernel.first;c.s=kernel.second;c.stride_h=stride.first;c.stride_w=stride.second;
    c.pad_h=pad.first;c.pad_w=pad.second;c.dilation_h=dilation.first;c.dilation_w=dilation.second;
    long ph=(long(c.h)+2*c.pad_h-c.dilation_h*(c.r-1)-1)/c.stride_h+1;
    long pw=(long(c.w)+2*c.pad_w-c.dilation_w*(c.s-1)-1)/c.stride_w+1;
    if(ph<=0 || pw<=0)throw std::invalid_argument("empty pooling output");
    c.p=ph;c.q=pw;c.input_layout=input;
    auto output=ImageBuffer(node.name,ph,pw,c.c);c.output_layout=output;
    auto conv=p.convolutions.size();p.convolutions.push_back(c);
    Halo(input,pad.first,pad.second);p.buffers[input].layout.fill=DmFill::kNegativeInfinity;
    auto stage=Stage(PlanTaskKind::kPool,node,output,{input,output});auto& t=p.stages[stage];
    t.conv=conv;t.extent=c.c;t.width=64;t.group=16;t.rows_per_batch=ph*pw;consumed.insert(input);
  }
  void GlobalPool(FxNodeRecord const& node) {
    auto input=Value(Ref(node,0));auto const layout=p.buffers[input].layout;
    if(layout.kind!=DmLayout::kNHWC || !owner.count(input))throw std::invalid_argument("global pool requires an image producer");
    if(Op(node)=="mean") {
      auto const& dims=Arg(node,1);
      if(dims.kind!=FxArgument::Kind::kList || dims.items.size()!=2)
        throw std::invalid_argument("DNN mean is not spatial global pooling");
      std::set<long> axes;for(auto const& dim:dims.items)axes.insert((Int(dim,0)+4)%4);
      if(axes!=std::set<long>{2,3})throw std::invalid_argument("DNN mean is not spatial global pooling");
    }else if(Pair(Arg(node,1),1)!=std::make_pair(1u,1u))
      throw std::invalid_argument("adaptive pool must reduce to one pixel");
    auto source_stage=owner.at(input);unsigned area=layout.logical[1]*layout.logical[2],channels=layout.logical[3];
    auto const& producer=p.stages[source_stage];bool dw=producer.kind==PlanTaskKind::kDepthwiseConv;
    if(!dw && producer.kind!=PlanTaskKind::kGemm)throw std::invalid_argument("global pool has no partial-sum producer");
    unsigned tile=dw?producer.group*layout.logical[2]:128;
    unsigned parts=dw?(area+tile-1)/tile:(Checked(std::uint64_t(options.batch)*area)+tile-1)/tile;
    PlanBuffer partial;partial.name=node.name+".partials";partial.dtype="f32";partial.per_batch=Checked(std::uint64_t(parts)*channels);
    partial.layout.rank=3;unsigned shape[]={options.batch,parts,channels};
    std::copy(shape,shape+3,partial.layout.logical);std::copy(shape,shape+3,partial.layout.physical);Strides(partial.layout);
    auto sums=Add(std::move(partial));
    if(dw)p.stages[source_stage].operands[3]=sums;
    else {
      auto& chain=p.gemms[p.stages[source_stage].gemm].chain;
      if(chain.side_count==5)throw std::invalid_argument("too many GEMM side outputs");
      chain.side[chain.side_count++]={DmSideOutputKind::kChannelPartialSums,sums};
    }
    auto output=RowsBuffer(node.name+".fp32",1,channels,"f32");
    auto rounded=RowsBuffer(node.name,1,channels);
    auto stage=Stage(PlanTaskKind::kGlobalPoolReduce,node,rounded,{sums,output,rounded});
    auto& t=p.stages[stage];t.width=128;t.extent=channels;t.group=tile;t.rows_per_batch=area;
    t.partial_rows_per_image=dw?parts:0;consumed.insert(input);
  }
  void Linear(FxNodeRecord const& node) {
    bool addmm=Op(node)=="addmm";auto weight=Ref(node,addmm?2:1);
    auto input=Value(Ref(node,addmm?1:0));auto const& shape=Parameter(weight).shape;
    if(shape.size()!=2)throw std::invalid_argument("linear weight must be a matrix");
    unsigned n=U(shape[0]),k=U(shape[1]);
    if(p.buffers[input].per_batch%k)throw std::invalid_argument("linear input rows do not divide K");
    unsigned rows=p.buffers[input].per_batch/k;
    auto output=RowsBuffer(node.name,rows,n);
    auto b=Weight(Fqn(weight)+".linear",llvm::json::Object{{"kind","alias"},{"source",Fqn(weight)}},Checked(std::uint64_t(n)*k));
    PlanGemm g;g.a=input;g.b=b;g.c=g.d=output;g.n=n;g.k=k;g.access.rows_per_batch=rows;g.access.write.layout=output;
    unsigned gemm=p.gemms.size();p.gemms.push_back(g);
    auto stage=Stage(PlanTaskKind::kGemm,node,output);p.stages[stage].gemm=gemm;consumed.insert(input);
    unsigned bias_arg=addmm?0:2;
    if(Arg(node,bias_arg).kind!=FxArgument::Kind::kNone) {
      DmEpilogueOp bias;bias.parameter[0]=Bias(Ref(node,bias_arg));Append(output,bias);
    }
    if(addmm && (Real(Arg(node,3),1)!=1 || Real(Arg(node,4),1)!=1))
      throw std::invalid_argument("addmm alpha/beta must both be one");
  }
  void Visit(FxNodeRecord const& node) {
    if(node.op!="call_function")return;
    auto kind=Op(node);
    if(!node.has_arguments)throw std::invalid_argument("DNN import requires structured FX arguments: "+node.name);
    if(kind=="sym_size" || (!node.inputs.empty() && std::all_of(node.inputs.begin(),node.inputs.end(),
        [&](auto const& name){return shape_values.count(name); }))) {
      shape_values.insert(node.name);return;
    }
    if(kind=="conv2d" || kind=="convolution")Conv(node);
    else if(kind=="batch_norm" || kind=="_native_batch_norm_legit_no_training")BatchNorm(node);
    else if(kind=="relu" || kind=="relu_" || kind=="hardtanh" || kind=="hardtanh_" ||
            kind=="gelu" || kind=="tanh" || kind=="silu")Activation(node);
    else if(kind=="add" || kind=="add_")Residual(node);
    else if(kind=="max_pool2d" || kind=="max_pool2d_with_indices")Pool(node);
    else if(kind=="mean" || kind=="adaptive_avg_pool2d")GlobalPool(node);
    else if(kind=="linear" || kind=="addmm")Linear(node);
    else if(kind=="getitem") {
      auto source=Ref(node,0);
      if(Int(Arg(node,1),-1)!=0) {
        if(users[node.name])throw std::invalid_argument("DNN tuple auxiliary output is observable: "+node.name);
        return;
      }
      Alias(node,source);
    }else if(kind=="view" || kind=="reshape" || kind=="flatten" || kind=="clone" ||
             kind=="contiguous" || kind=="alias" || kind=="permute" || kind=="transpose" ||
             kind=="unsqueeze" || kind=="squeeze" || kind=="expand")Alias(node,Ref(node,0));
    else if(kind=="dropout") {
      if(Arg(node,2).kind==FxArgument::Kind::kBool && Arg(node,2).boolean)
        throw std::invalid_argument("DNN dropout must be in evaluation mode");
      Alias(node,Ref(node,0));
    }else throw std::invalid_argument("unsupported DNN target: "+node.target+" at "+node.name);
  }
};
} // namespace
ModelPlan BuildDnnModelPlan(std::vector<FxNodeRecord> const& nodes,
    std::vector<SignatureInput> const& inputs,std::vector<std::string> const& outputs,
    DnnPlanOptions const& options) {
  Builder builder(nodes,inputs,outputs,options);
  for(auto const& node:nodes)builder.Visit(node);
  for(auto const& name:outputs) {
    auto id=builder.Value(name);auto& buffer=builder.p.buffers[id];
    buffer.role="external";buffer.external_name=name;builder.p.outputs.push_back({id,{}});
  }
  ValidateDmModelPlan(builder.p);return std::move(builder.p);
}
} // namespace tilemega::frontend
