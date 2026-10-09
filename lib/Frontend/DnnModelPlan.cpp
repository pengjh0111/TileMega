// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DnnModelPlan.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Base64.h>
#include <llvm/Support/Error.h>
#include <algorithm>
#include <limits>
#include <map>
#include <cmath>
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
  std::set<std::string> collapsed;
  struct Embeddings {std::string word,type,position,ids,types;};
  std::map<std::string,Embeddings> embeddings;
  struct Attention {std::array<std::string,3> projections;std::string output,mask;unsigned heads=0;};
  std::map<std::string,Attention> packed_attention;
  std::map<std::string,std::string> attention_anchor;
  std::map<std::string,unsigned> packed_buffer;
  struct RowView {unsigned stride=1,offset=0,rows=0;};
  std::map<std::string,RowView> row_views;
  struct ImageNorm {std::string input,gamma,beta;double epsilon;};
  std::map<std::string,ImageNorm> image_norms;
  std::map<std::string,std::string> image_gates;
  std::map<std::string,std::pair<unsigned,unsigned>> scaled_residuals,scaled_activations;
  std::map<std::string,std::pair<std::string,unsigned>> image_shuffles;
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
    for(auto const& x:nodes)if(x.op=="call_function" && !x.args.empty() &&
        x.args[0].kind==FxArgument::Kind::kNode && parameters.count(x.args[0].text)) {
      auto op=Op(x);
      if(op=="permute" || op=="transpose" || op=="view" || op=="expand" || op=="clone") {
        parameters[x.name]=parameters.at(x.args[0].text);parameter_root[x.name]=parameter_root.at(x.args[0].text);
      }
    }
    for(auto const& x:inputs)if(x.kind=="USER_INPUT")Input(*records.at(x.name));
    if(p.serving_seq!=1)PrepareEncoder();
    else {PrepareImageNorms();PrepareImageGates();PrepareImageShuffles();}
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
    if(row_views.count(source))row_views[n.name]=row_views.at(source);
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
  unsigned ChannelScale(std::string const& ref,unsigned channels) {
    auto const& shape=Parameter(ref).shape;
    if(shape!=std::vector<std::string>{"1",std::to_string(channels),"1","1"})
      throw std::invalid_argument("image residual scale must broadcast over channels");
    return Weight(Fqn(ref)+".channel_f32",llvm::json::Object{
        {"kind","linear_bias"},{"source",Fqn(ref)},{"channel_axis",1}},channels,"f32");
  }
  unsigned Table(std::string const& ref) {
    auto const& node=Parameter(ref);std::uint64_t elements=1;
    for(auto const& dim:node.shape)elements*=U(dim);
    auto id=Weight(Fqn(ref)+".table",llvm::json::Object{{"kind","alias"},{"source",Fqn(ref)}},Checked(elements));
    if(node.shape.size()==2) {
      auto& l=p.buffers[id].layout;l.rank=2;
      for(unsigned axis=0;axis<2;++axis)l.logical[axis]=l.physical[axis]=U(node.shape[axis]);
      Strides(l);
    }
    return id;
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
    if(n.dtype=="torch.int64" && n.shape.size()==2) {
      auto seq=U(n.shape[1]);
      if((seq!=128 && seq!=384 && seq!=512) || (p.serving_seq!=1 && unsigned(p.serving_seq)!=seq))
        throw std::invalid_argument("encoder inputs require a common supported sequence length");
      p.serving_seq=seq;
      auto id=RowsBuffer(n.name,seq,1,"i64");auto& b=p.buffers[id];
      b.role="external";b.external_name=n.name;p.node_buffer[n.name]=id;return;
    }
    if(n.dtype!="torch.bfloat16" || n.shape.size()!=4)
      throw std::invalid_argument("DNN input needs the image or encoder structural entry: "+n.name);
    auto c=U(n.shape[1]),h=U(n.shape[2]),w=U(n.shape[3]);
    PlanBuffer b;b.name=n.name;b.layout=Image(h,w,c,true);b.per_batch=Checked(b.layout.strides[0]);
    b.role="external";b.external_name=n.name;unsigned external=Add(std::move(b));
    unsigned internal=ImageBuffer(n.name+".nhwc",h,w,c);
    auto stage=Stage(PlanTaskKind::kLayoutConvert,n,internal,{external,internal});
    p.stages[stage].width=c;p.stages[stage].group=128;p.stages[stage].rows_per_batch=h*w;
  }
  void PositionSequence(std::string ref,unsigned sequence) {
    auto const& slice=*records.at(ref);
    if(Op(slice)!="slice" || Int(Arg(slice,1),-1)!=1 || Int(Arg(slice,2),0)!=0 ||
       Int(Arg(slice,3),0)!=sequence || Int(Arg(slice,4),1)!=1)
      throw std::invalid_argument("position embedding requires a proven initial position slice");
    auto const& value=records.at(Ref(slice,0))->immutable_buffer_value;
    std::vector<char> bytes;
    auto error=llvm::decodeBase64(value.data_base64,bytes);
    if(error) {llvm::consumeError(std::move(error));throw std::invalid_argument("invalid frozen position buffer");}
    if(!value.present || value.dtype!="torch.int64" || value.byte_order!="little" ||
       value.shape.size()!=2 || value.shape[0]!=1 || value.shape[1]<sequence ||
       bytes.size()!=std::uint64_t(value.shape[1])*8)
      throw std::invalid_argument("position buffer has no exact immutable value proof");
    for(unsigned i=0;i<sequence;++i) {
      std::uint64_t number=0;
      for(unsigned byte=0;byte<8;++byte)number|=std::uint64_t(static_cast<unsigned char>(bytes[8*i+byte]))<<(8*byte);
      if(number!=i)throw std::invalid_argument("position buffer is not the initial token sequence");
    }
    collapsed.insert(slice.name);
  }
  void PrepareImageNorms() {
    auto at=[&](FxNodeRecord const& n,unsigned index)->FxNodeRecord const* {
      auto const& a=Arg(n,index);auto found=records.find(a.text);
      return a.kind==FxArgument::Kind::kNode && found!=records.end()?found->second:nullptr;
    };
    auto channel_mean=[](FxNodeRecord const& n) {
      auto const& dims=Arg(n,1);auto const& keep=Arg(n,2);
      return Op(n)=="mean" && dims.kind==FxArgument::Kind::kList && dims.items.size()==1 &&
          dims.items[0].kind==FxArgument::Kind::kInt && dims.items[0].integer==1 &&
          keep.kind==FxArgument::Kind::kBool && keep.boolean;
    };
    for(auto const& output:nodes) {
      if(Op(output)!="add" || output.args.size()<2)continue;
      auto* affine=at(output,0);auto* beta=at(output,1);
      if(affine && beta && parameters.count(affine->name))std::swap(affine,beta);
      if(!affine || !beta || !parameters.count(beta->name) || Op(*affine)!="mul")continue;
      auto* gamma=at(*affine,0);auto* div=at(*affine,1);
      if(gamma && div && parameters.count(div->name))std::swap(gamma,div);
      if(!gamma || !div || !parameters.count(gamma->name) || Op(*div)!="div")continue;
      auto* centered=at(*div,0);auto* root=at(*div,1);
      if(!centered || !root || Op(*centered)!="sub" || Op(*root)!="sqrt")continue;
      auto* input=at(*centered,0);auto* mean=at(*centered,1);auto* epsilon=at(*root,0);
      if(!input || !mean || !epsilon || !channel_mean(*mean) || Op(*epsilon)!="add")continue;
      auto* variance=at(*epsilon,0);auto* square=variance?at(*variance,0):nullptr;
      auto* variance_center=square?at(*square,0):nullptr;
      if(!variance || !square || !variance_center || !channel_mean(*variance) || Op(*square)!="pow" ||
          Real(Arg(*square,1),0)!=2 || Op(*variance_center)!="sub")continue;
      if(Ref(*mean,0)!=input->name || Ref(*variance_center,0)!=input->name ||
         Ref(*variance_center,1)!=mean->name || input->shape.size()!=4 ||
         output.shape!=input->shape || input->dtype!="torch.bfloat16" ||
         Real(Arg(output,2),1)!=1 || Real(Arg(*centered,2),1)!=1 ||
         Real(Arg(*variance_center,2),1)!=1 || Real(Arg(*epsilon,2),1)!=1)
        throw std::invalid_argument("image LayerNorm has incompatible reductions or affine operands");
      auto channels=U(input->shape[1]);double eps=Real(Arg(*epsilon,1),0);
      if(!(eps>0) || !std::isfinite(eps) || channels>4096)
        throw std::invalid_argument("image LayerNorm has invalid epsilon or channel width");
      for(auto const* parameter:{gamma,beta}) {
        auto const& shape=Parameter(parameter->name).shape;
        if(shape.size()!=1 || U(shape[0])!=channels || parameter->shape!=std::vector<std::string>{"1",std::to_string(channels),"1","1"})
          throw std::invalid_argument("image LayerNorm affine parameters must broadcast over channels");
      }
      std::set<std::string> internal={affine->name,div->name,centered->name,root->name,
          mean->name,epsilon->name,variance->name,square->name,variance_center->name};
      for(auto const& ref:internal) {
        if(externally_observed.count(ref))throw std::invalid_argument("image LayerNorm exposes an internal value");
        for(auto const& use:nodes)if(!internal.count(use.name) && use.name!=output.name &&
            std::find(use.inputs.begin(),use.inputs.end(),ref)!=use.inputs.end())
          throw std::invalid_argument("image LayerNorm internal value has an unmatched consumer");
      }
      collapsed.insert(internal.begin(),internal.end());
      image_norms[output.name]={input->name,gamma->name,beta->name,eps};
    }
  }
  void PrepareImageGates() {
    for(auto const& node:nodes) {
      if(Op(node)!="mul" || node.args.size()!=2 || collapsed.count(node.name) ||
         Arg(node,0).kind!=FxArgument::Kind::kNode || Arg(node,1).kind!=FxArgument::Kind::kNode)continue;
      auto const& a=*records.at(Ref(node,0));auto const& b=*records.at(Ref(node,1));
      if(Op(a)!="getitem" || Op(b)!="getitem" || Ref(a,0)!=Ref(b,0) ||
         std::set<long>{Int(Arg(a,1),-1),Int(Arg(b,1),-1)}!=std::set<long>{0,1})continue;
      auto const& split=*records.at(Ref(a,0));auto kind=Op(split);
      if(kind!="chunk" && kind!="split_with_sizes")continue;
      auto const& input=*records.at(Ref(split,0));
      if(input.shape.size()!=4 || node.shape.size()!=4 || Int(Arg(split,2),0)!=1 ||
         U(input.shape[1])!=2*U(node.shape[1]) || a.shape!=node.shape || b.shape!=node.shape)
        throw std::invalid_argument("SimpleGate requires two equal channel halves");
      auto channels=U(node.shape[1]);
      if(kind=="chunk" && Int(Arg(split,1),0)!=2)
        throw std::invalid_argument("SimpleGate chunk count must be two");
      if(kind=="split_with_sizes") {
        auto const& sizes=Arg(split,1);
        if(sizes.kind!=FxArgument::Kind::kList || sizes.items.size()!=2 ||
           Int(sizes.items[0],0)!=channels || Int(sizes.items[1],0)!=channels)
          throw std::invalid_argument("SimpleGate split sizes must be equal");
      }
      RequireExclusive(input.name);RequireExclusive(a.name);RequireExclusive(b.name);
      if(users[split.name]!=2 || externally_observed.count(split.name))
        throw std::invalid_argument("SimpleGate split has an unmatched consumer");
      collapsed.insert({split.name,a.name,b.name});image_gates[node.name]=input.name;
    }
  }
  void PrepareImageShuffles() {
    for(auto const& node:nodes) {
      if(Op(node)=="pixel_shuffle") {image_shuffles[node.name]={Ref(node,0),unsigned(Int(Arg(node,1),0))};continue;}
      if(Op(node)!="view" || node.shape.size()!=4 || Arg(node,0).kind!=FxArgument::Kind::kNode)continue;
      auto const& clone=*records.at(Ref(node,0));if(Op(clone)!="clone")continue;
      auto const& permute=*records.at(Ref(clone,0));if(Op(permute)!="permute")continue;
      auto const& view=*records.at(Ref(permute,0));if(Op(view)!="view" || view.shape.size()!=6)continue;
      auto const& axes=Arg(permute,1);
      std::vector<long> actual;if(axes.kind==FxArgument::Kind::kList)
        for(auto const& axis:axes.items)actual.push_back(Int(axis,-1));
      if(actual!=std::vector<long>{0,1,4,2,5,3})continue;
      auto factor=U(view.shape[2]);
      if(factor!=U(view.shape[3]) || U(node.shape[1])!=U(view.shape[1]) ||
         U(node.shape[2])!=factor*U(view.shape[4]) || U(node.shape[3])!=factor*U(view.shape[5]))
        throw std::invalid_argument("Core pixel shuffle shape differs from its permutation");
      for(auto const* internal:{&view,&permute,&clone})RequireExclusive(internal->name);
      collapsed.insert({view.name,permute.name,clone.name});
      image_shuffles[node.name]={Ref(view,0),factor};
    }
  }
  void ImageGate(FxNodeRecord const& node) {
    auto input=Value(image_gates.at(node.name));auto& stage=p.stages.at(owner.at(input));
    auto layout=p.buffers[input].layout;
    if(layout.kind!=DmLayout::kNHWC || layout.logical[3]%32)
      throw std::invalid_argument("SimpleGate requires complete 16-channel pairs");
    auto channels=layout.logical[3]/2;
    DmEpilogueOp gate;gate.kind=DmEpilogueKind::kGatePair;gate.gate=DmGatePair::kSimpleGate;
    gate.unit=16;gate.input_rounding=DmRounding::kBF16;
    if(stage.kind==PlanTaskKind::kGemm) {
      auto& gemm=p.gemms.at(stage.gemm);
      auto interleave=[&](unsigned id) {
        auto parsed=llvm::json::parse(p.buffers[id].pack_json);
        if(!parsed)throw std::invalid_argument("SimpleGate parameter has no packing recipe");
        p.buffers[id].pack_json=Json(llvm::json::Object{
            {"kind","gate_pair_interleave"},{"u",16},{"source",std::move(*parsed)}});
      };
      interleave(gemm.b);
      for(unsigned i=0;i<gemm.chain.count;++i) {
        if(gemm.chain.operations[i].kind!=DmEpilogueKind::kBias)
          throw std::invalid_argument("SimpleGate GEMM needs a direct bias/conv anchor");
        interleave(gemm.chain.operations[i].parameter[0]);
      }
    }else if(stage.kind==PlanTaskKind::kDepthwiseConv)stage.extent=channels;
    else throw std::invalid_argument("SimpleGate has no convolution producer");
    Append(input,gate);
    p.buffers[input].layout=Image(layout.logical[1],layout.logical[2],channels);
    p.buffers[input].per_batch=Checked(p.buffers[input].layout.strides[0]);
    p.node_buffer[node.name]=input;
  }
  void ImageShuffle(FxNodeRecord const& node) {
    auto const& pattern=image_shuffles.at(node.name);RequireExclusive(pattern.first);
    auto input=Value(pattern.first);auto const layout=p.buffers[input].layout;
    if(!owner.count(input) || consumed.count(input) || p.stages[owner.at(input)].kind!=PlanTaskKind::kGemm ||
       layout.kind!=DmLayout::kNHWC || !pattern.second ||
       layout.logical[3]%(std::uint64_t(pattern.second)*pattern.second))
      throw std::invalid_argument("pixel shuffle requires a mutable convolution output");
    auto& gemm=p.gemms[p.stages[owner.at(input)].gemm];
    if(gemm.access.write.kind!=DmWriteKind::kDense)
      throw std::invalid_argument("pixel shuffle producer already has a write permutation");
    auto factor=pattern.second;
    p.buffers[input].layout=Image(Checked(std::uint64_t(layout.logical[1])*factor),
        Checked(std::uint64_t(layout.logical[2])*factor),layout.logical[3]/(factor*factor));
    p.buffers[input].per_batch=Checked(p.buffers[input].layout.strides[0]);
    gemm.access.write.kind=DmWriteKind::kPixelShuffle;gemm.access.write.factor=factor;
    p.node_buffer[node.name]=input;
  }
  void ImageMultiply(FxNodeRecord const& node) {
    for(unsigned i=0;i<2;++i)if(parameters.count(Ref(node,i))) {
      auto ref=Ref(node,1-i);RequireExclusive(ref);RequireExclusive(node.name);
      auto input=Value(ref);auto const& layout=p.buffers[input].layout;
      if(layout.kind!=DmLayout::kNHWC || !owner.count(input) ||
         p.stages[owner.at(input)].kind!=PlanTaskKind::kGemm)
        throw std::invalid_argument("scaled residual requires an image GEMM producer");
      scaled_residuals[node.name]={input,ChannelScale(Ref(node,i),layout.logical[3])};
      p.node_buffer[node.name]=input;return;
    }
    auto a=Value(Ref(node,0)),scale=Value(Ref(node,1));
    if(p.buffers[a].layout.logical[1]==1 && p.buffers[a].layout.logical[2]==1)std::swap(a,scale);
    auto const image=p.buffers[a].layout;auto vector=p.buffers[scale].layout;
    if(image.kind!=DmLayout::kNHWC || vector.kind!=DmLayout::kNHWC || vector.rank!=4 ||
       vector.logical[1]!=1 || vector.logical[2]!=1 || image.logical[3]!=vector.logical[3] ||
       vector.physical[3]!=vector.logical[3] || vector.halo_top || vector.halo_bottom ||
       vector.halo_left || vector.halo_right || image.logical[0]!=vector.logical[0])
      throw std::invalid_argument("SCA multiplication requires one channel vector per image");
    RequireExclusive(node.name);
    // The 1x1 producer's physical allocation already is [batch,channels].
    // Give the access descriptor the same explicit rank-two scale layout.
    DmBufferLayout matrix;matrix.rank=2;matrix.logical[0]=matrix.physical[0]=options.batch;
    matrix.logical[1]=matrix.physical[1]=vector.logical[3];Strides(matrix);
    p.buffers[scale].layout=matrix;
    scaled_activations[node.name]={a,scale};p.node_buffer[node.name]=a;consumed.insert(scale);
  }
  void ImageLayerNorm(FxNodeRecord const& node) {
    auto const& pattern=image_norms.at(node.name);auto input=Value(pattern.input);
    auto const layout=p.buffers[input].layout;
    if(layout.kind!=DmLayout::kNHWC || layout.rank!=4)
      throw std::invalid_argument("image LayerNorm requires an NHWC producer");
    auto output=ImageBuffer(node.name,layout.logical[1],layout.logical[2],layout.logical[3]);
    auto stage=Stage(PlanTaskKind::kLayerNorm,node,output,
        {input,Table(pattern.gamma),Table(pattern.beta),output});
    auto& t=p.stages[stage];t.width=layout.logical[3];t.group=4;
    t.rows_per_batch=Checked(std::uint64_t(layout.logical[1])*layout.logical[2]);t.norm_epsilon=pattern.epsilon;
    consumed.insert(input);
  }
  void Iota(std::string ref,unsigned axis) const {
    auto const* n=records.at(ref);
    while(Op(*n)=="unsqueeze" || (Op(*n)=="add" && Real(Arg(*n,1),1)==0))n=records.at(Ref(*n,0));
    if(Op(*n)!="arange")throw std::invalid_argument("key-padding index is not an iota");
    unsigned length_arg=n->target=="aten.arange.start_step"?1:0;
    if(length_arg && (Int(Arg(*n,0),-1)!=0 || Int(Arg(*n,2),1)!=1))
      throw std::invalid_argument("key-padding iota has a nonidentity range");
    auto const& length=Arg(*n,length_arg);
    if(axis==1) {
      if(Int(length,-1)!=p.serving_seq)throw std::invalid_argument("key-padding iota differs from sequence");
    }else {
      if(length.kind!=FxArgument::Kind::kNode)throw std::invalid_argument("batch iota needs a symbolic input extent");
      auto const& size=*records.at(length.text);
      if(Op(size)!="sym_size" || Int(Arg(size,1),-1)!=0 || !p.node_buffer.count(Ref(size,0)))
        throw std::invalid_argument("batch iota is not the input batch extent");
    }
  }
  std::string KeyPadding(std::string const& ref) {
    auto const& node=*records.at(ref);auto kind=Op(node);
    if(node.op=="placeholder") {
      if(!p.node_buffer.count(ref) || node.dtype!="torch.int64" || node.shape.size()!=2 || U(node.shape[1])!=unsigned(p.serving_seq))
        throw std::invalid_argument("key-padding input must be int64 [batch,sequence]");
      return ref;
    }
    collapsed.insert(ref);
    if(kind=="expand") {
      if(node.shape.size()!=4 || U(node.shape[1])!=1 || U(node.shape[2])!=unsigned(p.serving_seq) || U(node.shape[3])!=unsigned(p.serving_seq))
        throw std::invalid_argument("key-padding expansion changes the query or key geometry");
      return KeyPadding(Ref(node,0));
    }
    if(kind=="__and__" || kind=="bitwise_and") {
      if(node.dtype!="torch.bool")throw std::invalid_argument("key-padding conjunction must be boolean");
      for(unsigned i=0;i<2;++i) {
        auto const& literal=records.at(Ref(node,i))->constant;
        if(literal.has_all_true && literal.all_true)return KeyPadding(Ref(node,1-i));
      }
      throw std::invalid_argument("key-padding conjunction has another value-dependent term");
    }
    if(kind=="to" || kind=="_to_copy") {
      auto dtype=node.kwargs.find("dtype");bool boolean=dtype!=node.kwargs.end() && dtype->second.text=="torch.bool";
      for(auto const& a:node.args)boolean|=a.kind==FxArgument::Kind::kDtype && a.text=="torch.bool";
      if(!boolean)throw std::invalid_argument("key-padding cast must preserve boolean semantics");
      return KeyPadding(Ref(node,0));
    }
    if(kind=="index") {
      auto const& indices=Arg(node,1);
      if(indices.kind!=FxArgument::Kind::kList || indices.items.size()!=2 ||
         indices.items[0].kind!=FxArgument::Kind::kNode || indices.items[1].kind!=FxArgument::Kind::kNode)
        throw std::invalid_argument("key-padding gather requires batch/key identity indices");
      auto const& image=*records.at(indices.items[0].text);auto const& key=*records.at(indices.items[1].text);
      if(image.shape.size()!=4 || key.shape.size()!=4 || image.shape[0]!=node.shape[0] ||
         image.shape[1]!="1" || image.shape[2]!="1" || image.shape[3]!="1" ||
         key.shape[0]!="1" || key.shape[1]!="1" || key.shape[2]!="1" || U(key.shape[3])!=unsigned(p.serving_seq))
        throw std::invalid_argument("key-padding identity indices have incompatible broadcast axes");
      Iota(image.name,0);Iota(key.name,1);return KeyPadding(Ref(node,0));
    }
    throw std::invalid_argument("key-padding graph has an unsupported value transform: "+node.target);
  }
  std::string Unwrap(std::string ref,std::set<std::string>* cluster=nullptr) const {
    while(true) {
      auto const& n=*records.at(ref);auto k=Op(n);
      if(k!="view" && k!="reshape" && k!="expand" && k!="clone" && k!="permute" && k!="transpose" && k!="_to_copy")return ref;
      if(cluster)cluster->insert(ref);ref=Ref(n,0);
    }
  }
  static bool Permutation(FxNodeRecord const& n,std::initializer_list<long> order) {
    auto const& axes=Arg(n,1);if(Op(n)!="permute" || axes.kind!=FxArgument::Kind::kList || axes.items.size()!=order.size())return false;
    unsigned i=0;for(auto axis:order)if(Int(axes.items[i++],-1)!=axis)return false;return true;
  }
  std::string LinearInput(FxNodeRecord const& n) const {return Ref(n,Op(n)=="addmm"?1:0);}
  std::string LinearWeight(FxNodeRecord const& n) const {return Ref(n,Op(n)=="addmm"?2:1);}
  std::string LinearBias(FxNodeRecord const& n) const {return Ref(n,Op(n)=="addmm"?0:2);}
  void RegisterAttention(Attention const& attention,std::set<std::string> const& cluster) {
    auto const& q=*records.at(attention.projections[0]);
    for(auto const& name:attention.projections)
      if(Unwrap(LinearInput(*records.at(name)))!=Unwrap(LinearInput(q)))
        throw std::invalid_argument("encoder QKV projections must share an input");
    for(auto const& n:nodes)for(auto const& ref:n.inputs)
      if(ref!=attention.output && cluster.count(ref) && !cluster.count(n.name))
        throw std::invalid_argument("packed QKV intermediate has an observable use");
    for(auto const& ref:cluster)if(ref!=attention.output) {
      if(externally_observed.count(ref))throw std::invalid_argument("packed QKV intermediate is an output");
      collapsed.insert(ref);
    }
    collapsed.erase(q.name);collapsed.erase(attention.output);
    packed_attention[q.name]=attention;attention_anchor[attention.output]=q.name;
  }
  void PrepareCoreAttention() {
    for(auto const& pv:nodes)if(Op(pv)=="bmm") {
      std::set<std::string> cluster{pv.name};
      auto probability=Unwrap(Ref(pv,0),&cluster);auto const& guarded=*records.at(probability);
      if(Op(guarded)!="where")continue;
      auto const& softmax=*records.at(Ref(guarded,2));
      if(Op(softmax)!="_softmax")continue;
      if(Int(Arg(softmax,1),0)!=-1 || (Arg(softmax,2).kind==FxArgument::Kind::kBool && Arg(softmax,2).boolean))
        throw std::invalid_argument("encoder softmax must normalize the key axis in FP32");
      auto const& fill=*records.at(Ref(guarded,1));auto const& invalid=*records.at(Ref(guarded,0));
      if(Op(fill)!="full_like" || Ref(fill,0)!=softmax.name || Real(Arg(fill,1),1)!=0 || Op(invalid)!="logical_not")
        throw std::invalid_argument("encoder softmax guard must replace only fully masked rows by zero");
      auto const& any=*records.at(Ref(invalid,0));
      if(Op(any)!="any" || Int(Arg(any,1),0)!=-1 || Arg(any,2).kind!=FxArgument::Kind::kBool || !Arg(any,2).boolean)
        throw std::invalid_argument("encoder softmax guard reduction differs from the key axis");
      auto const& valid=*records.at(Ref(any,0));
      if(Op(valid)!="logical_not")throw std::invalid_argument("encoder mask guard has no validity inversion");
      auto const& eq=*records.at(Ref(valid,0));auto const& score=*records.at(Ref(softmax,0));
      if(Op(eq)!="eq" || Ref(eq,0)!=score.name || Real(Arg(eq,1),0)!=-std::numeric_limits<double>::infinity() || Op(score)!="add")
        throw std::invalid_argument("encoder mask guard must test these attention scores for -inf");
      auto const& qk=*records.at(Unwrap(Ref(score,0),&cluster));auto const& mask=*records.at(Ref(score,1));
      if(Op(qk)!="bmm" || Op(mask)!="where")throw std::invalid_argument("encoder scores require QK plus an additive boolean mask");
      auto const& zero=*records.at(Ref(mask,1));auto const& negative=*records.at(Ref(mask,2));
      if(Op(zero)!="scalar_tensor" || Real(Arg(zero,0),1)!=0 || Op(negative)!="scalar_tensor" ||
         Real(Arg(negative,0),0)!=-std::numeric_limits<double>::infinity())
        throw std::invalid_argument("encoder additive mask must be zero or -inf");
      Attention a;
      auto const& condition=*records.at(Ref(mask,0));
      if(!condition.constant.has_all_true || !condition.constant.all_true)a.mask=KeyPadding(condition.name);
      if(externally_observed.count(mask.name))throw std::invalid_argument("encoder additive mask is observable");
      collapsed.insert(mask.name);
      for(auto const* n:{&guarded,&softmax,&fill,&invalid,&any,&valid,&eq,&score,&qk})cluster.insert(n->name);
      for(unsigned i=0;i<3;++i) {
        auto source=Ref(i<2?qk:pv,i<2?i:1);
        while(true) {
          auto const& layout=*records.at(source);auto kind=Op(layout);
          if(kind!="view" && kind!="expand" && kind!="clone")break;
          cluster.insert(layout.name);source=Ref(layout,0);
        }
        if(i<2) {
          auto const& scale=*records.at(source);
          if(Op(scale)!="mul" || std::abs(Real(Arg(scale,1),0)-std::sqrt(.125))>1e-15)
            throw std::invalid_argument("encoder Q/K scaling differs from sqrt(1/sqrt(64))");
          cluster.insert(scale.name);source=Ref(scale,0);
        }
        std::vector<std::vector<long>> permutations;
        while(true) {
          auto const& layout=*records.at(source);auto kind=Op(layout);
          if(kind!="view" && kind!="expand" && kind!="clone" && kind!="permute" && kind!="_to_copy")break;
          cluster.insert(layout.name);
          if(kind=="permute") {
            std::vector<long> order;for(auto const& axis:Arg(layout,1).items)order.push_back(Int(axis,-1));
            permutations.push_back(std::move(order));
          }
          if(kind=="_to_copy" && layout.dtype!="torch.float32")
            throw std::invalid_argument("Core encoder attention inputs must promote BF16 to FP32");
          source=Ref(layout,0);
        }
        auto expected=i==1?std::vector<std::vector<long>>{{0,1,3,2},{0,2,1,3}}:
            std::vector<std::vector<long>>{{0,2,1,3}};
        if(permutations!=expected)throw std::invalid_argument("Core encoder QKV head layout differs from the attention contract");
        auto const& projection=*records.at(source);
        if(Op(projection)!="addmm")throw std::invalid_argument("Core encoder QKV has no addmm anchor");
        a.projections[i]=projection.name;cluster.insert(projection.name);
      }
      auto const& q=*records.at(a.projections[0]);auto width=U(Parameter(LinearWeight(q)).shape.at(0));
      if(width%64)throw std::invalid_argument("encoder projection width does not contain complete heads");
      a.heads=width/64;
      // The SDPA decomposition returns head-major context and then restores
      // token-major rows. Follow its sole layout chain to that row-major view.
      std::string current=pv.name;std::vector<unsigned> axes{0,1,2,3};bool head_view=false;
      while(true) {
        std::vector<FxNodeRecord const*> next;
        for(auto const& n:nodes)if(std::find(n.inputs.begin(),n.inputs.end(),current)!=n.inputs.end())next.push_back(&n);
        if(next.size()!=1)throw std::invalid_argument("encoder context layout chain must have one consumer");
        auto const& n=*next[0];auto kind=Op(n);cluster.insert(n.name);
        if(kind=="view") {
          if(!head_view) {
            if(n.shape.size()!=4 || U(n.shape[1])!=a.heads || U(n.shape[2])!=unsigned(p.serving_seq) || U(n.shape[3])!=64)
              throw std::invalid_argument("encoder context head view has incompatible extents");
            head_view=true;
          }else {
            if(n.shape.size()!=3 || U(n.shape[1])!=unsigned(p.serving_seq) || U(n.shape[2])!=width ||
               axes!=std::vector<unsigned>{0,2,1,3})
              throw std::invalid_argument("encoder context does not restore token-major rows");
            a.output=n.name;break;
          }
        }else if(kind=="permute") {
          auto const& list=Arg(n,1);if(!head_view || list.kind!=FxArgument::Kind::kList || list.items.size()!=4)
            throw std::invalid_argument("encoder context permutation has no head geometry");
          std::vector<unsigned> order;std::set<unsigned> seen;
          for(auto const& axis:list.items) {auto v=Int(axis,-1);if(v<0 || v>=4 || !seen.insert(v).second)
              throw std::invalid_argument("encoder context permutation is invalid");order.push_back(axes[v]);}
          axes=std::move(order);
        }else if(kind!="clone" && kind!="_to_copy")throw std::invalid_argument("encoder context has a non-layout consumer");
        current=n.name;
      }
      RegisterAttention(a,cluster);
    }
  }
  void PrepareEncoder() {
    for(auto const& node:nodes)if(Op(node)=="add" && node.inputs.size()==2) {
      auto const& first=*records.at(Ref(node,0));auto const& position=*records.at(Ref(node,1));
      if(Op(first)!="add" || Op(position)!="embedding")continue;
      auto const& word=*records.at(Ref(first,0));auto const& type=*records.at(Ref(first,1));
      if(Op(word)!="embedding" || Op(type)!="embedding")continue;
      if(!p.node_buffer.count(Ref(word,1)) || !p.node_buffer.count(Ref(type,1)))continue;
      PositionSequence(Ref(position,1),p.serving_seq);
      for(auto const* inner:{&word,&type,&position,&first}) {
        RequireExclusive(inner->name);collapsed.insert(inner->name);
      }
      embeddings[node.name]={Ref(word,0),Ref(type,0),Ref(position,0),Ref(word,1),Ref(type,1)};
    }
    PrepareCoreAttention();
    for(auto const& node:nodes)if(Op(node)=="scaled_dot_product_attention") {
      Attention attention;attention.output=node.name;
      std::set<std::string> cluster{node.name};
      for(unsigned i=0;i<3;++i) {
        auto const& transpose=*records.at(Ref(node,i));
        if(Op(transpose)!="transpose" || Int(Arg(transpose,1),-1)!=1 || Int(Arg(transpose,2),-1)!=2)
          throw std::invalid_argument("encoder QKV requires the token/head transpose");
        auto const& view=*records.at(Ref(transpose,0));
        if(Op(view)!="view" || view.shape.size()!=4 || U(view.shape[1])!=unsigned(p.serving_seq) || U(view.shape[3])!=64)
          throw std::invalid_argument("encoder QKV requires 64-dimensional heads");
        auto heads=U(view.shape[2]);
        if(i && heads!=attention.heads)throw std::invalid_argument("encoder QKV head counts disagree");
        attention.heads=heads;
        auto const& linear=*records.at(Ref(view,0));
        if(Op(linear)!="linear")throw std::invalid_argument("encoder QKV has no linear anchor");
        attention.projections[i]=linear.name;cluster.insert(linear.name);cluster.insert(view.name);cluster.insert(transpose.name);
      }
      if(Real(Arg(node,4),0)!=0 || (Arg(node,5).kind==FxArgument::Kind::kBool && Arg(node,5).boolean))
        throw std::invalid_argument("encoder attention requires zero dropout and noncausal masking");
      auto scale=node.kwargs.find("scale");
      if(scale!=node.kwargs.end() && Real(scale->second,.125)!=.125)
        throw std::invalid_argument("encoder attention scale differs from 1/sqrt(64)");
      if(Arg(node,3).kind!=FxArgument::Kind::kNone) {
        auto const& mask=*records.at(Ref(node,3));
        if(!mask.constant.has_all_true || !mask.constant.all_true)
          attention.mask=KeyPadding(mask.name);
      }
      if(externally_observed.count(node.name))
        throw std::invalid_argument("head-major attention output needs an explicit output layout");
      for(auto const& consumer:nodes)if(std::find(consumer.inputs.begin(),consumer.inputs.end(),node.name)!=consumer.inputs.end())
        if(Op(consumer)!="transpose" || Int(Arg(consumer,1),-1)!=1 || Int(Arg(consumer,2),-1)!=2)
          throw std::invalid_argument("encoder context requires the head/token transpose");
      RegisterAttention(attention,cluster);
    }
    if(embeddings.size()!=1 || packed_attention.empty())
      throw std::invalid_argument("encoder graph requires embedding-sum and attention structural entries");
  }
  void Embedding(FxNodeRecord const& node) {
    auto const& e=embeddings.at(node.name);auto width=U(Parameter(e.word).shape.at(1));
    if(Parameter(e.type).shape.size()!=2 || Parameter(e.position).shape.size()!=2 ||
       U(Parameter(e.type).shape[1])!=width || U(Parameter(e.position).shape[1])!=width)
      throw std::invalid_argument("embedding tables disagree on hidden width");
    auto output=RowsBuffer(node.name,p.serving_seq,width);
    auto stage=Stage(PlanTaskKind::kEmbeddingSum,node,output,
        {Value(e.ids),Value(e.types),Table(e.word),Table(e.type),Table(e.position),output});
    auto& t=p.stages[stage];t.width=width;t.group=1;t.rows_per_batch=p.serving_seq;t.extent=U(Parameter(e.word).shape[0]);
  }
  void LayerNorm(FxNodeRecord const& node) {
    auto input=Value(Ref(node,0));auto const& normalized=Arg(node,1);
    if(normalized.kind!=FxArgument::Kind::kList || normalized.items.size()!=1)
      throw std::invalid_argument("encoder LayerNorm must reduce one channel axis");
    auto width=Int(normalized.items[0],0);
    if(width<=0 || width>4096 || p.buffers[input].per_batch%width)
      throw std::invalid_argument("LayerNorm input disagrees with normalized width");
    unsigned rows=p.buffers[input].per_batch/width;
    auto output=RowsBuffer(node.name,rows,width);
    auto stage=Stage(PlanTaskKind::kLayerNorm,node,output,{input,Table(Ref(node,2)),Table(Ref(node,3)),output});
    auto& t=p.stages[stage];t.width=width;t.group=4;t.rows_per_batch=rows;t.norm_epsilon=Real(Arg(node,4),0);
    consumed.insert(input);
  }
  void Qkv(FxNodeRecord const& node) {
    auto const& attention=packed_attention.at(node.name);auto input=Value(LinearInput(node));
    auto width=attention.heads*64,k=U(Parameter(LinearWeight(node)).shape.at(1));
    llvm::json::Array weights,biases;
    for(auto const& name:attention.projections) {
      auto const& linear=*records.at(name);auto const& shape=Parameter(LinearWeight(linear)).shape;
      if(Op(linear)=="addmm" && (Real(Arg(linear,3),1)!=1 || Real(Arg(linear,4),1)!=1))
        throw std::invalid_argument("packed QKV addmm alpha/beta must both be one");
      if(shape.size()!=2 || U(shape[0])!=width || U(shape[1])!=k)
        throw std::invalid_argument("packed QKV weights disagree on shape");
      weights.push_back(Fqn(LinearWeight(linear)));biases.push_back(Fqn(LinearBias(linear)));
    }
    auto b=Weight(node.name+".qkv",llvm::json::Object{{"kind","qkv_concat_bias"},{"sources",std::move(weights)},
        {"heads",attention.heads},{"head_dim",64},{"part","weight"}},Checked(std::uint64_t(width)*3*k));
    auto bias=Weight(node.name+".qkv_bias",llvm::json::Object{{"kind","qkv_concat_bias"},{"sources",std::move(biases)},
        {"heads",attention.heads},{"head_dim",64},{"part","bias"}},width*3,"f32");
    auto output=RowsBuffer(node.name+".packed",p.serving_seq,width*3);
    PlanGemm g;g.a=input;g.b=b;g.c=g.d=output;g.n=width*3;g.k=k;
    g.access.rows_per_batch=p.serving_seq;g.access.write.layout=output;
    auto gemm=p.gemms.size();p.gemms.push_back(g);auto stage=Stage(PlanTaskKind::kGemm,node,output);p.stages[stage].gemm=gemm;
    DmEpilogueOp step;step.parameter[0]=bias;Append(output,step);consumed.insert(input);packed_buffer[node.name]=output;
  }
  void AttentionStage(FxNodeRecord const& node) {
    auto const& anchor=attention_anchor.at(node.name);auto const& a=packed_attention.at(anchor);
    auto output=RowsBuffer(node.name,p.serving_seq,a.heads*64);
    auto stage=Stage(PlanTaskKind::kEncoderAttention,node,output,{packed_buffer.at(anchor),output,
        a.mask.empty()?missing:Value(a.mask)});
    auto& t=p.stages[stage];t.width=p.serving_seq;t.group=64;t.rows_per_batch=p.serving_seq;t.extent=a.heads;
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
    if(in.kind!=DmLayout::kNHWC || in.rank!=4)throw std::invalid_argument("convolution requires a rank-four image");
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
      auto scaled=scaled_activations.find(Ref(node,0));
      if(scaled!=scaled_activations.end()) {
        if(r!=1 || s!=1 || stride!=std::make_pair(1u,1u))
          throw std::invalid_argument("SCA scale producer requires pixel-preserving pointwise convolution");
        g.access.a_scale=scaled->second.second;
      }
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
        {"gamma",Fqn(Ref(node,1))},{"mean",Fqn(Ref(node,3))},{"variance",Fqn(Ref(node,4))},
        {"epsilon",epsilon},{"scale_in_epilogue",true}};
    weight.pack_json=Json(std::move(folded));common["kind"]="bn_bias";
    unsigned bias=Weight(node.name+".bn_bias",std::move(common),c.k,"f32");
    unsigned scale=Weight(node.name+".bn_scale",llvm::json::Object{{"kind","bn_scale"},
        {"gamma",Fqn(Ref(node,1))},{"mean",Fqn(Ref(node,3))},
        {"variance",Fqn(Ref(node,4))},{"epsilon",epsilon}},c.k,"f32");
    chain.count=2;chain.operations[0]=DmEpilogueOp{};
    chain.operations[0].kind=DmEpilogueKind::kScale;chain.operations[0].parameter[0]=scale;
    chain.operations[1]=DmEpilogueOp{};chain.operations[1].parameter[0]=bias;
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
    for(unsigned i=0;i<2;++i)if(parameters.count(Ref(node,i))) {
      auto input=Value(Ref(node,1-i));auto const& shape=Parameter(Ref(node,i)).shape;
      if(shape.size()!=1 || Real(Arg(node,2),1)!=1)
        throw std::invalid_argument("DNN parameter add requires a unit-scaled channel bias");
      RequireExclusive(Ref(node,1-i));DmEpilogueOp bias;bias.parameter[0]=Bias(Ref(node,i));
      Append(input,bias);p.node_buffer[node.name]=input;return;
    }
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
    auto source=Value(Ref(node,0))==a?Ref(node,0):Ref(node,1);
    if(auto scaled=scaled_residuals.find(source);scaled!=scaled_residuals.end())step.parameter[1]=scaled->second.second;
    auto const& gemm=p.gemms[p.stages[owner.at(a)].gemm];
    if(gemm.access.write.kind==DmWriteKind::kPixelShuffle) {
      step.residual_map.kind=DmWriteKind::kPixelShuffle;step.residual_map.factor=gemm.access.write.factor;
    }
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
    auto rounded=node.shape.size()==4 && channels%8==0?ImageBuffer(node.name,1,1,channels):
        RowsBuffer(node.name,1,channels);
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
    unsigned rows=row_views.count(Ref(node,addmm?1:0))?row_views.at(Ref(node,addmm?1:0)).rows:p.buffers[input].per_batch/k;
    auto output=RowsBuffer(node.name,rows,n);
    auto b=Weight(Fqn(weight)+".linear",llvm::json::Object{{"kind","alias"},{"source",Fqn(weight)}},Checked(std::uint64_t(n)*k));
    PlanGemm g;g.a=input;g.b=b;g.c=g.d=output;g.n=n;g.k=k;g.access.rows_per_batch=rows;g.access.write.layout=output;
    if(row_views.count(Ref(node,addmm?1:0))) {
      auto const& view=row_views.at(Ref(node,addmm?1:0));g.access.a_row_stride=view.stride;g.access.a_row_offset=view.offset;
    }
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
    if(collapsed.count(node.name))return;
    auto kind=Op(node);
    if(!node.has_arguments)throw std::invalid_argument("DNN import requires structured FX arguments: "+node.name);
    if(image_norms.count(node.name)) {ImageLayerNorm(node);return;}
    if(image_gates.count(node.name)) {ImageGate(node);return;}
    if(image_shuffles.count(node.name)) {ImageShuffle(node);return;}
    if(embeddings.count(node.name)) {Embedding(node);return;}
    if(packed_attention.count(node.name)) {Qkv(node);return;}
    if(attention_anchor.count(node.name)) {AttentionStage(node);return;}
    if(kind=="sym_size" || (!node.inputs.empty() && std::all_of(node.inputs.begin(),node.inputs.end(),
        [&](auto const& name){return shape_values.count(name); }))) {
      shape_values.insert(node.name);return;
    }
    if(p.serving_seq!=1 && node.constant.present && !externally_observed.count(node.name))return;
    if(p.serving_seq!=1 && kind=="_assert_tensor_metadata")return;
    if(kind=="conv2d" || kind=="convolution")Conv(node);
    else if(kind=="batch_norm" || kind=="_native_batch_norm_legit_no_training")BatchNorm(node);
    else if(kind=="relu" || kind=="relu_" || kind=="hardtanh" || kind=="hardtanh_" ||
            kind=="gelu" || kind=="tanh" || kind=="silu")Activation(node);
    else if(kind=="add" || kind=="add_")Residual(node);
    else if(kind=="mul" && p.serving_seq==1)ImageMultiply(node);
    else if(kind=="max_pool2d" || kind=="max_pool2d_with_indices")Pool(node);
    else if(kind=="mean" || kind=="adaptive_avg_pool2d")GlobalPool(node);
    else if(kind=="linear" || kind=="addmm" || (kind=="bmm" && parameters.count(Ref(node,1))))Linear(node);
    else if(kind=="layer_norm" || kind=="native_layer_norm")LayerNorm(node);
    else if(kind=="pad" || kind=="constant_pad_nd") {
      auto const& pad=Arg(node,1);
      if(pad.kind!=FxArgument::Kind::kList || pad.items.size()%2 || pad.items.size()>8 ||
         std::any_of(pad.items.begin(),pad.items.end(),[](auto const& item) {
           return item.kind!=FxArgument::Kind::kInt || item.integer!=0;}))
        throw std::invalid_argument("DNN layout padding must be identically zero");
      Alias(node,Ref(node,0));
    }
    else if(kind=="slice") {
      auto const& input=*records.at(Ref(node,0));long axis=Int(Arg(node,1),0);
      if(axis<0)axis+=input.shape.size();
      if(axis<0 || unsigned(axis)>=input.shape.size() || Int(Arg(node,2),0)!=0 ||
         Int(Arg(node,4),1)!=1 || node.shape!=input.shape)
        throw std::invalid_argument("DNN slice is not a range identity");
      auto extent=axis==0?options.batch:U(input.shape[axis]);
      if(Int(Arg(node,3),0)<extent)throw std::invalid_argument("DNN slice truncates its axis");
      Alias(node,Ref(node,0));
    }
    else if(kind=="select") {
      auto const& input=*records.at(Ref(node,0));
      if(p.serving_seq==1 || input.shape.size()!=3 || Int(Arg(node,1),-1)!=1 ||
         Int(Arg(node,2),-1)!=0 || U(input.shape[1])!=unsigned(p.serving_seq))
        throw std::invalid_argument("DNN row select requires the initial token of each sequence");
      Alias(node,Ref(node,0));row_views[node.name]={unsigned(p.serving_seq),0,1};
    }
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
    if(buffer.layout.kind==DmLayout::kNHWC && builder.owner.count(id) &&
       builder.p.stages[builder.owner.at(id)].kind==PlanTaskKind::kGemm) {
      auto const shape=buffer.layout;
      auto& gemm=builder.p.gemms[builder.p.stages[builder.owner.at(id)].gemm];
      if(builder.consumed.count(id) || gemm.access.write.kind!=DmWriteKind::kDense)
        throw std::invalid_argument("external NCHW output requires an unconsumed dense GEMM result");
      buffer.layout=builder.Image(shape.logical[1],shape.logical[2],shape.logical[3],true);
      buffer.per_batch=Checked(buffer.layout.strides[0]);gemm.access.write.kind=DmWriteKind::kNCHW;
    }
    buffer.role="external";buffer.external_name=name;builder.p.outputs.push_back({id,{}});
  }
  ValidateDmModelPlan(builder.p);return std::move(builder.p);
}
} // namespace tilemega::frontend
