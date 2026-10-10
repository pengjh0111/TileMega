// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/MoeRegionPattern.h>
#include <tilemega/Analysis/ClosedForm.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <functional>

namespace tilemega::frontend {
namespace {
using K=FxArgument::Kind;
struct Reader {
  std::map<std::string,FxNodeRecord const*> nodes;
  std::map<std::string,std::string> parameters;
  std::set<std::string> covered;
  std::string input;
  [[noreturn]] static void Fail(std::string const& why) {
    throw std::invalid_argument("MoE region: "+why);
  }
  FxNodeRecord const& Node(std::string const& name) {
    auto found=nodes.find(name);if(found==nodes.end())Fail("unknown value "+name);
    covered.insert(name);return *found->second;
  }
  FxArgument Arg(FxNodeRecord const& n,unsigned i,char const* key=nullptr,
                 FxArgument fallback={}) const {
    if(i<n.args.size() && key && n.kwargs.count(key))Fail("argument supplied twice at "+n.name);
    if(i<n.args.size())return n.args[i];
    if(key)if(auto it=n.kwargs.find(key);it!=n.kwargs.end())return it->second;
    return fallback;
  }
  FxNodeRecord const& Ref(FxNodeRecord const& n,unsigned i) {
    auto a=Arg(n,i);if(a.kind!=K::kNode)Fail(n.name+" lacks tensor operand");return Node(a.text);
  }
  static long Int(FxArgument const& a) {
    if(a.kind!=K::kInt)Fail("integer argument required");return a.integer;
  }
  static bool Bool(FxArgument const& a,bool fallback) {
    if(a.kind==K::kNone)return fallback;
    if(a.kind!=K::kBool)Fail("boolean argument required");return a.boolean;
  }
  static bool Target(FxNodeRecord const& n,std::initializer_list<char const*> names) {
    for(auto name:names)if(n.target==name)return true;return false;
  }
  void UnitAlpha(FxNodeRecord const& n) const {
    auto a=Arg(n,2,"alpha");
    if(a.kind!=K::kNone && !((a.kind==K::kInt && a.integer==1) ||
        (a.kind==K::kFloat && a.real==1)))Fail("addition has non-unit alpha at "+n.name);
  }
  FxNodeRecord const& Rows(FxNodeRecord const& n) {
    if(!Target(n,{"aten.reshape.default","aten.view.default"}))return n;
    auto const& source=Ref(n,0);
    if(n.shape.size()<2 || n.shape.size()>3 || source.shape.size()<2 || source.shape.size()>3 ||
        n.shape.back()!=source.shape.back())Fail("row reshape changes geometry");
    auto elements=[](FxNodeRecord const& value) {
      auto result=analysis::ClosedForm::Constant(1);
      for(auto const& dim:value.shape)result=result*analysis::ClosedForm::Parse(dim);
      return result.FactorStrings();
    };
    if(elements(n)!=elements(source))Fail("row reshape changes element count");
    return Rows(source);
  }
  FxNodeRecord const& Cast(FxNodeRecord const& n,char const* to,char const* from) {
    if(!Target(n,{"aten.to.dtype","aten._to_copy.default"}) || n.dtype!=to)
      Fail("missing dtype boundary at "+n.name);
    auto dtype=Arg(n,1,"dtype");
    if(dtype.kind!=K::kDtype || dtype.text!=to)Fail("dtype metadata disagrees with cast");
    auto const& source=Ref(n,0);
    if(source.dtype!=from || source.shape!=n.shape)Fail("cast changes shape or source dtype");
    return source;
  }
  void LastAxis(FxNodeRecord const& n,unsigned argument,bool keepdim=false) {
    auto a=Arg(n,argument,"dim");
    if(a.kind==K::kList) {
      if(a.items.size()!=1)Fail("reduction must have one axis");a=a.items.front();
    }
    auto axis=Int(a);auto const& source=Ref(n,0);
    if(axis!=-1 && axis!=long(source.shape.size())-1)
      Fail("reduction is not along the row's last axis");
    if(keepdim && !Bool(Arg(n,argument+1,"keepdim"),false))Fail("reduction must retain the last axis");
  }
  unsigned Extent(FxNodeRecord const& n,unsigned i) {
    if(i>=n.shape.size())Fail("missing tensor geometry");auto const& s=n.shape[i];
    if(s.empty() || !std::all_of(s.begin(),s.end(),[](unsigned char c){return c>='0' && c<='9';}))
      Fail("parameter extent is not static");
    auto x=std::stoull(s);if(!x || x>std::numeric_limits<unsigned>::max())Fail("invalid tensor extent");
    return unsigned(x);
  }
  void Shape(FxNodeRecord const& n,std::vector<unsigned> const& shape) {
    if(n.dtype!="torch.bfloat16" || n.shape.size()!=shape.size())Fail("weight rank or dtype mismatch");
    for(unsigned i=0;i<shape.size();++i)if(Extent(n,i)!=shape[i])Fail("expert/router dimensions disagree");
    if(!parameters.count(n.name))Fail("weight must be a parameter");
  }
  FxNodeRecord const& Tuple(FxNodeRecord const& n,long field) {
    if(n.target!="<built-in function getitem>" || Int(Arg(n,1))!=field)
      Fail("top-k tuple selector differs from its use");
    auto const& topk=Ref(n,0);if(topk.target!="aten.topk.default")Fail("tuple is not top-k");return topk;
  }
};
} // namespace

static MoeRegionMatch Match(std::vector<FxNodeRecord> const& nodes,
    std::vector<SignatureInput> const& signature,std::vector<std::string> const& outputs,
    bool decoder) {
  Reader r;
  for(auto const& n:nodes)if(!r.nodes.emplace(n.name,&n).second)Reader::Fail("duplicate FX value");
  for(auto const& s:signature) {
    if(s.kind=="PARAMETER")r.parameters[s.name]=s.target;
    if(s.kind=="USER_INPUT") {
      if(!r.input.empty())Reader::Fail("region requires one hidden-state input");r.input=s.name;
    }
  }
  if(r.input.empty() || outputs.size()!=1)Reader::Fail("region requires one input and output");
  FxNodeRecord const* experts=nullptr;
  for(auto const& n:nodes)if(n.target=="tilemega.moe_experts.default") {
    if(experts)Reader::Fail("ambiguous expert boundary");experts=&r.Node(n.name);
  }
  if(!experts || experts->args.size()!=5)Reader::Fail("missing expert custom-op boundary");
  MoeRegionMatch result;result.input=r.input;result.experts=experts->name;
  auto const& h=r.Node(r.input);
  if(h.dtype!="torch.bfloat16" || (h.shape.size()!=2 && !(decoder && h.shape.size()==3)))
    Reader::Fail("hidden input must be BF16 token rows");
  result.hidden=r.Extent(h,h.shape.size()-1);
  auto const& gu=r.Ref(*experts,3);auto const& down=r.Ref(*experts,4);
  result.expert_count=r.Extent(gu,0);result.intermediate=r.Extent(down,2);
  if(result.expert_count>128 || result.intermediate%16)Reader::Fail("unsupported expert capacity or gate-pair width");
  r.Shape(gu,{result.expert_count,2*result.intermediate,result.hidden});
  r.Shape(down,{result.expert_count,result.hidden,result.intermediate});
  result.gate_up_weight=gu.name;result.down_weight=down.name;
  auto const& normalized=r.Rows(r.Ref(*experts,0));result.normalized=normalized.name;

  auto const& indices=r.Ref(*experts,1);auto const& topk=r.Tuple(indices,1);
  result.indices=indices.name;result.topk=topk.name;
  result.top_k=r.Int(r.Arg(topk,1,"k"));
  if(!result.top_k || result.top_k>32 || result.top_k>result.expert_count)
    Reader::Fail("invalid top-k count");
  auto dimension=r.Arg(topk,2,"dim");
  if(dimension.kind!=K::kNone && r.Int(dimension)!=-1 && r.Int(dimension)!=1)
    Reader::Fail("top-k dimension differs from experts");
  if(!r.Bool(r.Arg(topk,3,"largest"),true) || !r.Bool(r.Arg(topk,4,"sorted"),true))
    Reader::Fail("top-k must select sorted largest values");
  auto const& weights=r.Ref(*experts,2);result.weights=weights.name;
  auto const& division=r.Cast(weights,"torch.bfloat16","torch.float32");
  if(division.target!="aten.div.Tensor")Reader::Fail("selected probabilities must be normalized");
  auto const& selected=r.Ref(division,0);auto const& sum=r.Ref(division,1);
  if(r.Tuple(selected,0).name!=topk.name || sum.target!="aten.sum.dim_IntList" ||
      r.Ref(sum,0).name!=selected.name)Reader::Fail("normalization does not reduce the selected probabilities");
  r.LastAxis(sum,1,true);
  auto const& probability=r.Ref(topk,0);
  if(!Reader::Target(probability,{"aten.softmax.int","aten._softmax.default"}) ||
      probability.dtype!="torch.float32")Reader::Fail("router requires FP32 softmax");
  r.LastAxis(probability,1);
  auto const* logits=&r.Ref(probability,0);
  if(probability.target=="aten._softmax.default") {
    if(r.Bool(r.Arg(probability,2,"half_to_float"),false))Reader::Fail("unexpected softmax dtype conversion");
    logits=&r.Cast(*logits,"torch.float32","torch.bfloat16");
  }else {
    auto dtype=r.Arg(probability,2,"dtype");
    if(dtype.kind!=K::kDtype || dtype.text!="torch.float32")Reader::Fail("softmax dtype is not explicit FP32");
  }
  if(logits->dtype!="torch.bfloat16")Reader::Fail("router logits must round to BF16");
  result.router=logits->name;
  if(!Reader::Target(*logits,{"aten.linear.default","aten.mm.default"}) ||
      r.Rows(r.Ref(*logits,0)).name!=normalized.name)Reader::Fail("router and experts must share normalized rows");
  auto const* router_weight=&r.Ref(*logits,1);
  if(logits->target=="aten.mm.default") {
    auto const& transpose=*router_weight;
    if(transpose.target!="aten.permute.default")Reader::Fail("router mm requires transposed weight");
    auto axes=r.Arg(transpose,1);
    if(axes.kind!=K::kList || axes.items.size()!=2 || r.Int(axes.items[0])!=1 || r.Int(axes.items[1])!=0)
      Reader::Fail("router weight permutation differs from a transpose");
    router_weight=&r.Ref(transpose,0);
  }else if(r.Arg(*logits,2,"bias").kind!=K::kNone)Reader::Fail("router must be bias-free");
  r.Shape(*router_weight,{result.expert_count,result.hidden});result.router_weight=router_weight->name;

  if(normalized.target!="aten.mul.Tensor")Reader::Fail("missing weighted RMSNorm");
  auto const* gamma=&r.Ref(normalized,0);auto const* scaled=&r.Ref(normalized,1);
  if(!r.parameters.count(gamma->name))std::swap(gamma,scaled);
  r.Shape(*gamma,{result.hidden});result.norm_weight=gamma->name;
  auto const& product=r.Cast(*scaled,"torch.bfloat16","torch.float32");
  if(product.target!="aten.mul.Tensor")Reader::Fail("missing RMS scaling product");
  auto const* cast_input=&r.Ref(product,0);auto const* rsqrt=&r.Ref(product,1);
  if(rsqrt->target!="aten.rsqrt.default")std::swap(cast_input,rsqrt);
  if(rsqrt->target!="aten.rsqrt.default" ||
      r.Cast(*cast_input,"torch.float32","torch.bfloat16").name!=h.name)
    Reader::Fail("RMSNorm does not normalize the region input");
  auto const& add=r.Ref(*rsqrt,0);
  if(add.target!="aten.add.Tensor")Reader::Fail("RMSNorm epsilon must follow variance");
  r.UnitAlpha(add);
  auto epsilon=r.Arg(add,1);
  result.epsilon=epsilon.kind==K::kFloat?epsilon.real:epsilon.kind==K::kInt?epsilon.integer:0;
  if(!(result.epsilon>0) || !std::isfinite(result.epsilon))Reader::Fail("invalid RMSNorm epsilon");
  auto const& mean=r.Ref(add,0);
  if(mean.target!="aten.mean.dim" || mean.dtype!="torch.float32")Reader::Fail("RMSNorm needs FP32 square mean");
  r.LastAxis(mean,1,true);auto const& square=r.Ref(mean,0);
  if(!Reader::Target(square,{"aten.square.default","aten.pow.Tensor_Scalar"}) ||
      (square.target=="aten.pow.Tensor_Scalar" && r.Int(r.Arg(square,1))!=2) ||
      r.Ref(square,0).name!=cast_input->name)Reader::Fail("RMSNorm variance is not the input square");

  auto const& output=r.Node(outputs.front());result.output=output.name;
  auto expected_rows=analysis::ClosedForm::Constant(1);
  for(unsigned axis=0;axis+1<h.shape.size();++axis)
    expected_rows=expected_rows*analysis::ClosedForm::Parse(h.shape[axis]);
  if(output.target!="aten.add.Tensor" || output.shape!=h.shape || experts->shape.size()!=2 ||
      experts->shape.back()!=h.shape.back() ||
      analysis::ClosedForm::Parse(experts->shape.front()).FactorStrings()!=expected_rows.FactorStrings())
    Reader::Fail("output must be the hidden-state residual sum");
  r.UnitAlpha(output);
  auto const* residual=&r.Rows(r.Ref(output,0));auto const* branch=&r.Rows(r.Ref(output,1));
  if(residual->name!=h.name)std::swap(residual,branch);
  if(residual->name!=h.name || branch->name!=experts->name)Reader::Fail("residual source differs from region input");
  for(auto const& n:nodes)if(n.op=="call_function" && !r.covered.count(n.name) &&
      !Reader::Target(n,{"aten._assert_tensor_metadata.default","aten.sym_size.int"}))
    Reader::Fail("uncovered target "+n.target+" at "+n.name);
  return result;
}

MoeRegionMatch MatchMoeRegion(std::vector<FxNodeRecord> const& nodes,
    std::vector<SignatureInput> const& signature,std::vector<std::string> const& outputs) {
  return Match(nodes,signature,outputs,false);
}

std::vector<MoeRegionMatch> FindDecoderMoeBlocks(std::vector<FxNodeRecord> const& nodes,
    std::vector<SignatureInput> const& signature) {
  std::map<std::string,FxNodeRecord const*> by_name;
  for(auto const& n:nodes)if(!by_name.emplace(n.name,&n).second)Reader::Fail("duplicate FX value");
  auto value=[&](std::string name) {
    while(Reader::Target(*by_name.at(name),{"aten.reshape.default","aten.view.default"})) {
      auto const& n=*by_name.at(name);
      if(n.args.empty() || n.args[0].kind!=K::kNode)Reader::Fail("reshape lacks its source");
      name=n.args[0].text;
    }
    return name;
  };
  std::vector<MoeRegionMatch> result;
  std::set<std::string> matched;
  for(auto const& output:nodes) {
    if(output.target!="aten.add.Tensor" || output.inputs.size()<2)continue;
    for(unsigned side=0;side<2;++side) {
      auto expert=value(output.inputs[side]);
      if(by_name.at(expert)->target!="tilemega.moe_experts.default")continue;
      auto input=output.inputs[1-side];
      if(!matched.insert(expert).second)Reader::Fail("expert boundary has multiple residual outputs");
      std::set<std::string> selected;
      std::function<void(std::string const&)> visit=[&](std::string const& name) {
        if(!selected.insert(name).second || name==input)return;
        auto const& node=*by_name.at(name);
        if(node.target=="aten.sym_size.int")return;
        for(auto const& source:node.inputs)visit(source);
      };
      visit(output.name);
      std::vector<FxNodeRecord> region;
      for(auto const& node:nodes)if(selected.count(node.name)) {
        region.push_back(node);
        // A region boundary declares an existing tensor value as its input.
        // Its producing attention/residual graph remains in the decoder plan.
        if(node.name==input) {
          auto& boundary=region.back();boundary.op="placeholder";boundary.inputs.clear();
          boundary.args.clear();boundary.kwargs.clear();
        }
      }
      std::vector<SignatureInput> inputs;
      for(auto const& item:signature)if(item.kind=="PARAMETER" && selected.count(item.name))inputs.push_back(item);
      inputs.push_back({input,"USER_INPUT"});
      result.push_back(Match(region,inputs,{output.name},true));
    }
  }
  for(auto const& node:nodes)if(node.target=="tilemega.moe_experts.default" && !matched.count(node.name))
    Reader::Fail("expert boundary lacks a proved residual region at "+node.name);
  return result;
}
} // namespace tilemega::frontend
