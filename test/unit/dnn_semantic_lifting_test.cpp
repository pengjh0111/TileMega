// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/DependencyTable.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <cassert>
#include <iostream>
#include <set>

namespace tilemega::tests::dnn_semantic_lifting_test {
namespace {
using namespace analysis;
using namespace frontend;
using Pair=std::pair<std::vector<long>,std::vector<long>>;
using Points=std::set<Pair>;
codegen::DmBufferLayout Image(unsigned batch,unsigned h,unsigned w,unsigned c,unsigned cp,unsigned pad) {
  codegen::DmBufferLayout l;l.kind=codegen::DmLayout::kNHWC;l.rank=4;
  l.logical[0]=l.physical[0]=batch;l.logical[1]=h;l.logical[2]=w;l.logical[3]=c;
  l.physical[1]=h+2*pad;l.physical[2]=w+2*pad;l.physical[3]=cp;
  l.strides[3]=1;l.strides[2]=cp;l.strides[1]=l.physical[2]*cp;l.strides[0]=l.physical[1]*l.strides[1];
  l.halo_top=l.halo_bottom=l.halo_left=l.halo_right=pad;return l;
}
ModelPlan Fixture(unsigned batch) {
  ModelPlan p;p.dm=p.forward=true;p.dtype="bf16";p.serving_seq=1;
  p.buffers.resize(8);
  for(unsigned i=0;i<8;++i)p.buffers[i].name="buffer"+std::to_string(i);
  p.buffers[1].layout=Image(batch,7,11,3,8,1);
  p.buffers[3].layout=Image(batch,4,6,19,24,0);
  codegen::ConvDesc c;c.n=batch;c.h=7;c.w=11;c.c=3;c.k=19;c.r=c.s=3;
  c.pad_h=c.pad_w=1;c.stride_h=c.stride_w=2;c.p=4;c.q=6;c.input_layout=1;c.output_layout=3;
  p.convolutions={c};
  PlanGemm g;g.a=1;g.b=2;g.c=g.d=3;g.n=19;g.k=27;
  g.access.a=codegen::DmAAccess::kIm2Col;g.access.conv=0;
  g.access.rows_per_batch=24;g.access.write.layout=3;p.gemms={g};
  PlanStage convert;convert.kind=PlanTaskKind::kLayoutConvert;
  convert.width=3;convert.group=17;convert.rows_per_batch=77;
  convert.operands.fill(codegen::kDmNoIndex);convert.operands[0]=0;convert.operands[1]=1;
  PlanStage conv;conv.kind=PlanTaskKind::kGemm;conv.gemm=0;
  PlanStage norm;norm.kind=PlanTaskKind::kLayerNorm;norm.width=19;norm.group=4;
  norm.rows_per_batch=24;norm.norm_epsilon=1e-6;norm.operands.fill(codegen::kDmNoIndex);
  norm.operands[0]=3;norm.operands[1]=4;norm.operands[2]=5;norm.operands[3]=6;norm.operands[4]=7;
  p.stages={convert,conv,norm};return p;
}
std::vector<long> Coords(OperatorNode const& node,std::vector<long> const& full) {
  assert(node.output.axes.size()==full.size());std::vector<long> out;
  for(unsigned i=0;i<full.size();++i)if(node.IsTiled(i))out.push_back(full[i]);
  return out;
}
Points Listed(CouplingRelation const& relation) {
  auto points=relation.Points();return {points.begin(),points.end()};
}
CouplingRelation Relation(Points const& points,unsigned left,unsigned right) {
  auto tuple=[](std::vector<long> const& p) {
    std::string s="[";for(auto x:p){if(s.size()>1)s+=",";s+=std::to_string(x);}return s+"]";
  };
  std::string text="{ ";
  for(auto const& [a,b]:points)text+=tuple(a)+" -> "+tuple(b)+"; ";
  if(points.empty())text+=tuple(std::vector<long>(left))+" -> "+tuple(std::vector<long>(right))+" : false";
  return CouplingRelation::FromIslText(text+" }");
}
void Equal(CouplingRelation const& actual,Points const& points) {
  auto oracle=Relation(points,actual.DomainDimNames().size(),actual.RangeDimNames().size());
  assert(Contains(actual,oracle) && Contains(oracle,actual));assert(Listed(actual)==points);
}
}
int TestDnnSemanticLifting(int,char**) {
  IslContext isl;unsigned cases=0;
  LiftOptions options;options.forward=true;options.batch_symbol="B";options.static_seq=1;
  for(unsigned batch:{1,2,3})for(int split:{1,5}) {
    ParamBinding known;known.Bind("B",batch);
    auto p=Fixture(batch);auto model=LiftSemantics(p,options);
    assert(model.has_plan && model.degraded.empty() && model.sem.ops.size()==3);
    auto text=model.sem.Serialize();
    for(auto const& op:model.sem.ops)assert(EncodeSemanticOp(DecodeSemanticOp(EncodeSemanticOp(op)))==EncodeSemanticOp(op));
    auto partition=LaunchGranularity(model,p,{{16,16,16,3,split}});
    auto graph=Instantiate(model.sem,partition);
    assert(model.sem.Serialize()==text);
    auto const& writer=*graph.Find("dnn.s0");auto const& reader=*graph.Find("dnn.s1");
    auto const& access=*reader.element_access;auto const& operand=access.semantic.operands.front();
    auto reads=ProjectTaskRead(access.semantic,reader,access.partition,operand.tensor,operand.map,{},known);
    Points expected;
    for(unsigned row=0;row<batch*24;++row)for(unsigned n=0;n<19;++n)
      for(unsigned r=0;r<3;++r)for(unsigned s=0;s<3;++s)for(unsigned c=0;c<3;++c) {
        std::vector<long> task{long(row/16),long(n/16)};
        if(split>1)task.push_back(((r*3+s)*8+c)/16);
        expected.insert({Coords(reader,task),{long(row/24),long((row%24)/6*2+r),long(row%6*2+s),long(c)}});
      }
    Equal(reads,expected);
    auto const& wa=*writer.element_access;
    auto writes=ProjectTaskWrite(wa.semantic,writer,wa.partition,wa.semantic.result,wa.semantic.result_map,{},known);
    Points expected_edges;
    for(auto const& [task,element]:expected) {
      auto image=element[0],y=element[1]-1,x=element[2]-1;
      if(y<0 || y>=7 || x<0 || x>=11)continue;
      expected_edges.insert({task,Coords(writer,{(image*77+y*11+x)/17,0})});
    }
    auto coupling=DeriveExactTaskCoupling(writes,reads,reader,known);
    Equal(coupling.relation,expected_edges);
    auto table=BuildDependencyTable(coupling.relation,writer,reader,known);
    assert(Contains(table.encoded_relation,table.linear_relation) && Contains(table.linear_relation,table.encoded_relation));
    auto const& normalized=model.sem.ops.back();
    assert(normalized.arithmetic=="layernorm" && normalized.additional_writes.size()==2);
    assert(model.written[1] && model.written[3] && model.written[6] && model.written[7]);
    assert(Instantiate(model.sem,LaunchGranularity(model,p,{{32,32,16,3,1}})).nodes.size()==3);
    ++cases;
  }
  ModelPlan bert;bert.dm=bert.forward=true;bert.dtype="bf16";bert.serving_seq=128;
  bert.buffers.resize(7);for(unsigned i=0;i<7;++i)bert.buffers[i].name="bert"+std::to_string(i);
  for(unsigned i=2;i<5;++i) {auto& l=bert.buffers[i].layout;l.rank=2;l.logical[0]=i==2?257:i==3?2:128;l.logical[1]=768;}
  PlanStage embed;embed.kind=PlanTaskKind::kEmbeddingSum;embed.width=768;embed.extent=257;
  for(unsigned i=0;i<7;++i)embed.operands[i]=i;
  bert.stages={embed};options.static_seq=128;
  auto embedding=LiftSemantics(bert,options);auto const& op=embedding.sem.ops.front();
  assert(op.arithmetic=="embedding_sum" && op.Dim("table")->extent.IsLiteral(3));
  for(unsigned i=0;i<2;++i) {
    auto const& index=op.operands[2+i].map.results.front();
    assert(index.kind==IndexResult::Kind::kDataDependent && index.binding_source==bert.buffers[i].name);
    assert(index.request_dims==std::vector<std::string>{"m"});
  }
  assert(op.operands[4].map.results.front().kind==IndexResult::Kind::kAffine);
  auto graph=Instantiate(embedding.sem,LaunchGranularity(embedding,bert,{}));
  ParamBinding known;known.Bind("B",2);
  assert(graph.nodes.front().Count().Eval(known,{})==256);
  auto const& access=*graph.nodes.front().element_access;
  auto const& table=access.semantic.operands[2];
  auto requests=ProjectTaskRequests(access.semantic,graph.nodes.front(),access.partition,
      table.tensor,table.map,{},known);
  assert(requests.Card().SumDomain().Eval({})==256*768);
  std::cout<<"DNN_LIFT cases="<<cases<<" exact_halo_split_ownership_and_binding_requests PASS\n";
  return 0;
}
} // namespace tilemega::tests::dnn_semantic_lifting_test
