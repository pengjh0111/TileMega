// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/DmConvReductionPartition.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <iostream>
#include <set>

namespace tilemega::tests::conv_reduction_partition_test {
namespace {
using namespace analysis;
ClosedForm F(long n) {return ClosedForm::Constant(n);}
SemanticOp Conv(unsigned c,unsigned r,unsigned s) {
  SemanticOp op;op.name="conv";op.kind=OperatorKind::kMatmul;
  op.dtype=ScalarType::kBF16;op.arithmetic="matmul";op.exact_task_access=true;
  auto rows=ClosedForm::Symbol("B")*F(5);
  op.domain={{"m",rows},{"n",F(7)},
      {"c",F(c),F(0),IteratorType::kReduction},
      {"r",F(r),F(0),IteratorType::kReduction},
      {"s",F(s),F(0),IteratorType::kReduction}};
  op.result={"output",{{"m",rows},{"n",F(7)}}};
  op.result_map.results={IndexResult::Dim("m"),IndexResult::Dim("n")};
  op.task_space=op.result;op.task_map=op.result_map;
  for(auto const& name:{"m","n"}) {
    TensorSpace input{name,{{name,name==std::string("m")?rows:F(7)},
        {"r",F(r)},{"s",F(s)},{"c",F(c)}}};
    op.operands.push_back({"",input,{{IndexResult::Dim(name),IndexResult::Dim("r"),
        IndexResult::Dim("s"),IndexResult::Dim("c")}}, {}});
  }
  op.reduction={"c","add","partial","conv.combine",true};return op;
}
template<class F> void Reject(F call,unsigned& rejected) {
  bool failed=false;try{call();}catch(std::invalid_argument const&){failed=true;}
  assert(failed);++rejected;
}
}
int TestConvReductionPartition(int,char**) {
  IslContext isl;
  unsigned cases=0,rejected=0;std::uint64_t checked=0;
  ParamBinding known;known.Bind("B",2);
  for(unsigned c:{3,24,27,64})for(unsigned r:{1,3})for(unsigned s:{2,3})
    for(unsigned tk:{16,32,64})for(unsigned split:{1,2,4}) {
    codegen::ConvDesc conv;conv.c=c;conv.r=r;conv.s=s;
    codegen::DmBufferLayout layout;layout.kind=codegen::DmLayout::kNHWC;layout.rank=4;
    layout.logical[3]=c;layout.physical[3]=c==3?4:(c+7)/8*8;
    if(layout.physical[3]<tk && tk%layout.physical[3]) {
      Granularity invalid;
      Reject([&]{solver::PartitionDmConvGemm(Conv(c,r,s),conv,layout,
          {16,16,int(tk),2,int(split)},invalid,known);},rejected);
      continue;
    }
    auto geometry=backend::ConvIterationGeometry::Build(conv,layout,tk);
    if(geometry.iterations%split)continue;
    auto op=Conv(c,r,s);auto original=EncodeSemanticOp(op);
    solver::GemmConfig config{16,16,int(tk),2,int(split)};
    Granularity g;solver::PartitionDmConvGemm(op,conv,layout,config,g,known);
    auto graph=Instantiate({{op}},g);assert(graph.nodes.size()==(split>1?2u:1u));
    auto const& task=graph.nodes.front();auto const& access=*task.element_access;
    assert(EncodeSemanticOp(op)==original);
    assert(DecodeSemanticOp(EncodeSemanticOp(access.semantic)).Serialize()==access.semantic.Serialize());
    auto chunk=geometry.iterations/split;
    for(auto const& input:op.operands) {
      auto actual=ProjectTaskRead(access.semantic,task,access.partition,
          input.tensor,input.map,{},known).Points();
      std::set<decltype(actual)::value_type> expected;
      for(long row=0;row<(input.tensor.name=="m"?10:7);++row)
        for(unsigned rr=0;rr<r;++rr)for(unsigned ss=0;ss<s;++ss)for(unsigned cc=0;cc<c;++cc) {
          // Reference ordering is nested filter positions then channel blocks.
          auto cp=layout.physical[3];
          auto iteration=cp>=tk?std::uint64_t(rr*s+ss)*((cp+tk-1)/tk)+cc/tk:
              std::uint64_t((rr*s+ss)*cp+cc)/tk;
          std::vector<long> owner{0,0};if(split>1)owner.push_back(iteration/chunk);
          expected.insert({owner,{row,long(rr),long(ss),long(cc)}});
        }
      assert(std::set<decltype(actual)::value_type>(actual.begin(),actual.end())==expected);
      checked+=actual.size();
    }
    auto writes=ProjectTaskElements(access.semantic,task,access.partition,
        access.semantic.result,access.semantic.result_map,{},known);
    assert(writes.Reverse().IsSingleValued());
    assert(writes.Points().size()==70*split);
    for(auto const& [owner,element]:writes.Points())
      if(split>1)assert(owner.back()==element.back());
    TaskWorkOptions options;options.reduction_tiles["c"]=F(tk);
    auto work=DeriveTaskWork(op,task,known,options);
    assert(work.parallel_extent.Eval(known)==70 && work.reduce_extent.Eval(known)==c*r*s);
    assert(work.task_reduce_extent.SumDomain().Eval(known)==c*r*s);
    assert(work.nominal_task_reduce_extent.SumDomain().Eval(known)==geometry.iterations*tk);
    assert(work.write_elements.SumDomain().Eval(known)==70*split);
    assert(work.nominal_write_elements.SumDomain().Eval(known)==256*split);
    if(split>1) {
      auto final=DeriveTaskWork(op,graph.nodes.back(),known);
      assert(final.write_elements.SumDomain().Eval(known)==70);
      assert(final.read_elements.SumDomain().Eval(known)==70*split);
    }
    auto incomplete=g;incomplete.reduction_index.at(op.name).capacity=F(0);
    Reject([&]{Instantiate({{op}},incomplete);},rejected);
    auto wrong=g;wrong.reduction_index.at(op.name).index=IndexResult::Dim("m");
    Reject([&]{Instantiate({{op}},wrong);},rejected);
    ++cases;
  }
  auto op=Conv(24,3,3);codegen::ConvDesc conv;conv.c=24;conv.r=conv.s=3;
  codegen::DmBufferLayout layout;layout.kind=codegen::DmLayout::kNHWC;layout.rank=4;
  layout.logical[3]=layout.physical[3]=24;
  Granularity g;solver::PartitionDmConvGemm(op,conv,layout,{16,16,16,2,2},g);
  assert(g.reduction_index.at(op.name).capacity.IsLiteral(18));
  assert(g.reduction_chunk.at(op.name).IsLiteral(9));
  auto clipped=g;clipped.reduction_index.at(op.name).capacity=F(17);
  Reject([&]{Instantiate({{op}},clipped);},rejected);
  auto zero=g.reduction_index.at(op.name);zero.index.outer_divisor=F(0);
  Reject([&]{Granularity{}.IndexReduction(op.name,zero);},rejected);
  Reject([&]{solver::PartitionDmConvGemm(op,conv,layout,{16,16,16,2,4},g);},rejected);
  Granularity unsplit;solver::PartitionDmConvGemm(op,conv,layout,{16,16,16,2,1},unsplit);
  (void)Instantiate({{op}},unsplit);
  auto split_graph=Instantiate({{op}},g);
  auto split_work=DeriveTaskWork(op,split_graph.nodes.front(),known);
  for(long chunk:{0,1}) {
    ParamBinding point;point.Bind("m",0).Bind("n",0).Bind("j",chunk);
    assert(split_work.task_reduce_extent.BindCoordinates(point).Eval(known)==(chunk?104:112));
  }
  // Length-prefixed keys distinguish (capacity,width)=(18,16) and (1,816).
  auto cache_collision=unsplit;cache_collision.reduction_index.at(op.name).capacity=F(1);
  cache_collision.reduction_index.at(op.name).issued_width=F(816);
  Reject([&]{Instantiate({{op}},cache_collision);},rejected);
  std::cout<<"CONV_REDUCTION cases="<<cases<<" rejected="<<rejected
      <<" enumerated_accesses="<<checked<<" PASS\n";
  return 0;
}
} // namespace tilemega::tests::conv_reduction_partition_test
