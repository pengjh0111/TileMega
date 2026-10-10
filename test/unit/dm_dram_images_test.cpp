// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/DramFloor.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace tilemega::tests::dm_dram_images_test {
int TestDmDramImages(int,char**) {
  using namespace analysis;
  IslContext isl;
  auto f=[](long n){return ClosedForm::Constant(n);};
  auto image=[](std::string const& text){return CouplingRelation::FromIslText(text);};
  unsigned cases=0;
  for(int element_reads:{0,1})for(int side:{0,1})for(int weight_bytes:{2,4}) {
    SemanticOp op;op.name="scatter";op.exact_task_access=true;op.dtype=ScalarType::kBF16;
    op.domain={{"v",f(3)},{"row",f(4)},{"n",f(7)},
        {"k",f(5),f(0),IteratorType::kReduction}};
    op.task_space={"owners",{{"v",f(3)},{"row",f(4)},{"n",f(7)}}};
    op.task_map.results={IndexResult::Dim("v"),IndexResult::Dim("row"),IndexResult::Dim("n")};
    op.result={"out",{{"t",f(10)},{"rank",f(3)},{"n",f(7)}}};
    op.result_map.results={IndexResult::DataDependent("rows",{"v","row"}),
        IndexResult::DataDependent("rows",{"v","row"}),IndexResult::Dim("n")};
    TensorSpace hidden{"hidden",{{"t",f(10)},{"k",f(5)}}};
    TensorSpace weights{"weights",{{"n",f(7)},{"k",f(5)}}};
    op.operands={{{},hidden,{{IndexResult::DataDependent("rows",{"v","row"}),IndexResult::Dim("k")}},{}},
        {{},weights,{{IndexResult::Dim("n"),IndexResult::Dim("k")}}, {}}};
    if(element_reads)for(auto const& operand:op.operands)
      op.element_reads.push_back({operand.tensor,operand.map,{}});
    if(side) {
      TensorSpace stats{"stats",{{"t",f(10)},{"rank",f(3)}}};
      IndexingMap map;map.results={op.result_map.results[0],op.result_map.results[1]};
      op.additional_writes.push_back({stats,map,{}, {}});
    }
    DramFloorOptions options;options.dram_gbps=1000;options.tc_gflops=100000;
    options.element_bytes["weights"]=weight_bytes;options.element_bytes["stats"]=4;
    options.indirect_read_images["hidden"]=image("{ [] -> [t,k] : (t=3 or t=7) and 0<=k<5 }");
    options.indirect_write_images["out"]=image("{ [] -> [t,r,n] : ((t=2 and r=0) or (t=8 and r=2)) and 0<=n<7 }");
    if(side)options.indirect_write_images["stats"]=image("{ [] -> [t,r] : (t=2 and r=0) or (t=8 and r=2) }");
    auto floor=DeriveDramFloor({{op}},options);auto value=floor.Evaluate({});
    assert(value.read_bytes==2*5*2+7*5*weight_bytes);
    assert(value.write_bytes==2*7*2+(side?2*4:0));
    assert(floor.tensors.at("out").writes.ImageCard().Eval({})==14);
    assert(floor.tensors.at("hidden").no_producer.ImageCard().Eval({})==10);
    auto reject=[&](DramFloorOptions bad) {
      bool caught=false;try{(void)DeriveDramFloor({{op}},bad);}
      catch(std::invalid_argument const&){caught=true;}assert(caught);
    };
    auto wrong=options;wrong.indirect_write_images.erase("out");reject(wrong);
    wrong=options;wrong.indirect_read_images.clear();reject(wrong);
    wrong=options;wrong.indirect_read_images["hidden"]=image("{ [] -> [t,k] : t=10 and 0<=k<5 }");reject(wrong);
    wrong=options;wrong.indirect_write_images["out"]=image("{ [] -> [t,r,n] : t=10 and r=0 and 0<=n<7 }");reject(wrong);
    wrong=options;wrong.indirect_write_images["out"]=image("{ [q] -> [t,r,n] : q=0 and t=2 and r=0 and 0<=n<7 }");reject(wrong);
    if(side) {wrong=options;wrong.indirect_write_images.erase("stats");reject(wrong);}
    ++cases;
  }
  // A tensor's actual image may span several readers with disjoint affine columns.
  SemanticOp a;a.name="a";a.exact_task_access=true;a.dtype=ScalarType::kBF16;
  a.domain={{"m",f(2)},{"k",f(2)}};
  a.result={"a_out",{{"m",f(2)},{"k",f(2)}}};
  a.result_map.results={IndexResult::Dim("m"),IndexResult::Dim("k")};
  a.task_space=a.result;a.task_map=a.result_map;
  TensorSpace source{"shared",{{"t",f(8)},{"k",f(4)}}};
  a.operands={{{},source,{{IndexResult::DataDependent("rows",{"m"}),IndexResult::Dim("k")}}, {}}};
  auto b=a;b.name="b";b.result.name="b_out";b.task_space.name="b_owners";
  b.operands[0].map.results[1].offset=f(2);
  DramFloorOptions shared;shared.dram_gbps=1000;shared.tc_gflops=100000;
  shared.indirect_read_images["shared"]=image("{ [] -> [t,k] : t=3 and 0<=k<4 }");
  auto floor=DeriveDramFloor({{a,b}},shared);
  assert(floor.Evaluate({}).read_bytes==8 && floor.Evaluate({}).write_bytes==16);
  // The sampled mean counts distinct rows, not a fabricated set of addresses.
  DramFloorOptions expected;expected.dram_gbps=1000;expected.tc_gflops=100000;
  auto q=QuasiPolynomial::Constant(7).ScaleRational("1/3");
  assert(std::abs(q.EvalReal({})-7./3)<1e-12);
  bool nonintegral=false;try{(void)q.Eval({});}catch(std::runtime_error const&){nonintegral=true;}
  assert(nonintegral);
  expected.expected_indirect_reads["shared"]={q,"unit-test routing histogram"};
  floor=DeriveDramFloor({{a,b}},expected);
  assert(std::abs(floor.Evaluate({}).read_bytes-14./3)<1e-12);
  assert(floor.tensors.at("shared").reads.empty());
  assert(floor.tensors.at("shared").no_producer.empty());
  assert(floor.tensors.at("shared").expected_read_elements);
  auto reject_expected=[&](SemanticGraph const& graph,DramFloorOptions const& options) {
    bool caught=false;try{(void)DeriveDramFloor(graph,options);}
    catch(std::exception const&){caught=true;}assert(caught);
  };
  auto wrong=expected;wrong.expected_indirect_reads["shared"].elements=QuasiPolynomial::Constant(33);
  reject_expected({{a,b}},wrong);
  wrong=expected;wrong.expected_indirect_reads["shared"].elements=QuasiPolynomial::Constant(-1);
  reject_expected({{a,b}},wrong);
  wrong=expected;wrong.expected_indirect_reads["shared"].source.clear();reject_expected({{a,b}},wrong);
  wrong=expected;wrong.indirect_read_images=shared.indirect_read_images;reject_expected({{a,b}},wrong);
  wrong=expected;wrong.expected_indirect_reads["unused"]={q,"unused"};reject_expected({{a,b}},wrong);
  auto direct=b;direct.operands[0].map.results[0]=IndexResult::Dim("m");
  reject_expected({{a,direct}},expected);
  auto producer=b;producer.result=source;producer.result_map.results={IndexResult::Dim("m"),IndexResult::Dim("k")};
  producer.operands.clear();reject_expected({{producer,a}},expected);
  auto varying=QuasiPolynomial::FromIslText("[T] -> { T : T>0 }").ScaleRational("1/3");
  bool unbound=false;try{(void)varying.EvalReal({});}catch(std::out_of_range const&){unbound=true;}
  assert(unbound);
  ParamBinding bound;bound.Bind("T",7);assert(std::abs(varying.EvalReal(bound)-7./3)<1e-12);
  std::cout<<"DM physical images: "<<cases<<" typed gather/scatter cases, shared-column union and malformed witness rejection PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_dram_images_test
