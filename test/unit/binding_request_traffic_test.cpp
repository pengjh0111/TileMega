// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/BindingRequestTraffic.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace tilemega::tests::binding_request_traffic_test {
int TestBindingRequestTraffic(int,char**) {
  using namespace tilemega::analysis;
  using tilemega::solver::DeriveBindingRequestTraffic;
  IslContext isl;
  auto f=[](long n){return ClosedForm::Constant(n);};
  unsigned checks=0;
  for(long capacity:{1,5,17})for(long tokens:{2,19})for(int weight_bytes:{2,4}) {
    SemanticOp op;op.name="expert";op.exact_task_access=true;op.dtype=ScalarType::kBF16;
    op.domain={{"v",f(capacity)},{"row",f(4)},{"n",f(7)},
        {"k",f(5),f(0),IteratorType::kReduction}};
    op.task_space={"virtual",{{"v",f(capacity)},{"row",f(4)},{"n",f(7)}}};
    op.task_map.results={IndexResult::Dim("v"),IndexResult::Dim("row"),IndexResult::Dim("n")};
    op.result={"partial",{{"t",f(tokens)},{"rank",f(3)},{"n",f(7)}}};
    op.result_map.results={IndexResult::DataDependent("rows",{"v","row"}),
        IndexResult::DataDependent("rows",{"v","row"}),IndexResult::Dim("n")};
    TensorSpace a{"hidden",{{"t",f(tokens)},{"k",f(5)}}};
    TensorSpace b{"experts",{{"e",f(2)},{"n",f(7)},{"k",f(5)}}};
    op.operands={{"input",a,{{IndexResult::DataDependent("rows",{"v","row"}),IndexResult::Dim("k")}},{}},
        {"",b,{{IndexResult::DataDependent("bindings",{"v"}),IndexResult::Dim("n"),IndexResult::Dim("k")}}, {}}};
    auto graph=Instantiate({{op}},Granularity{}.Tile("expert","v",f(1))
        .Tile("expert","row",f(3)).Tile("expert","n",f(4)));
    DramFloor floor;
    auto full_a=CouplingRelation::FromIslText("{ [] -> [t,k] : 0<=t<"+std::to_string(tokens)+" and 0<=k<5 }");
    auto full_out=CouplingRelation::FromIslText("{ [] -> [t,r,n] : 0<=t<"+std::to_string(tokens)+" and 0<=r<3 and 0<=n<7 }");
    floor.tensors["hidden"].element_bytes=2;floor.tensors["hidden"].writes=full_a;
    floor.tensors["experts"].element_bytes=weight_bytes;
    floor.tensors["partial"].element_bytes=2;floor.tensors["partial"].external_writes=full_out;
    auto traffic=DeriveBindingRequestTraffic(graph.nodes[0],floor);
    assert(traffic.produced_live_bytes==unsigned(tokens*5*2));
    for(long v=0;v<capacity;++v)for(long row=0;row<2;++row)for(long n=0;n<2;++n) {
      ParamBinding owner;owner.Bind("v",v);owner.Bind("row",row);owner.Bind("n",n);
      long rows=row==0?3:1,cols=n==0?4:3;
      auto value=[&](auto const& count){return count.BindCoordinates(owner).Eval({});};
      assert(value(traffic.read_bytes)==rows*5*2+cols*5*weight_bytes);
      assert(value(traffic.no_producer_read_bytes)==cols*5*weight_bytes);
      assert(value(traffic.write_bytes)==rows*cols*2);
      assert(value(traffic.external_write_bytes)==rows*cols*2);
      ++checks;
    }
    auto reject=[&](DramFloor wrong) {
      bool caught=false;try{(void)DeriveBindingRequestTraffic(graph.nodes[0],wrong);}
      catch(std::invalid_argument const&){caught=true;}assert(caught);
    };
    auto wrong=floor;wrong.tensors.erase("experts");reject(wrong);
    wrong=floor;wrong.tensors["experts"].element_bytes=0;reject(wrong);
    wrong=floor;
    wrong.tensors["hidden"].writes=CouplingRelation::FromIslText("{ [] -> [t,k] : 0<=t<1 and 0<=k<5 }");
    wrong.tensors["hidden"].no_producer=full_a.Subtract(wrong.tensors["hidden"].writes);reject(wrong);
    wrong=floor;
    wrong.tensors["partial"].external_writes=CouplingRelation::FromIslText("{ [] -> [t,r,n] : t=0 and 0<=r<3 and 0<=n<7 }");reject(wrong);
    floor.tensors["hidden"].writes={};floor.tensors["hidden"].no_producer=full_a;
    auto external=DeriveBindingRequestTraffic(graph.nodes[0],floor);
    assert(external.read_bytes.SemanticallyEqual(external.no_producer_read_bytes,{}));
    assert(external.produced_live_bytes==0);
    floor.tensors["partial"].external_writes={};
    auto internal=DeriveBindingRequestTraffic(graph.nodes[0],floor);
    assert(internal.external_write_bytes.IsZero());
  }
  assert(checks==(1+5+17)*2*2*2*2);
  std::cout<<"Binding request traffic: 368 tile byte/provenance cases PASS\n";
  return 0;
}

} // namespace tilemega::tests::binding_request_traffic_test
