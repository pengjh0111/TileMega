// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <cassert>
#include <iostream>

namespace tilemega::tests::binding_request_pricing_test {
int TestBindingRequestPricing(int,char**) {
  using namespace analysis;
  IslContext isl;
  auto f=[](long n){return ClosedForm::Constant(n);};
  unsigned cases=0,live_cases=0;
  for(long blocks:{1,5})for(unsigned bm:{4,24})for(int weight_bytes:{2,4}) {
    SemanticOp op;op.name="expert";op.kind=OperatorKind::kMatmul;
    op.dtype=analysis::ScalarType::kBF16;op.arithmetic="gemm";op.exact_task_access=true;
    op.domain={{"v",f(blocks)},{"row",f(bm)},{"n",f(19)},
        {"k",f(5),f(0),IteratorType::kReduction}};
    auto& virtual_dim=op.domain.front();virtual_dim.runtime=true;virtual_dim.capacity=f(blocks);
    virtual_dim.binding_source="bindings";virtual_dim.binding_requirement="prefix_sum";
    op.task_space={"owners",{{"v",f(blocks)},{"row",f(bm)},{"n",f(19)}}};
    op.task_map.results={IndexResult::Dim("v"),IndexResult::Dim("row"),IndexResult::Dim("n")};
    op.result={"partial",{{"t",f(19)},{"rank",f(3)},{"n",f(19)}}};
    op.result_map.results={IndexResult::DataDependent("rows",{"v","row"}),
        IndexResult::DataDependent("rows",{"v","row"}),IndexResult::Dim("n")};
    TensorSpace a{"hidden",{{"t",f(19)},{"k",f(5)}}};
    TensorSpace b{"experts",{{"e",f(2)},{"n",f(19)},{"k",f(5)}}};
    op.operands={{"input",a,{{IndexResult::DataDependent("rows",{"v","row"}),IndexResult::Dim("k")}},{}},
        {"",b,{{IndexResult::DataDependent("bindings",{"v"}),IndexResult::Dim("n"),IndexResult::Dim("k")}}, {}}};
    TensorSpace diagnostic{"diagnostic",op.task_space.axes};
    op.additional_writes.push_back({diagnostic,op.task_map,{}, {}});
    solver::ModelDescription model;model.dm=model.serving=true;
    model.dtype=solver::ScalarType::kBF16;model.dims={1,0,1};
    model.gemms={{19,5,0,1}};model.stages.resize(1);model.stages.front().gemm=0;
    model.gemm_access.resize(1);model.gemm_access.front().b=codegen::DmBAccess::kExpertIndirect;
    model.gemm_access.front().block_rows=bm;
    model.task_semantics={{op,{{"v",f(1)},{"row",f(16)},{"n",f(16)}},0,false}};
    solver::GemmConfig geometry{16,16,16,2,1};
    auto graph=solver::InstantiateModelTasks(model,{geometry});
    auto input=solver::DeriveModelTaskInput(model,model.task_semantics.front(),graph,&geometry);
    DramFloor floor;
    floor.tensors["hidden"].element_bytes=2;
    floor.tensors["hidden"].writes=CouplingRelation::FromIslText("{ [] -> [t,k] : 0<=t<19 and 0<=k<5 }");
    floor.tensors["experts"].element_bytes=weight_bytes;
    floor.tensors["partial"].element_bytes=2;
    floor.tensors["partial"].external_writes=CouplingRelation::FromIslText("{ [] -> [t,r,n] : 0<=t<19 and 0<=r<3 and 0<=n<19 }");
    floor.tensors["diagnostic"].element_bytes=4;
    floor.no_producer_bytes=QuasiPolynomial::Constant(2*19*5*weight_bytes);
    solver::BindTaskDramProvenance(input,model.task_semantics.front(),floor,{},true);
    assert(input.physical_read_bytes && input.physical_write_bytes);
    assert(input.produced_live_bytes==19*5*2 && input.stream_bytes==2*19*5*weight_bytes);
    std::vector<ParamBinding> points;
    std::vector<std::pair<long,long>> tails;
    for(long v=0;v<blocks;++v)for(long r=0;r<(bm+15)/16;++r)for(long n=0;n<2;++n) {
      ParamBinding at;
      at.Bind("v",v).Bind("row",r).Bind("n",n);points.push_back(at);
      tails.push_back({std::min<long>(16,bm-16*r),n?3:16});
    }
    auto traffic=solver::DeriveTaskMemoryTrafficBatch(input,{},points,2,2);
    for(unsigned i=0;i<points.size();++i) {
      auto [rows,columns]=tails[i];
      auto single=solver::DeriveTaskMemoryTraffic(input,{},points[i],2,2);
      assert(traffic[i].global_read_bytes==rows*5*2+columns*5*weight_bytes);
      assert(traffic[i].global_write_bytes==rows*columns*6);
      assert(traffic[i].no_producer_read_bytes==columns*5*weight_bytes);
      assert(traffic[i].produced_read_bytes==rows*5*2);
      assert(traffic[i].external_write_bytes==rows*columns*2);
      assert(single.global_read_bytes==traffic[i].global_read_bytes);
      assert(single.global_write_bytes==traffic[i].global_write_bytes);
      assert(single.no_producer_read_bytes==traffic[i].no_producer_read_bytes);
      assert(single.external_write_bytes==traffic[i].external_write_bytes);
      ++cases;
    }
    for(unsigned live:{0,1,3,4,8,16,17,23,24}) {
      if(live>bm)continue;
      auto traits=solver::ModelTaskTraits(model,0,geometry);
      auto restricted=solver::RestrictVirtualTaskRows(input,live,traits);
      assert(restricted.task.element_access!=input.task.element_access);
      assert(input.task.element_access->semantic.Dim("row")->BoundExtent().IsLiteral(bm));
      assert(restricted.work.task_count.SemanticallyEqual(input.work.task_count,{}));
      assert(restricted.work.nominal_write_elements.SemanticallyEqual(input.work.nominal_write_elements,{}));
      assert(restricted.work.nominal_task_reduce_extent.SemanticallyEqual(input.work.nominal_task_reduce_extent,{}));
      solver::BindTaskDramProvenance(restricted,model.task_semantics.front(),floor,{},true);
      auto actual=solver::DeriveTaskMemoryTrafficBatch(restricted,{},points,2,2);
      for(unsigned i=0;i<points.size();++i) {
        long row_begin=points[i].At("row")*16;
        long rows=std::max<long>(0,std::min<long>(16,long(live)-row_begin));
        auto columns=tails[i].second;
        assert(actual[i].global_read_bytes==rows*5*2+(rows?columns*5*weight_bytes:0));
        assert(actual[i].global_write_bytes==rows*columns*6);
        assert(actual[i].no_producer_read_bytes==(rows?columns*5*weight_bytes:0));
        assert(actual[i].produced_read_bytes==rows*5*2);
        assert(actual[i].external_write_bytes==rows*columns*2);
        auto const& conditioned=restricted.task.element_access->semantic;
        auto reads=ProjectTaskRequests(conditioned,restricted.task,
            restricted.task.element_access->partition,a,conditioned.operands[0].map,{},{});
        auto all=ProjectTaskRequests(op,input.task,input.task.element_access->partition,a,op.operands[0].map,{},{});
        assert(reads.IsSubset(all));
        ++live_cases;
      }
    }
    bool rejected=false;
    try{(void)solver::RestrictVirtualTaskRows(input,bm+1,solver::ModelTaskTraits(model,0,geometry));}
    catch(std::invalid_argument const&){rejected=true;}assert(rejected);
  }
  assert(cases==72);
  std::cout<<"Binding request pricing: 72 capacity and "<<live_cases
      <<" live-row mixed-width traffic, empty subtile and request containment cases PASS\n";
  return 0;
}
} // namespace tilemega::tests::binding_request_pricing_test
