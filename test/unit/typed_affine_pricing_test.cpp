// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <cassert>
#include <iostream>
namespace tilemega::tests::typed_affine_pricing_test {
int TestTypedAffinePricing(int,char**) {
  using namespace analysis;IslContext isl;
  auto f=[](long n){return ClosedForm::Constant(n);};unsigned cases=0;
  for(long m:{1,17,35})for(long n:{1,19,33})for(int bytes:{2,4}) {
    SemanticOp op;op.name="affine";op.kind=OperatorKind::kMatmul;
    op.dtype=ScalarType::kBF16;op.arithmetic="gemm";op.exact_task_access=true;
    op.domain={{"m",f(m)},{"n",f(n)},{"k",f(5),f(0),IteratorType::kReduction}};
    op.task_space={"owners",{{"m",f(m)},{"n",f(n)}}};
    op.task_map.results={IndexResult::Dim("m"),IndexResult::Dim("n")};
    op.result={"out",op.task_space.axes};op.result_map=op.task_map;
    TensorSpace a{"hidden",{{"m",f(m)},{"k",f(5)}}};
    TensorSpace b{"weight",{{"n",f(n)},{"k",f(5)}}};
    op.operands={{"input",a,{{IndexResult::Dim("m"),IndexResult::Dim("k")}},{}},
        {"",b,{{IndexResult::Dim("n"),IndexResult::Dim("k")}}, {}}};
    TensorSpace stats{"stats",{{"m",f(m)},{"part",f((n+15)/16)},{"stat",f(2)}}};
    for(long statistic:{0,1})op.additional_writes.push_back({stats,
        {{IndexResult::Dim("m"),IndexResult::Dim("n",f(1),f(16)),IndexResult::Affine({},f(statistic))}}, {}, {}});
    solver::ModelDescription model;model.dm=model.serving=true;
    model.dtype=solver::ScalarType::kBF16;model.dims={1,0,1};
    model.gemms={{n,5,0,1}};model.stages.resize(1);model.stages.front().gemm=0;
    model.task_semantics={{op,{{"m",f(16)},{"n",f(16)}},0,false}};
    solver::GemmConfig geometry{16,16,16,2,1};
    auto graph=solver::InstantiateModelTasks(model,{geometry});
    auto input=solver::DeriveModelTaskInput(model,model.task_semantics.front(),graph,&geometry);
    auto image=[](std::string body){return CouplingRelation::FromIslText("{ [] -> "+body+" }");};
    long produced=std::min(9L,m);DramFloor floor;
    floor.tensors["hidden"].element_bytes=2;
    floor.tensors["hidden"].writes=image("[m,k] : 0<=m<"+std::to_string(produced)+" and 0<=k<5");
    floor.tensors["hidden"].no_producer=image("[m,k] : "+std::to_string(produced)+"<=m<"+std::to_string(m)+" and 0<=k<5");
    floor.tensors["weight"].element_bytes=bytes;
    floor.tensors["weight"].no_producer=image("[n,k] : 0<=n<"+std::to_string(n)+" and 0<=k<5");
    floor.tensors["out"].element_bytes=2;
    floor.tensors["out"].external_writes=image("[m,n] : 0<=m<"+std::to_string(m)+" and 0<=n<"+std::to_string(n));
    floor.tensors["stats"].element_bytes=4;
    floor.no_producer_bytes=QuasiPolynomial::Constant(n*5*bytes+(m-produced)*5*2);
    solver::BindTaskDramProvenance(input,model.task_semantics.front(),floor,{},true);
    assert(input.physical_read_bytes && input.physical_write_bytes);
    assert(input.produced_live_bytes==produced*5*2);
    assert(input.stream_bytes==n*5*bytes+(m-produced)*5*2);
    std::vector<ParamBinding> points;
    for(long r=0;r<(m+15)/16;++r)for(long c=0;c<(n+15)/16;++c) {
      ParamBinding point;point.Bind("m",r).Bind("n",c);points.push_back(point);
    }
    auto batch=solver::DeriveTaskMemoryTrafficBatch(input,{},points,2,2);
    for(unsigned i=0;i<points.size();++i) {
      long r=points[i].At("m"),c=points[i].At("n");
      long rows=std::min(16L,m-16*r),columns=std::min(16L,n-16*c);
      long live_rows=std::max(0L,std::min(rows,produced-16*r));
      auto single=solver::DeriveTaskMemoryTraffic(input,{},points[i],2,2);
      for(auto const* traffic:{&single,&batch[i]}) {
        assert(traffic->global_read_bytes==rows*5*2+columns*5*bytes);
        assert(traffic->global_write_bytes==rows*columns*2+rows*2*4);
        assert(traffic->no_producer_read_bytes==(rows-live_rows)*5*2+columns*5*bytes);
        assert(traffic->produced_read_bytes==live_rows*5*2);
        assert(traffic->external_write_bytes==rows*columns*2);
      }
      ++cases;
    }
  }
  assert(cases==72);
  std::cout<<"Typed affine pricing: 72 main/statistic-store, mixed-provenance and tail cases PASS\n";
  return 0;
}
}
