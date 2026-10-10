// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Solver/DmOperatorClasses.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/DmVirtualGemmPartition.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <iostream>

namespace tilemega::tests::dm_virtual_frontend_test {
int TestDmVirtualFrontend(int,char**) {
  using namespace analysis;
  IslContext isl;
  auto f=[](long n){return ClosedForm::Constant(n);};
  unsigned cases=0;
  for(bool serving:{false,true})for(int bm:{16,32,64,128})
      for(int tm:{16,32,64,128})for(int tn:{16,32,64,128})for(int factor:{1,2}) {
    frontend::ModelPlan plan;plan.dm=true;plan.serving=serving;
    plan.buffers.resize(3);plan.gemms.resize(1);plan.stages.resize(1);
    auto& gemm=plan.gemms.front();gemm.a=0;gemm.b=1;gemm.d=2;
    gemm.n=70;gemm.k=48;gemm.access.b=codegen::DmBAccess::kExpertIndirect;
    gemm.access.block_rows=bm;plan.stages.front().gemm=0;
    SemanticOp op;op.name="layer0.expert";op.kind=OperatorKind::kMatmul;
    op.dtype=ScalarType::kBF16;op.arithmetic=factor==2?"swiglu_gemm":"gemm";
    op.exact_task_access=true;
    op.domain={{"block",f(5)},{"row",f(bm)},{"column",f(70/factor)},
        {"k",f(48),f(0),IteratorType::kReduction}};
    auto& v=op.domain[0];v.runtime=true;v.capacity=f(5);
    v.binding_source="layer0.bindings";v.binding_requirement="prefix_sum";
    op.result={"layer0.partial",{{"t",f(19)},{"rank",f(3)},{"n",f(70/factor)}}};
    op.result_map.results={IndexResult::DataDependent("layer0.rows",{"block","row"}),
        IndexResult::DataDependent("layer0.rows",{"block","row"}),IndexResult::Dim("column")};
    op.task_space={"layer0.owners",{{"virtual_owner",f(5)},{"row_owner",f(bm)},
        {"column_owner",f(70/factor)}}};
    op.task_map.results={IndexResult::Dim("block"),IndexResult::Dim("row"),IndexResult::Dim("column")};
    frontend::LiftedModel lifted;lifted.sem.ops={op};
    lifted.ops={{op.name,frontend::OpRole::kProjection,frontend::OwnershipKind::kTilePerBlock,0,0,{}}};
    auto g=frontend::LaunchGranularity(lifted,plan,{{tm,tn,16,2,1}});
    auto graph=Instantiate(lifted.sem,g);auto const& task=graph.nodes.front();
    assert(task.Count().Eval({},{})==5*((bm+tm-1)/tm)*((70/factor+tn/factor-1)/(tn/factor)));
    assert(task.tile[0].IsLiteral(1) && task.tile[1].IsLiteral(tm) && task.tile[2].IsLiteral(tn/factor));
    auto symbolic=op;
    auto tokens=ClosedForm::Symbol("T");
    symbolic.domain[0].capacity=(tokens*f(8)).CeilDiv(f(bm))+f(128).Min(tokens*f(8));
    lifted.sem.ops={symbolic};
    auto symbolic_g=frontend::LaunchGranularity(lifted,plan,{{tm,tn,16,2,1}});
    auto symbolic_graph=Instantiate(lifted.sem,symbolic_g);
    for(long token_count:{1,19,4096}) {
      ParamBinding theta;theta.Bind("T",token_count);
      long capacity=(token_count*8+bm-1)/bm+std::min<long>(128,token_count*8);
      assert(symbolic_graph.nodes.front().Count().Eval(theta,{})==
          capacity*((bm+tm-1)/tm)*((70/factor+tn/factor-1)/(tn/factor)));
    }
    bool rejected=false;
    try {Granularity invalid;ParamBinding theta;theta.Bind("T",0);
      solver::PartitionDmVirtualGemm(symbolic,gemm.access,{tm,tn,16,2,1},invalid,theta);}
    catch(std::invalid_argument const&){rejected=true;}assert(rejected);
    auto key=solver::GemmSemanticSignature(op,gemm,plan);
    auto renamed=op;renamed.name="layer47.expert";renamed.result.name="layer47.partial";
    renamed.task_space.name="layer47.owners";renamed.domain[0].binding_source="layer47.bindings";
    for(auto& index:renamed.result_map.results)
      if(!index.binding_source.empty())index.binding_source="layer47.rows";
    assert(solver::GemmSemanticSignature(renamed,gemm,plan)==key);
    auto changed=renamed;changed.domain[0].capacity=f(6);
    assert(solver::GemmSemanticSignature(changed,gemm,plan)!=key);
    ++cases;
  }
  frontend::ModelPlan legacy;legacy.buffers.resize(3);
  frontend::PlanGemm dense;dense.a=0;dense.b=1;dense.d=2;
  SemanticOp op;op.name="legacy";op.result={"result",{{"m",f(2)},{"n",f(3)}}};
  op.result_map.results={IndexResult::Dim("m"),IndexResult::Dim("n")};
  op.domain={{"m",f(2)},{"n",f(3)}};
  assert(solver::GemmSemanticSignature(op,dense,legacy)==SemanticSignature(op));
  legacy.dm=true;
  assert(solver::GemmSemanticSignature(op,dense,legacy)==SemanticSignature(op));
  assert(cases==256);
  std::cout<<"DM frontend: 256 virtual GEMM geometries, 768 symbolic bindings, canonical class integration and legacy key preservation PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_virtual_frontend_test
