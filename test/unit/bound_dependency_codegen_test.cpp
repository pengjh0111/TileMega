// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/BoundDependency.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/FlowPreparation.h>
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <mlir/IR/Verifier.h>
#include <cassert>
#include <fstream>
#include <set>

namespace tilemega::tests::bound_dependency_codegen_test {
namespace {
using namespace analysis;
using namespace frontend;
ClosedForm F(long value) { return ClosedForm::Constant(value); }
SemanticOp Tensor(char const* name, char const* value, long rows) {
  SemanticOp op; op.name=name; op.kind=OperatorKind::kMatmul; op.arithmetic="gemm";
  op.exact_task_access=true;
  auto m=ClosedForm::Symbol("B")*F(rows);
  op.domain={{"m",m},{"n",F(64)}};
  op.result={value,{{"m",m},{"n",F(64)}}};
  op.result_map.results={IndexResult::Dim("m"),IndexResult::Dim("n")};
  op.task_space=op.result; op.task_map=op.result_map; return op;
}
ImportedSemantics Fixture() {
  ImportedSemantics prepared;
  auto& bridge=prepared.bridge;
  FxNodeRecord input; input.name="x"; input.op="placeholder"; input.target="x";
  input.shape={"B","384","64"}; bridge.nodes.push_back(input);
  bridge.inputs.push_back({"x","USER_INPUT",""});
  bridge.range_texts={{"B","VR[1, 64]"}};
  prepared.symbolic=SymbolicShapeBridge{}.Parse(bridge.range_texts,{},{{"B","384","64"}});
  auto& plan=prepared.plan; plan.dm=plan.forward=true; plan.dtype="bf16"; plan.serving_seq=1;
  plan.buffers.resize(5);
  for (unsigned i=0;i<5;++i) plan.buffers[i].name="buffer"+std::to_string(i);
  plan.buffers[0].per_batch=384*64;
  plan.buffers[1].constant=plan.buffers[2].constant=64*64;
  plan.buffers[3].per_batch=384*64; plan.buffers[4].per_batch=112*64;
  PlanGemm p; p.n=p.k=64; p.a=0; p.b=1; p.d=3; p.access.rows_per_batch=384;
  PlanGemm c=p; c.a=3; c.b=2; c.d=4; c.access.rows_per_batch=112;
  plan.gemms={p,c};
  for (unsigned i=0;i<2;++i) {
    PlanStage stage; stage.gemm=i; stage.representative=i?"consumer":"producer";
    stage.representative_index=i+1; plan.stages.push_back(stage);
    FxNodeRecord fx; fx.name=stage.representative; fx.index=i+1; fx.op="call_function";
    fx.target="aten.linear.default"; fx.shape={"B",i?"112":"384","64"};
    bridge.nodes.push_back(fx); bridge.tasks.push_back(fx);
  }
  bridge.outputs={"consumer"}; plan.outputs={{4,""}};
  auto producer=Tensor("producer","produced",384), consumer=Tensor("consumer","consumed",112);
  consumer.domain.push_back({"r",F(2),F(0),IteratorType::kReduction});
  // Each 16-row owner reads rows from producer tiles 3*c and 3*c+2.
  // The missing middle tile makes a single interval incorrect.
  auto row=IndexResult::Affine({{"m",F(48),F(16)}, {"r",F(32),F(1)}});
  consumer.operands.push_back({producer.name,producer.result,{{row,IndexResult::Dim("n")}}, {}});
  prepared.lifted.sem.ops={producer,consumer}; prepared.lifted.has_plan=true;
  prepared.lifted.ops={{"producer",OpRole::kProjection,OwnershipKind::kTilePerBlock,0,0,"producer"},
                       {"consumer",OpRole::kProjection,OwnershipKind::kTilePerBlock,1,0,"consumer"}};
  prepared.lifted.written={0,0,0,1,1}; return prepared;
}
}
int TestBoundDependencyCodegen(int argc, char** argv) {
  IslContext isl; mlir::MLIRContext context; mlir::OpBuilder builder(&context);
  auto fixture=Fixture();
  for (long batch : {1,2,8}) {
    ImportOptions options; options.gemms={{16,64,64,3,1},{16,64,64,3,1}}; options.phase_batch=batch;
    auto module=TorchExportImporter{}.InstantiateForGranularity(fixture,context,options);
    assert(mlir::succeeded(mlir::verify(*module)));
    auto runtime=codegen::ReadRuntimePlan(*module);
    assert(runtime.dependencies.size()==1 && runtime.dependencies[0].table);
    auto table=*runtime.dependencies[0].table;
    assert(runtime.task_binding.At("B")==batch);
    assert(table.producers==24*batch && table.consumers==7*batch && table.stride==2);
    for (unsigned row=0;row<table.consumers;++row) {
      assert(table.intervals[row*2].first==row*3 && table.intervals[row*2].count==1);
      assert(table.intervals[row*2+1].first==row*3+2 && table.intervals[row*2+1].count==1);
    }
    solver::ModelDims dims;dims.seq=dims.total=1;dims.batch=batch;
    auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"bound-table");
    for (int grid : {1,3,8}) for (int kappa : {0,1,4,16}) {
      auto projected=solver::ProjectRuntimeQueues(model,runtime,{grid,128,kappa});
      assert(projected.runtime_windows.empty() && projected.runtime_tables.size()==1);
      assert(projected.runtime_task_refs.Eval(runtime.task_binding)==31*batch);
      std::set<std::vector<long>> expected_waits;
      auto dependencies=projected.dependencies.BindParams(runtime.task_binding).Points();
      assert(dependencies.size()==14*batch);
      for (auto const& [consumer,producer] : dependencies) {
        assert(consumer[0]==1 && producer[0]==0 &&
            (producer[1]==3*consumer[1] || producer[1]==3*consumer[1]+2));
      }
      for (long c=0;c<7*batch;++c) for (long p : {3*c,3*c+2}) {
        if (kappa==1 && p%grid==c%grid) continue;
        expected_waits.insert({c%grid,0,kappa==0?0:1,kappa==0?0:p/kappa});
      }
      assert(projected.runtime_wait_entries.Eval(runtime.task_binding)==expected_waits.size());
      std::set<std::vector<long>> actual_waits;
      for (auto const& [consumer,event] : projected.waits.BindParams(runtime.task_binding).Points())
        actual_waits.insert(event);
      assert(actual_waits==expected_waits);
      auto balanced=solver::BalanceProjectedQueues(projected,runtime.task_binding,grid);
      assert(balanced.task_ids.size()==31*batch);
    }
    for (int tile : {16,32,64}) {
      solver::SymbolicProblem base;base.model=model;base.runtime=runtime;base.threads=128;
      base.geometry={{tile,64,64,3,1},{tile,64,64,3,1}};
      analysis::CouplingCache cache;
      auto flow=solver::PrepareFlowStructure(base,base.geometry,8,4,cache);
      auto waits=solver::BindRuntimeWindows(flow.projection,0,1,model.MetricBindings());
      auto exact=flow.data_edges.at(0).relation.BindParams(model.MetricBindings()).Reverse().Points();
      std::vector<int> last(flow.counts[1],-1);
      for(auto const& [c,p] : exact) last[c[0]]=std::max(last[c[0]],int(p[0]));
      for(int c=0;c<flow.counts[1];++c)
        assert(solver::RuntimeReleaseEndpoint(-1,c,flow.counts[0],waits,false)==last[c]);
    }
    auto cu=codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,1,1}});
    assert(cu.find("StageDependency::Map::kTable")!=std::string::npos);
    assert(cu.find("#define TILEMEGA_DM_BOUND_BATCH "+std::to_string(batch)+"\n")!=std::string::npos);
    assert(cu.find("constexpr RuntimeDependencyInterval kDependencyIntervals0[]")!=std::string::npos);
    assert(cu.find("kDependencyIntervals0, "+std::to_string(14*batch)+"u")!=std::string::npos);
    if (argc==3 && std::string(argv[1])=="--emit" && batch==1) {
      std::ofstream output(argv[2]); assert(output); output<<cu;
    }
    auto edge=*module->getOps<dialect::CouplingOp>().begin();
    auto saved=edge->getAttrOfType<mlir::DictionaryAttr>("dependency_table");
    auto mutate=[&](char const* key,mlir::Attribute value) {
      llvm::SmallVector<mlir::NamedAttribute> attrs(saved.getValue());
      for (auto& attr:attrs) if(attr.getName().strref()==key) attr=builder.getNamedAttr(key,value);
      edge->setAttr("dependency_table",builder.getDictionaryAttr(attrs));
      assert(mlir::failed(mlir::verify(*module)));
      edge->setAttr("dependency_table",saved);
    };
    mutate("stride",builder.getI64IntegerAttr(1));
    mutate("producers",builder.getI64IntegerAttr(24*batch-1));
    auto values=saved.getAs<mlir::DenseI64ArrayAttr>("intervals").asArrayRef();
    llvm::SmallVector<std::int64_t> wrong(values); wrong[0]=1;
    mutate("intervals",builder.getDenseI64ArrayAttr(wrong));
    assert(mlir::succeeded(mlir::verify(*module)));

    // Stage pairs may have several input edges. Their union must be refit,
    // otherwise the first table would omit the additional middle tile.
    auto joined=fixture;
    auto middle=IndexResult::Affine({{"m",F(48),F(16)}},F(16));
    joined.lifted.sem.ops[1].operands.push_back({"producer",joined.lifted.sem.ops[0].result,
        {{middle,IndexResult::Dim("n")}}, {}});
    auto combined=TorchExportImporter{}.InstantiateForGranularity(joined,context,options);
    auto joined_runtime=codegen::ReadRuntimePlan(*combined);
    assert(joined_runtime.dependencies.size()==1 && !joined_runtime.dependencies[0].table);
    auto window=joined_runtime.dependencies[0].window;
    assert(window.narrowed && window.div==1 && window.scale==3 && window.offset==0 && window.count==3);
    auto single=codegen::CouplingGraphToCUDA{}.Lower(*combined);
    assert(single.find("StageDependency::Map::kWindow, 1u, 3, 0, 3u")!=std::string::npos);
  }
  bool rejected=false;
  try { (void)TorchExportImporter{}.InstantiateForGranularity(fixture,context,{}); }
  catch (std::exception const&) { rejected=true; }
  assert(rejected);
  return 0;
}
}
