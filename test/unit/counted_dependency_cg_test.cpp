// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/CountedDependency.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Analysis/ISLContext.h>
#include <mlir/IR/Verifier.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <set>
#include <fstream>
#include <cassert>
#include <iostream>

namespace tilemega::tests::counted_dependency_cg_test {
namespace {
using namespace analysis;
using namespace frontend;
ClosedForm F(long x){return ClosedForm::Constant(x);}
ImportedSemantics Fixture(int tokens) {
  ImportedSemantics prepared;
  FxNodeRecord input;input.name="x";input.op="placeholder";input.target="x";
  input.shape={"B","T","64"};prepared.bridge.nodes.push_back(input);
  prepared.bridge.inputs.push_back({"x","USER_INPUT",""});
  prepared.bridge.range_texts={{"B","VR[1, 64]"},{"T","VR[1, 65]"}};
  prepared.symbolic=SymbolicShapeBridge{}.Parse(prepared.bridge.range_texts,{},{{"B","T","64"}});
  auto& plan=prepared.plan;plan.dm=plan.forward=true;
  plan.dtype="bf16";plan.serving_seq=1;plan.buffers.resize(5);
  for(unsigned i=0;i<5;++i)plan.buffers[i].name="buffer"+std::to_string(i);
  plan.buffers[0].constant=tokens*64;
  plan.buffers[1].constant=plan.buffers[2].constant=64*16;
  plan.buffers[3].constant=tokens*2*64;plan.buffers[4].constant=tokens*64;
  PlanGemm p;p.n=64;p.k=16;p.a=0;p.b=1;p.d=3;p.access.rows_per_batch=tokens*2;
  auto c=p;c.a=3;c.b=2;c.d=4;c.access.rows_per_batch=tokens;
  plan.gemms={p,c};
  for(unsigned i=0;i<2;++i) {
    PlanStage stage;stage.gemm=i;stage.representative=i?"combine":"down";
    stage.representative_index=i+1;plan.stages.push_back(stage);
    FxNodeRecord fx;fx.name=stage.representative;fx.index=i+1;fx.op="call_function";
    fx.target="aten.linear.default";fx.shape={"T","64"};
    prepared.bridge.nodes.push_back(fx);prepared.bridge.tasks.push_back(fx);
  }
  prepared.bridge.outputs={"combine"};plan.outputs={{4,""}};
  auto t=ClosedForm::Symbol("T");
  SemanticOp down;down.name="down";down.exact_task_access=true;
  IterationDim m;m.name="m";m.extent=ClosedForm::Symbol("live");m.runtime=true;
  m.capacity=t*F(2);m.binding_source="rows";m.binding_requirement="tensor_values";
  down.domain={m,{"n",F(64)}};
  down.result={"partial",{{"t",t},{"r",F(2)},{"c",F(64)}}};
  down.result_map.results={IndexResult::DataDependent("rows"),IndexResult::DataDependent("rows"),IndexResult::Dim("n")};
  down.task_space={"virtual",{{"m",t*F(2)},{"n",F(64)}}};
  down.task_map.results={IndexResult::Dim("m"),IndexResult::Dim("n")};
  SemanticOp combine;combine.name="combine";combine.exact_task_access=true;
  combine.domain={{"m",t},{"r",F(2),F(0),IteratorType::kReduction},{"n",F(64)}};
  combine.result={"output",{{"m",t},{"n",F(64)}}};
  combine.result_map.results={IndexResult::Dim("m"),IndexResult::Dim("n")};
  combine.task_space=combine.result;combine.task_map=combine.result_map;
  combine.operands.push_back({"down",down.result,
      {{IndexResult::Dim("m"),IndexResult::Dim("r"),IndexResult::Dim("n")}}, {}});
  prepared.lifted.sem.ops={down,combine};prepared.lifted.has_plan=true;
  prepared.lifted.ops={{"down",OpRole::kProjection,OwnershipKind::kTilePerBlock,0,0,"down"},
      {"combine",OpRole::kProjection,OwnershipKind::kTilePerBlock,1,0,"combine"}};
  prepared.lifted.written={0,0,0,1,1};return prepared;
}
}
int TestCountedDependencyCg(int argc,char** argv) {
  IslContext isl;mlir::MLIRContext context;mlir::OpBuilder builder(&context);
  for(int tokens:{1,2,17,65})for(int block:{16,32,64,128})
    for(int consumer_rows:{16,32,64})for(int n:{16,32}) {
    ImportOptions options;options.gemms={{block,n,16,2,1},{consumer_rows,n,16,2,1}};
    options.phase_batch=1;options.task_binding.Bind("T",tokens);
    auto module=TorchExportImporter{}.InstantiateForGranularity(Fixture(tokens),context,options);
    assert(mlir::succeeded(mlir::verify(*module)));
    auto spaces=module->getOps<dialect::TileSpaceOp>();auto p=*spaces.begin();auto c=*std::next(spaces.begin());
    auto edge=*module->getOps<dialect::CouplingOp>().begin();
    assert(!dialect::ReadBoundCountedScatter(edge,options.task_binding));
    // Counted waits replace the conservative table, while I2 geometry remains.
    edge->removeAttr("dependency_table");
    auto saved=dialect::EncodeBoundCountedScatter(builder,p,c,"partial",{0,1},"rows",options.task_binding);
    edge->setAttr("dependency_counted",saved);
    auto contract=*dialect::ReadBoundCountedScatter(edge,options.task_binding);
    assert(mlir::succeeded(mlir::verify(*module)));
    auto runtime=codegen::ReadRuntimePlan(*module);
    assert(runtime.dependencies.size()==1 && runtime.dependencies[0].counted && !runtime.dependencies[0].table);
    assert(runtime.dependencies[0].counted->contributions.expected==contract.expected);
    assert(runtime.task_binding.At("T")==tokens);
    auto cu=codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,1,1}});
    assert(cu.find("StageDependency::Map::kCounted")!=std::string::npos);
    assert(cu.find("constexpr std::uint32_t kCountedThresholds0[]")!=std::string::npos);
    assert(cu.find("constexpr RuntimeDependencyInterval kDependencyIntervals0[]")!=std::string::npos);
    auto ordering=analysis::BuildDependencyTableLinear(runtime.dependencies[0].counted->conservative_relation,
        runtime.dependencies[0].counted->producers,contract.expected.size());
    assert(cu.find("kDependencyIntervals0, "+std::to_string(ordering.intervals.size())+"u")!=std::string::npos);
    assert(cu.find("kCountedThresholds0, "+std::to_string(contract.expected.size())+"u")!=std::string::npos);
    if(argc==3 && std::string(argv[1])=="--emit" && tokens==17 && block==16 && consumer_rows==16 && n==16) {
      std::ofstream output(argv[2]);assert(output);output<<cu;
    }
    solver::ModelDims dims;dims.seq=dims.total=dims.batch=1;
    auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"counted-projection");
    for(int grid:{1,3,8})for(int kappa:{0,1,4,16}) {
      auto projected=solver::ProjectRuntimeQueues(model,runtime,{grid,128,kappa});
      assert(projected.runtime_windows.empty() && projected.runtime_tables.empty());
      assert(projected.runtime_counted.size()==1);
      assert(projected.runtime_wait_entries.Eval(runtime.task_binding)==long(contract.expected.size()));
      long producers=((tokens*2+block-1)/block)*(64/n);
      assert(projected.runtime_task_refs.Eval(runtime.task_binding)==producers+long(contract.expected.size()));
      std::set<std::vector<long>> waits;
      for(auto const& [consumer,event]:projected.waits.BindParams(runtime.task_binding).Points()) {
        assert(consumer[0]==1 && event[0]==consumer[1]%grid && event[1]==0 && event[2]==3 && event[3]==consumer[1]);
        waits.insert(event);
      }
      assert(waits.size()==contract.expected.size());
      auto ordered=solver::BalanceProjectedQueues(projected,runtime.task_binding,grid);
      assert(ordered.wait_entries==long(contract.expected.size()));
      auto dependencies=projected.dependencies.BindParams(runtime.task_binding).Points();
      auto exact=runtime.dependencies[0].counted->conservative_relation.Points();
      assert(dependencies.size()==exact.size());
      for(auto const& [consumer,producer]:dependencies) {
        assert(consumer[0]==1 && producer[0]==0);
        assert(consumer[1]>=0 && consumer[1]<long(contract.expected.size()));
        assert(producer[1]>=0 && producer[1]<producers);
      }
    }
    int ntiles=64/n;
    assert(contract.expected.size()==unsigned(((tokens+consumer_rows-1)/consumer_rows)*ntiles));
    for(unsigned target=0;target<contract.expected.size();++target)
      assert(contract.expected[target]==unsigned(2*std::min(consumer_rows,tokens-int(target/ntiles)*consumer_rows)));
    auto reject=[&](auto action) {
      bool rejected=false;try{action();}catch(std::exception const&){rejected=true;}assert(rejected);
    };
    auto mutate=[&](char const* name,mlir::Attribute value) {
      llvm::SmallVector<mlir::NamedAttribute> attrs(saved.getValue());
      for(auto& attr:attrs)if(attr.getName().strref()==name)attr=builder.getNamedAttr(name,value);
      edge->setAttr("dependency_counted",builder.getDictionaryAttr(attrs));
      reject([&]{dialect::ReadBoundCountedScatter(edge,options.task_binding);});
      assert(mlir::failed(mlir::verify(*module)));
      edge->setAttr("dependency_counted",saved);
    };
    auto counts=saved.getAs<mlir::DenseI64ArrayAttr>("expected").asArrayRef();
    llvm::SmallVector<std::int64_t> wrong(counts);++wrong[0];
    mutate("expected",builder.getDenseI64ArrayAttr(wrong));
    mutate("expected",builder.getDenseI64ArrayAttr({}));
    mutate("unit_axes",builder.getDenseI64ArrayAttr({1,0}));
    mutate("binding_source",builder.getStringAttr("other"));
    mutate("binding_contract",builder.getStringAttr("arbitrary"));
    mutate("tensor",builder.getStringAttr("missing"));
    auto target_units=saved.getAs<dialect::CouplingMapAttr>("target_units").getMap();
    mutate("target_units",dialect::CouplingMapAttr::get(&context,target_units.Subtract(target_units)));
    mutate("binding",builder.getDictionaryAttr({builder.getNamedAttr("T",builder.getI64IntegerAttr(tokens+1))}));
    auto shape=options.task_binding;shape.Bind("T",tokens+1);
    reject([&]{dialect::ReadBoundCountedScatter(edge,shape);});
    auto original=edge.getRelationAttr();
    edge.setRelationAttr(dialect::CouplingMapAttr::get(&context,CouplingRelation::FromIslText("{ [c0,c1] -> [p0,p1] : false }")));
    reject([&]{dialect::ReadBoundCountedScatter(edge,options.task_binding);});
    edge.setRelationAttr(original);
    edge->setAttr("dependency_table",builder.getDictionaryAttr({}));
    reject([&]{dialect::ReadBoundCountedScatter(edge,options.task_binding);});
    edge->removeAttr("dependency_table");
    assert(dialect::ReadBoundCountedScatter(edge,options.task_binding));
    assert(mlir::succeeded(mlir::verify(*module)));
  }
  std::cout<<"Counted CG projection: 96 bound shapes x 12 grid/kappa choices, distinct counted events, I2 queue ordering and CUDA threshold arrays PASS\n";
  return 0;
}
} // namespace tilemega::tests::counted_dependency_cg_test
