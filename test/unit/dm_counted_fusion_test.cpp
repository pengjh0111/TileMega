// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CountedDependency.h>
#include <tilemega/Dialect/CouplingGraph/FusionPass.h>
#include <tilemega/Dialect/CouplingGraph/BoundDependency.h>
#include <tilemega/Solver/TaskModel.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/SymbolTable.h>
#include <cassert>
#include <iostream>
#include <set>
namespace tilemega::tests::dm_counted_fusion_test {
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
  auto c=p;c.k=2;c.a=3;c.b=2;c.d=4;c.access.rows_per_batch=tokens;
  plan.buffers[2].constant=64*2;
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
  down.kind=OperatorKind::kMatmul;down.arithmetic="gemm";
  down.domain={m,{"n",F(64)},{"k",F(16),F(0),IteratorType::kReduction}};
  down.reduction={"k","add","","",false};
  down.operands={{"",{"x",{{"t",t},{"k",F(16)}}},
      {{IndexResult::DataDependent("rows",{"m"}),IndexResult::Dim("k")}},{}},
      {"",{"weight",{{"n",F(64)},{"k",F(16)}}},
      {{IndexResult::Dim("n"),IndexResult::Dim("k")}}, {}}};
  down.result={"partial",{{"t",t},{"r",F(2)},{"c",F(64)}}};
  down.result_map.results={IndexResult::DataDependent("rows",{"m"}),IndexResult::DataDependent("rows",{"m"}),IndexResult::Dim("n")};
  down.task_space={"virtual",{{"m",t*F(2)},{"n",F(64)}}};
  down.task_map.results={IndexResult::Dim("m"),IndexResult::Dim("n")};
  SemanticOp combine;combine.name="combine";combine.exact_task_access=true;
  combine.kind=OperatorKind::kMatmul;combine.arithmetic="gemm";
  combine.reduction={"r","add","","",false};
  combine.domain={{"m",t},{"r",F(2),F(0),IteratorType::kReduction},{"n",F(64)}};
  combine.result={"output",{{"m",t},{"n",F(64)}}};
  combine.result_map.results={IndexResult::Dim("m"),IndexResult::Dim("n")};
  combine.task_space=combine.result;combine.task_map=combine.result_map;
  combine.operands.push_back({"down",down.result,
      {{IndexResult::Dim("m"),IndexResult::Dim("r"),IndexResult::Dim("n")}}, {}});
  combine.operands.push_back({"",{"combine.weight",{{"r",F(2)},{"n",F(64)}}},
      {{IndexResult::Dim("r"),IndexResult::Dim("n")}}, {}});
  prepared.lifted.sem.ops={down,combine};prepared.lifted.has_plan=true;
  prepared.lifted.ops={{"down",OpRole::kProjection,OwnershipKind::kTilePerBlock,0,0,"down"},
      {"combine",OpRole::kProjection,OwnershipKind::kTilePerBlock,1,0,"combine"}};
  prepared.lifted.written={0,0,0,1,1};
  auto pre=combine;pre.name="pre";pre.domain={{"m",t*F(2)},{"n",F(16)},
      {"k",F(16),F(0),IteratorType::kReduction}};pre.reduction={"k","add","","",false};
  pre.result={"x",{{"t",t*F(2)},{"k",F(16)}}};
  pre.result_map.results={IndexResult::Dim("m"),IndexResult::Dim("n")};
  pre.task_space={"pre.owners",{{"m",t*F(2)},{"n",F(16)}}};pre.task_map=pre.result_map;
  pre.operands={{"",{"input",{{"t",t*F(2)},{"k",F(16)}}},
      {{IndexResult::Dim("m"),IndexResult::Dim("k")}},{}},
      {"",{"pre.weight",{{"n",F(16)},{"k",F(16)}}},
      {{IndexResult::Dim("n"),IndexResult::Dim("k")}}, {}}};
  down.operands[0].producer="pre";
  down.operands[0].tensor=pre.result;
  down.operands[0].map.results[0]=IndexResult::Dim("m");
  auto sink=pre;sink.name="sink";sink.domain={{"m",t},{"n",F(48)},
      {"k",F(64),F(0),IteratorType::kReduction}};
  sink.result={"sink.output",{{"m",t},{"n",F(48)}}};
  sink.task_space={"sink.owners",{{"m",t},{"n",F(48)}}};
  sink.operands={{"combine",combine.result,
      {{IndexResult::Dim("m"),IndexResult::Dim("k")}},{}},
      {"",{"sink.weight",{{"n",F(48)},{"k",F(64)}}},
      {{IndexResult::Dim("n"),IndexResult::Dim("k")}}, {}}};
  plan.buffers.resize(9);
  for(unsigned i=0;i<9;++i)plan.buffers[i].name="buffer"+std::to_string(i);
  plan.buffers[0].constant=tokens*2*16;plan.buffers[5].constant=16*16;
  plan.buffers[6].constant=tokens*2*16;plan.buffers[7].constant=48*64;
  plan.buffers[8].constant=tokens*48;
  auto pg=p;pg.n=pg.k=16;pg.b=5;pg.d=6;pg.access.rows_per_batch=tokens*2;
  p.a=6;
  auto sg=c;sg.n=48;sg.k=64;sg.a=4;sg.b=7;sg.d=8;
  plan.gemms={pg,p,c,sg};plan.stages.clear();
  prepared.bridge.nodes.resize(1);prepared.bridge.tasks.clear();
  for(unsigned i=0;i<4;++i) {
    auto name=std::vector<std::string>{"pre","down","combine","sink"}[i];
    PlanStage stage;stage.gemm=i;stage.representative=name;stage.representative_index=i+1;
    plan.stages.push_back(stage);
    FxNodeRecord fx;fx.name=name;fx.index=i+1;fx.op="call_function";
    fx.target="aten.linear.default";fx.shape={"T",std::to_string(plan.gemms[i].n)};
    prepared.bridge.nodes.push_back(fx);prepared.bridge.tasks.push_back(fx);
  }
  prepared.bridge.outputs={"sink"};plan.outputs={{8,""}};
  prepared.lifted.sem.ops={pre,down,combine,sink};
  prepared.lifted.ops.clear();
  for(unsigned i=0;i<4;++i) {
    auto name=plan.stages[i].representative;
    prepared.lifted.ops.push_back({name,OpRole::kProjection,OwnershipKind::kTilePerBlock,int(i),0,name});
  }
  return prepared;
}
}
int TestDmCountedFusion(int,char**) {
  IslContext isl;mlir::MLIRContext context;mlir::OpBuilder builder(&context);
  unsigned checked=0,rejected=0,unit_pairs=0;
  for(int tokens:{1,17,65})for(int block:{16,32,128})
    for(int rows:{16,32,64})for(int mode:{1,2,3}) {
    std::cerr<<"CASE T="<<tokens<<" block="<<block<<" rows="<<rows<<" mode="<<mode<<"\n";
    ImportOptions options;options.gemms={{block,16,16,2,1},{block,64,16,2,1},
        {rows,64,16,2,1},{16,16,16,2,1}};
    options.phase_batch=1;options.task_binding.Bind("T",tokens);
    auto module=TorchExportImporter{}.InstantiateForGranularity(Fixture(tokens),context,options);
    dialect::TileSpaceOp down,combine;
    for(auto task:module->getOps<dialect::TileSpaceOp>()) {
      if(task.getOperatorName()=="down")down=task;
      if(task.getOperatorName()=="combine")combine=task;
    }
    dialect::CouplingOp counted;
    for(auto edge:module->getOps<dialect::CouplingOp>())
      if(edge.getSrc()==down.getSymName() && edge.getDst()==combine.getSymName())counted=edge;
    assert(counted);counted->removeAttr("dependency_table");
    counted->setAttr("dependency_counted",dialect::EncodeBoundCountedScatter(
        builder,down,combine,"partial",{0,1},"rows",options.task_binding));
    assert(mlir::succeeded(mlir::verify(*module)));
    auto old=*dialect::ReadBoundCountedScatter(counted,options.task_binding);
    auto old_relation=counted.getRelation().getMap();
    std::vector<std::pair<std::string,std::string>> pairs;
    if(mode&1)pairs.push_back({"pre","down"});
    if(mode&2)pairs.push_back({"combine","sink"});
    solver::ModelDims dims;dims.seq=dims.total=dims.batch=1;
    auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"counted-fusion");
    analysis::CouplingRelation consumer_map;
    if(mode&2)consumer_map=solver::DeriveLogicalFusionCandidate(model,
        {{block,16,16,2,1},{block,64,16,2,1},{rows,64,16,2,1},{16,16,16,2,1}},
        "combine","sink").accesses.consumer_to_producer;
    try {dialect::FuseTaskPairs(*module,pairs);}
    catch(std::exception const& e) {throw std::invalid_argument("counted fixture T="+
        std::to_string(tokens)+" block="+std::to_string(block)+" rows="+std::to_string(rows)+
        " mode="+std::to_string(mode)+": "+e.what());}
    assert(mlir::succeeded(mlir::verify(*module)));
    assert(solver::ReadFusedTaskInputs(*module).size()==pairs.size());
    for(auto edge:module->getOps<dialect::CouplingOp>())
      if(edge->hasAttr("dependency_counted"))counted=edge;
    assert(!counted->hasAttr("dependency_table") && !counted->hasAttr("wait_map"));
    auto result=*dialect::ReadBoundCountedScatter(counted,options.task_binding);
    auto geometry=*dialect::ReadBoundTaskGeometry(counted,options.task_binding);
    unsigned targets=mode&2?((tokens+15)/16)*3:(tokens+rows-1)/rows;
    assert(result.expected.size()==targets && geometry.consumers==targets);
    std::set<std::pair<long,std::vector<long>>> expected;
    for(unsigned target=0;target<targets;++target) {
      long original_row=(mode&2?long(target/3)*16:long(target)*rows)/rows;
      unsigned count=2*std::min(rows,tokens-int(original_row)*rows);
      assert(result.expected[target]==count);
      for(long token=original_row*rows;token<std::min(long(tokens),(original_row+1)*rows);++token)
        for(long rank=0;rank<2;++rank)expected.insert({target,{token,rank}});
    }
    std::set<std::pair<long,std::vector<long>>> actual;
    for(auto const& [task,unit]:result.target_units.Points()) {
      assert(task.size()==1);actual.insert({task[0],unit});++unit_pairs;
    }
    assert(actual==expected);
    if(mode&2) {
      auto rebased=consumer_map.ApplyRange(old_relation);
      assert(rebased.IsSubset(counted.getRelation().getMap()) &&
          counted.getRelation().getMap().IsSubset(rebased));
    }
    auto saved=counted->getAttrOfType<mlir::DictionaryAttr>("dependency_counted");
    auto reject=[&](auto action) {
      bool caught=false;try{action();}catch(std::invalid_argument const&){caught=true;}
      assert(caught);++rejected;
    };
    llvm::SmallVector<mlir::NamedAttribute> attrs(saved.getValue());
    auto corrupt=[&](char const* field,mlir::Attribute value) {
      auto bad=attrs;
      for(auto& entry:bad)if(entry.getName().strref()==field)entry=builder.getNamedAttr(field,value);
      counted->setAttr("dependency_counted",builder.getDictionaryAttr(bad));
      reject([&]{(void)dialect::ReadBoundCountedScatter(counted,options.task_binding);});
      counted->setAttr("dependency_counted",saved);
    };
    auto counts=saved.getAs<mlir::DenseI64ArrayAttr>("expected").asArrayRef();
    llvm::SmallVector<int64_t> wrong(counts);++wrong[0];
    corrupt("expected",builder.getDenseI64ArrayAttr(wrong));
    corrupt("unit_axes",builder.getDenseI64ArrayAttr({1,0}));
    corrupt("binding_source",builder.getStringAttr("wrong"));
    corrupt("target_units",dialect::CouplingMapAttr::get(&context,
        result.target_units.Subtract(result.target_units)));
    auto relation=counted.getRelationAttr();
    counted.setRelationAttr(dialect::CouplingMapAttr::get(&context,
        counted.getRelation().getMap().Subtract(counted.getRelation().getMap())));
    reject([&]{(void)dialect::ReadBoundCountedScatter(counted,options.task_binding);});
    counted.setRelationAttr(relation);
    if(mode&1) {
      auto fused=llvm::cast<dialect::FusedTileSpaceOp>(mlir::SymbolTable::lookupNearestSymbolFrom(
          counted,counted.getSrcAttr()));
      auto writes=fused.getWrites();auto values=mlir::NamedAttrList(writes);
      auto physical=writes.getAs<dialect::CouplingMapAttr>("partial").getMap();
      values.set("partial",dialect::CouplingMapAttr::get(&context,physical.Subtract(physical)));
      fused->setAttr("writes",values.getDictionary(&context));
      reject([&]{(void)dialect::ReadBoundCountedScatter(counted,options.task_binding);});
      fused->setAttr("writes",writes);
      auto maps=fused.getPhaseMaps();llvm::SmallVector<mlir::Attribute> phases(maps.getValue());
      auto map=llvm::cast<dialect::CouplingMapAttr>(phases[1]).getMap();
      phases[1]=dialect::CouplingMapAttr::get(&context,map.Subtract(map));
      fused->setAttr("phase_maps",builder.getArrayAttr(phases));
      reject([&]{(void)dialect::ReadBoundCountedScatter(counted,options.task_binding);});
      fused->setAttr("phase_maps",maps);
    }
    if(mode&2) {
      auto fused=llvm::cast<dialect::FusedTileSpaceOp>(mlir::SymbolTable::lookupNearestSymbolFrom(
          counted,counted.getDstAttr()));
      auto reads=fused.getReads();auto values=mlir::NamedAttrList(reads);
      auto physical=reads.getAs<dialect::CouplingMapAttr>("partial").getMap();
      values.set("partial",dialect::CouplingMapAttr::get(&context,physical.Subtract(physical)));
      fused->setAttr("reads",values.getDictionary(&context));
      reject([&]{(void)dialect::ReadBoundCountedScatter(counted,options.task_binding);});
      fused->setAttr("reads",reads);
    }
    assert(mlir::succeeded(mlir::verify(*module)));
    ++checked;
  }
  assert(checked==81 && rejected==567);
  std::cout<<"COUNTED_FUSION shapes="<<checked<<" rejected="<<rejected
      <<" exact_unit_pairs="<<unit_pairs<<" PASS\n";
  return 0;
}
}
