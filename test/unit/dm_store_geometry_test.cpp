// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskOwnershipGeometry.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/FlowPreparation.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <cassert>
#include <iostream>

namespace tilemega::tests::dm_store_geometry_test {
namespace {
using namespace analysis;
using namespace frontend;
ClosedForm F(long x){return ClosedForm::Constant(x);}
IndexResult I(char const* name,long coefficient=1,long group=1) {
  return IndexResult::Dim(name,F(coefficient),F(group));
}
IndexResult Sum(std::initializer_list<IndexResult> terms) {
  IndexResult result;
  for(auto const& term:terms)result.terms.insert(result.terms.end(),term.terms.begin(),term.terms.end());
  return result;
}
ImportedSemantics Fixture(int batch,bool shuffle) {
  int rows=15*batch,n=shuffle?64:32,height=shuffle?6:3,width=shuffle?10:5;
  int channels=shuffle?16:32,consumer_rows=height*width*batch;
  ImportedSemantics fixture;
  FxNodeRecord input;input.name="input";input.op="placeholder";input.target="input";
  input.shape={"B","15","64"};fixture.bridge.nodes.push_back(input);
  fixture.bridge.inputs.push_back({"input","USER_INPUT",""});
  fixture.bridge.range_texts={{"B","VR[1, 64]"}};
  fixture.symbolic=SymbolicShapeBridge{}.Parse(fixture.bridge.range_texts,{},{{"B","15","64"}});
  auto& plan=fixture.plan;plan.dm=plan.forward=true;plan.dtype="bf16";plan.serving_seq=1;
  plan.buffers.resize(5);
  for(unsigned i=0;i<5;++i)plan.buffers[i].name="buffer"+std::to_string(i);
  plan.buffers[0].per_batch=15*64;plan.buffers[1].constant=n*64;
  plan.buffers[2].constant=48*channels;plan.buffers[3].per_batch=15*n;
  plan.buffers[4].per_batch=height*width*48;
  PlanGemm p;p.n=n;p.k=64;p.a=0;p.b=1;p.d=3;p.access.rows_per_batch=15;
  auto c=p;c.n=48;c.k=channels;c.a=3;c.b=2;c.d=4;c.access.rows_per_batch=height*width;
  plan.gemms={p,c};
  for(unsigned i=0;i<2;++i) {
    PlanStage stage;stage.gemm=i;stage.representative=i?"consume":"store";
    stage.representative_index=i+1;plan.stages.push_back(stage);
    FxNodeRecord node;node.name=stage.representative;node.index=i+1;node.op="call_function";
    node.target="aten.linear.default";node.shape={"B",std::to_string(i?height*width:15),std::to_string(i?48:n)};
    fixture.bridge.nodes.push_back(node);fixture.bridge.tasks.push_back(node);
  }
  fixture.bridge.outputs={"consume"};plan.outputs={{4,""}};
  auto operation=[](char const* name,int m,int n,int k) {
    SemanticOp op;op.name=name;op.kind=OperatorKind::kMatmul;op.arithmetic="gemm";
    op.dtype=ScalarType::kBF16;op.exact_task_access=true;
    op.domain={{"m",F(m)},{"n",F(n)},{"k",F(k),F(0),IteratorType::kReduction}};
    op.task_space={std::string(name)+".owners",{{"row",F(m)},{"column",F(n)}}};
    op.task_map.results={I("m"),I("n")};
    op.reduction={"k","add",std::string(name)+".partial",std::string(name)+".combine",true};
    return op;
  };
  auto store=operation("store",rows,n,64);
  store.result={"mapped",{{"image",F(batch)},{"channel",F(channels)},{"height",F(height)},{"width",F(width)}}};
  store.result_map.results=shuffle?std::vector<IndexResult>{I("m",1,15),I("n",1,4),
      Sum({I("m",2,5),I("m",-6,15),I("n",1,2),I("n",-2,4)}),
      Sum({I("m",2),I("m",-10,5),I("n"),I("n",-2,2)})}:
      std::vector<IndexResult>{I("m",1,15),I("n"),Sum({I("m",1,5),I("m",-3,15)}),Sum({I("m"),I("m",-5,5)})};
  store.operands={{"",{"input",{{"m",F(rows)},{"k",F(64)}}},{{I("m"),I("k")}},{}},
      {"",{"weight0",{{"n",F(n)},{"k",F(64)}}},{{I("n"),I("k")}}, {}}};
  auto consume=operation("consume",consumer_rows,48,channels);
  consume.result={"output",{{"m",F(consumer_rows)},{"n",F(48)}}};
  consume.result_map.results={I("m"),I("n")};
  consume.operands={{"store",store.result,{{I("m",1,height*width),I("k"),
      Sum({I("m",1,width),I("m",-height,height*width)}),Sum({I("m"),I("m",-width,width)})}},{}},
      {"",{"weight1",{{"n",F(48)},{"k",F(channels)}}},{{I("n"),I("k")}}, {}}};
  fixture.lifted.sem.ops={store,consume};fixture.lifted.has_plan=true;
  fixture.lifted.ops={{"store",OpRole::kProjection,OwnershipKind::kTilePerBlock,0,0,"store"},
      {"consume",OpRole::kProjection,OwnershipKind::kTilePerBlock,1,0,"consume"}};
  fixture.lifted.written={0,0,0,1,1};return fixture;
}
}
int TestDmStoreGeometry(int,char**) {
  using namespace tilemega;
  IslContext isl;mlir::MLIRContext context;
  for(int batch:{1,2})for(bool shuffle:{false,true}) {
    auto fixture=Fixture(batch,shuffle);
    ImportOptions options;options.gemms={{16,16,16,2,1},{16,16,16,2,1}};
    options.phase_batch=batch;options.combiner_tile_per_block=true;
    auto module=TorchExportImporter{}.InstantiateForGranularity(fixture,context,options);
    solver::ModelDims dims;dims.seq=dims.total=1;dims.batch=batch;
    auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"mapped-geometry");
    assert(fixture.plan.forward && !fixture.plan.serving);
    assert(model.dm && model.task_semantics.size()==2);
    assert(model.task_semantics[0].tiles.count("row") && model.task_semantics[0].tiles.count("column"));
    auto const& op=model.task_semantics[0].op;
    assert(TaskOwnershipSpace(op).axes.size()==2 && op.result.axes.size()==4);
    assert(UnitTaskOwnershipDimension(op,0)=="m" && UnitTaskOwnershipDimension(op,1)=="n");
    auto traits=solver::ModelTaskTraits(model,0,{16,16,16,2,1});
    assert(traits.threads==solver::DmServingBF16Traits(16,16,16,2).threads && traits.shape_legal);
    for(int m:{16,32})for(int n:{16,32})for(int split:{1,3}) {
      std::vector<solver::GemmConfig> geometry{{m,n,16,2,split},{16,16,16,2,1}};
      auto graph=solver::InstantiateModelTasks(model,geometry);
      auto const& source=*graph.Find("store");
      int chunks=split==1?1:3;
      assert(source.tile[0].Eval({},{})==m && source.tile[1].Eval({},{})==n);
      auto input=solver::DeriveModelTaskInput(model,model.task_semantics[0],graph,&geometry[0]);
      auto writes=input.work.write_elements.SumDomain().Eval({});
      assert(writes==15*batch*(shuffle?64:32)*chunks);
      if(chunks>1) {
        auto combine=solver::DeriveCombineTaskInput(model,0,geometry[0],graph,128,true,true);
        long output=15*batch*(shuffle?64:32);
        assert(combine.work.write_elements.SumDomain().Eval({})==output);
        assert(combine.work.read_elements.SumDomain().Eval({})==output*chunks);
        assert(combine.physical_read_bytes->SumDomain().Eval({})==output*chunks*4);
        assert(combine.scalar_access->writes.RangeDimNames().size()==4);
      }
      solver::SymbolicProblem base;base.model=model;base.runtime=codegen::ReadRuntimePlan(*module);
      base.geometry=geometry;base.threads=128;
      analysis::CouplingCache cache;
      auto flow=solver::PrepareFlowStructure(base,geometry,3,4,cache);
      assert(!flow.counts.empty() && !flow.data_edges.empty());
      long producer=((15*batch+m-1)/m)*((int(shuffle?64:32)+n-1)/n);
      long consumer=(((shuffle?60:15)*batch+15)/16)*3;
      assert(flow.counts[0]==producer*chunks && flow.counts.back()==consumer);
      if(chunks>1)assert(flow.counts[1]==producer);
      long count=0;
      for(auto value:flow.counts)count+=value;
      assert(flow.projection.runtime_task_refs.Eval(model.MetricBindings())==count);
    }
  }
  std::cout<<"DM store geometry: NCHW/pixel shuffle, aliased ownership axes, split-K work and flow rebinding PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_store_geometry_test
