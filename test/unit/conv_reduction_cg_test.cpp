// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/FlowPreparation.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Dialect/CouplingGraph/HandoffPass.h>
#include <tilemega/Dialect/CouplingGraph/TaskReductionGeometry.h>
#include <mlir/IR/Verifier.h>
#include <cassert>
#include <iostream>

namespace tilemega::tests::conv_reduction_cg_test {
namespace {
using namespace analysis;
ClosedForm F(long value) {return ClosedForm::Constant(value);}
IndexResult I(char const* name,long scale=1,long group=1) {
  return IndexResult::Dim(name,F(scale),F(group));
}
IndexResult Sum(std::initializer_list<IndexResult> parts) {
  IndexResult out;
  for(auto const& part:parts)out.terms.insert(out.terms.end(),part.terms.begin(),part.terms.end());
  return out;
}
frontend::ImportedSemantics Fixture(unsigned batch,unsigned channels,unsigned filter=3) {
  using namespace frontend;
  ImportedSemantics fixture;
  FxNodeRecord input;input.name="input";input.op="placeholder";input.target="input";
  input.shape={"B",std::to_string(filter),std::to_string(filter),std::to_string(channels)};
  FxNodeRecord node;node.name="conv";node.op="call_function";node.target="aten.conv2d.default";
  node.index=1;node.shape={"B","7","3","3"};node.dtype="torch.bfloat16";
  fixture.bridge.nodes={input,node};fixture.bridge.tasks={node};
  fixture.bridge.inputs={{"input","USER_INPUT",""}};fixture.bridge.outputs={"conv"};
  fixture.bridge.range_texts={{"B","VR[1, 64]"}};
  fixture.symbolic=SymbolicShapeBridge{}.Parse(fixture.bridge.range_texts,{},{{"B","3","3",std::to_string(channels)}});
  auto& plan=fixture.plan;plan.dm=plan.forward=true;plan.dtype="bf16";plan.serving_seq=1;
  plan.buffers.resize(3);
  auto cp=channels==3?4:(channels+7)/8*8;
  auto pitch=(cp+7)/8*8,side=filter+2;
  for(unsigned id=0;id<3;++id) {plan.buffers[id].name="buffer"+std::to_string(id);plan.buffers[id].dtype="torch.bfloat16";}
  auto& layout=plan.buffers[0].layout;layout.kind=codegen::DmLayout::kNHWC;layout.rank=4;
  layout.logical[0]=batch;layout.logical[1]=layout.logical[2]=filter;layout.logical[3]=channels;
  layout.physical[0]=batch;layout.physical[1]=layout.physical[2]=side;layout.physical[3]=cp;
  layout.halo_top=layout.halo_bottom=layout.halo_left=layout.halo_right=1;
  layout.strides[3]=1;layout.strides[2]=pitch;layout.strides[1]=side*pitch;layout.strides[0]=side*side*pitch;
  auto& output=plan.buffers[2].layout;output.kind=codegen::DmLayout::kNHWC;output.rank=4;
  output.logical[0]=batch;output.logical[1]=output.logical[2]=3;output.logical[3]=7;
  output.physical[0]=batch;output.physical[1]=output.physical[2]=3;output.physical[3]=8;
  output.strides[3]=1;output.strides[2]=8;output.strides[1]=24;output.strides[0]=72;
  plan.buffers[0].per_batch=side*side*pitch;plan.buffers[1].constant=7*filter*filter*cp;plan.buffers[2].per_batch=72;
  codegen::ConvDesc conv;conv.n=batch;conv.h=conv.w=filter;conv.p=conv.q=3;
  conv.c=channels;conv.k=7;conv.r=conv.s=filter;conv.pad_h=conv.pad_w=1;
  conv.input_layout=0;conv.output_layout=2;
  plan.convolutions={conv};
  PlanGemm gemm;gemm.n=7;gemm.k=filter*filter*channels;gemm.a=0;gemm.b=1;gemm.d=2;
  gemm.access.a=codegen::DmAAccess::kIm2Col;gemm.access.conv=0;gemm.access.rows_per_batch=9;
  plan.gemms={gemm};PlanStage stage;stage.gemm=0;stage.representative="conv";
  stage.representative_index=1;stage.conv=0;plan.stages={stage};plan.outputs={{2,""}};
  SemanticOp op;op.name="conv";op.kind=OperatorKind::kMatmul;op.arithmetic="gemm";
  op.dtype=ScalarType::kBF16;op.exact_task_access=true;
  op.domain={{"m",F(9*batch)},{"n",F(7)},
      {"c",F(channels),F(0),IteratorType::kReduction},
      {"r",F(filter),F(0),IteratorType::kReduction},{"s",F(filter),F(0),IteratorType::kReduction}};
  op.task_space={"owners",{{"m",F(9*batch)},{"n",F(7)}}};op.task_map.results={I("m"),I("n")};
  op.result={"output",{{"image",F(batch)},{"height",F(3)},{"width",F(3)},{"channel",F(7)}}};
  op.result_map.results={I("m",1,9),Sum({I("m",1,3),I("m",-3,9)}),Sum({I("m"),I("m",-3,3)}),I("n")};
  op.operands={{"",{"input",{{"image",F(batch)},{"height",F(side)},{"width",F(side)},{"channel",F(cp)}}},
      {{I("m",1,9),Sum({I("m",1,3),I("m",-3,9),I("r")}),Sum({I("m"),I("m",-3,3),I("s")}),I("c")}},{}},
      {"",{"weight",{{"n",F(7)},{"r",F(filter)},{"s",F(filter)},{"c",F(cp)}}},{{I("n"),I("r"),I("s"),I("c")}}, {}}};
  op.reduction={"c","add","partials","conv.combine",true};
  fixture.lifted.sem.ops={op};fixture.lifted.has_plan=true;
  fixture.lifted.ops={{"conv",OpRole::kProjection,OwnershipKind::kTilePerBlock,0,0,"conv"}};
  fixture.lifted.written={0,0,1};return fixture;
}
}
int TestConvReductionCg(int,char**) {
  IslContext isl;
  mlir::MLIRContext context;unsigned cases=0,rejected=0;
  auto equal=[](auto const& a,auto const& b) {return Contains(a,b)&&Contains(b,a);};
  for(unsigned batch:{1,2})for(unsigned c:{3,24})for(int tk:{16,32})for(int split:{1,2}) {
    auto fixture=Fixture(batch,c);
    auto cp=fixture.plan.buffers[0].layout.physical[3];
    if(cp<unsigned(tk) && unsigned(tk)%cp) {
      frontend::ImportOptions invalid;invalid.gemms={{16,16,tk,2,split}};
      bool failed=false;
      try{frontend::TorchExportImporter{}.InstantiateForGranularity(fixture,context,invalid);}
      catch(std::invalid_argument const&){failed=true;}
      assert(failed);++rejected;continue;
    }
    auto iterations=cp>=unsigned(tk)?9*((cp+tk-1)/tk):(9*cp+tk-1)/tk;
    if(iterations%split)continue;
    frontend::ImportOptions options;options.gemms={{16,16,tk,2,split}};
    options.phase_batch=batch;options.combiner_tile_per_block=true;
    auto module=frontend::TorchExportImporter{}.InstantiateForGranularity(fixture,context,options);
    assert(mlir::succeeded(mlir::verify(*module)));
    dialect::TileSpaceOp tile;
    for(auto candidate:module->getOps<dialect::TileSpaceOp>())
      if(candidate.getOperatorName()=="conv")tile=candidate;
    assert(tile && tile.getGranularity().get("reduction_index"));
    auto encoded=tile.getGranularity().getAs<mlir::StringAttr>("reduction_index").getValue().str();
    auto index=DecodeTaskReductionIndex(encoded);
    assert(index.capacity.IsLiteral(iterations) && index.issued_width.IsLiteral(tk));
    assert(EncodeTaskReductionIndex(index)==encoded);
    solver::ModelDims dims;dims.seq=dims.total=1;dims.batch=batch;
    auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"conv-partition");
    std::vector<solver::GemmConfig> configs{{16,16,tk,2,split}};
    auto graph=solver::InstantiateModelTasks(model,configs);auto const& node=*graph.Find("conv");
    auto accesses=dialect::HandoffTaskAccesses(tile);
    auto const& access=*node.element_access;
    auto write=ProjectTaskWrite(access.semantic,node,access.partition,
        access.semantic.result,access.semantic.result_map,{},{});
    assert(equal(accesses.writes.at(access.semantic.result.name),write));
    for(auto const& input:access.semantic.operands)
      assert(equal(accesses.reads.at(input.tensor.name),ProjectTaskRead(access.semantic,
          node,access.partition,input.tensor,input.map,{},{})));
    auto derived=solver::DeriveModelTaskInput(model,model.task_semantics.front(),graph,&configs.front(),false);
    assert(derived.serving_body_kind=="gemm_im2col");
    assert(derived.work.nominal_task_reduce_extent.SumDomain().Eval({})==
        long(iterations*tk*((9*batch+15)/16)));
    auto attrs=tile.getGranularity();mlir::NamedAttrList changed(attrs);
    index.capacity=F(0);
    changed.set("reduction_index",mlir::StringAttr::get(&context,EncodeTaskReductionIndex(index)));
    tile->setAttr("granularity",changed.getDictionary(&context));
    assert(mlir::failed(mlir::verify(*module)));++rejected;
    tile->setAttr("granularity",attrs);assert(mlir::succeeded(mlir::verify(*module)));
    ++cases;
  }
  std::cout<<"CONV_REDUCTION_CG cases="<<cases<<" rejected="<<rejected
      <<" lossless_geometry_handoff_and_pricing PASS\n";
  // Four filter positions issue eight K tiles although logical K would
  // suggest only six. Every layer of the solver must retain all eight chunks.
  auto target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+
      "/docs/experiments/DNN_MOE_R1/inputs/regression/llama_B1/prefill/target.json");
  solver::CostModel cost(target,solver::ScalarType::kBF16);
  for(unsigned batch:{1,2}) {
    auto fixture=Fixture(batch,24,2);
    frontend::ImportOptions options;options.gemms={{16,16,16,2,1}};
    options.phase_batch=batch;options.combiner_tile_per_block=true;
    auto module=frontend::TorchExportImporter{}.InstantiateForGranularity(fixture,context,options);
    solver::ModelDims dims;dims.seq=dims.total=1;dims.batch=batch;
    solver::SymbolicProblem base;base.model=solver::ModelDescription::FromCouplingGraph(*module,dims,"conv-flow");
    base.runtime=codegen::ReadRuntimePlan(*module);base.threads=128;
    CouplingCache cache;solver::FlowPreparationCache prepared;
    for(int split:{1,2,4,8,2}) {
      std::vector<solver::GemmConfig> geometry{{16,16,16,2,split}};
      assert(cost.Chunks(base.model,0,geometry.front())==split);
      assert(cost.Chunks(base.model.gemms.front(),geometry.front())==std::min(split,6));
      Granularity partition;
      auto graph=solver::InstantiateModelTasks(base.model,geometry,&partition);
      assert(partition.reduction_index.at("conv").capacity.IsLiteral(8));
      auto flow=solver::PrepareFlowStructure(base,geometry,3,4,cache,&prepared);
      int tiles=(9*batch+15)/16;
      assert(flow.counts.front()==tiles*split);
      assert(flow.counts.size()==(split>1?2u:1u));
      if(split>1) {
        assert(flow.counts.back()==tiles && flow.data_edges.size()==1);
        auto relation=flow.data_edges.front().relation;
        auto pairs=relation.Points();assert(pairs.size()==unsigned(tiles*split));
        for(auto const& [producer,consumer]:pairs)assert(producer.front()/split==consumer.front());
        auto windows=solver::BindRuntimeWindows(flow.projection,0,1,base.model.MetricBindings());
        for(int tile=0;tile<tiles;++tile)
          assert(solver::RuntimeReleaseEndpoint(-1,tile,tiles*split,windows,false)==(tile+1)*split-1);
      }
      auto projected=solver::ProjectRuntimeQueues(flow.model,flow.runtime,{3,128,4});
      assert(projected.runtime_task_refs.Eval(base.model.MetricBindings())==tiles*(split+(split>1)));
    }
  }
  std::cout<<"CONV_FLOW candidate_transitions=10 issued_chunk_projection PASS\n";
  return 0;
}
} // namespace tilemega::tests::conv_reduction_cg_test
