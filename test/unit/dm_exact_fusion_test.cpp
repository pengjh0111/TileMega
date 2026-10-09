// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskOwnershipGeometry.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/FlowPreparation.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <tilemega/Dialect/CouplingGraph/FusionPass.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/Builders.h>
#include <cassert>
#include <iostream>
#include <set>

namespace tilemega::tests::dm_exact_fusion_test {
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
int TestDmExactFusion(int,char**) {
  using namespace tilemega;
  IslContext isl;mlir::MLIRContext context;
  unsigned cases=0,rejections=0;
  auto equal=[](auto const& a,auto const& b){return a.IsSubset(b)&&b.IsSubset(a);};
  for(int batch:{1,2,3})for(bool shuffle:{false,true}) {
    auto fixture=Fixture(batch,shuffle);
    if(batch==3 && shuffle) {
      // Independent element enumeration: output rows 112..127 of a TM=16
      // consumer span input rows on both sides of the producer's TM=32 cut.
      std::set<long> producer_rows;
      for(long m=112;m<128;++m) {
        long image=m/60,h=(m%60)/10,w=m%10;
        long input_row=image*15+(h/2)*5+w/2;
        producer_rows.insert(input_row/32);
      }
      assert(producer_rows==std::set<long>({0,1}));
      ImportOptions invalid;invalid.gemms={{32,64,16,2,1},{16,16,16,2,1}};
      invalid.phase_batch=batch;invalid.combiner_tile_per_block=true;
      auto module=TorchExportImporter{}.InstantiateForGranularity(fixture,context,invalid);
      solver::ModelDims dims;dims.seq=dims.total=1;dims.batch=batch;
      auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"crossed-fusion");
      bool rejected=false;
      try {(void)solver::DeriveLogicalFusionCandidate(model,
          {{32,64,16,2,1},{16,16,16,2,1}},"store","consume");}
      catch(std::invalid_argument const& e) {
        rejected=std::string(e.what())=="fusion tile constraint: consumer spans multiple producer tasks";
      }
      assert(rejected);++rejections;
      continue;
    }
    ImportOptions options;options.gemms={{32,shuffle?64:32,16,2,1},{16,16,16,2,1}};
    options.phase_batch=batch;options.combiner_tile_per_block=true;
    auto module=TorchExportImporter{}.InstantiateForGranularity(fixture,context,options);
    solver::ModelDims dims;dims.seq=dims.total=1;dims.batch=batch;
    auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"mapped-fusion");
    std::vector<solver::GemmConfig> geometry{{32,shuffle?64:32,16,2,1},{16,16,16,2,1}};
    auto candidate=solver::DeriveLogicalFusionCandidate(model,geometry,"store","consume");
    assert(candidate.producer.task.element_access && candidate.consumer.task.element_access);
    assert(candidate.producer.task.output.axes.size()==2 && candidate.producer_accesses.writes.at("mapped").RangeDimNames().size()==4);
    dialect::FuseTaskPair(*module,"store","consume");
    assert(mlir::succeeded(mlir::verify(*module)));
    auto inputs=solver::ReadFusedTaskInputs(*module);assert(inputs.size()==1);
    auto written=solver::DeriveWrittenFusionCandidate(inputs.front(),model,geometry);
    assert(equal(candidate.accesses.consumer_to_producer,written.accesses.consumer_to_producer));
    assert(candidate.accesses.task_count.SemanticallyEqual(inputs.front().task_count,{}));
    for(unsigned phase=0;phase<2;++phase) {
      auto const& expected=candidate.arithmetic.phases[phase];
      auto const& actual=inputs.front().arithmetic.phases[phase];
      assert(actual.output_elements.SemanticallyEqual(expected.output_elements,{}));
      assert(actual.arithmetic.flops_per_output_element.numerator.SemanticallyEqual(
          expected.arithmetic.flops_per_output_element.numerator,{}));
      assert(actual.arithmetic.flops_per_output_element.denominator==expected.arithmetic.flops_per_output_element.denominator);
    }
    // Independent GEMM arithmetic and tail geometry at every fused task.
    auto relation=candidate.accesses.consumer_to_producer;
    for (auto const& [consumer,producer]:relation.Points()) {
      ParamBinding point;
      auto names=relation.DomainDimNames();
      for(unsigned axis=0;axis<names.size();++axis)point.Bind(names[axis],consumer[axis]);
      long producer_row=candidate.producer.task.IsTiled(0)?producer.front():0;
      long consumer_row=candidate.consumer.task.IsTiled(0)?consumer.front():0;
      long consumer_column=candidate.consumer.task.IsTiled(1)?consumer.back():0;
      long producer_elements=std::min(32L,15L*batch-32*producer_row)*(shuffle?64:32);
      long consumer_elements=std::min(16L,(shuffle?60L:15L)*batch-16*consumer_row)*
          std::min(16L,48L-16*consumer_column);
      for(auto const* mixed:{&candidate.arithmetic,&inputs.front().arithmetic}) {
        assert(mixed->phases[0].output_elements.BindCoordinates(point).Eval({})==producer_elements);
        assert(mixed->phases[1].output_elements.BindCoordinates(point).Eval({})==consumer_elements);
        assert(mixed->phases[0].arithmetic.flops_per_output_element.numerator.BindCoordinates(point).Eval({})==128);
        assert(mixed->phases[1].arithmetic.flops_per_output_element.numerator.BindCoordinates(point).Eval({})==(shuffle?32:64));
      }
    }
    auto reject=[&](auto f){bool failed=false;try{f();}catch(std::invalid_argument const&){failed=true;}assert(failed);++rejections;};
    reject([&]{auto wrong=geometry;wrong[0].tile_m=16;(void)solver::DeriveWrittenFusionCandidate(inputs.front(),model,wrong);});
    auto fused=*module->getOps<dialect::FusedTileSpaceOp>().begin();
    auto saved=fused.getPhaseGranularities();mlir::OpBuilder b(&context);
    auto dictionaries=llvm::SmallVector<mlir::Attribute>(saved.begin(),saved.end());
    auto first=llvm::cast<mlir::DictionaryAttr>(dictionaries[0]);
    mlir::NamedAttrList attrs(first);attrs.erase("row");dictionaries[0]=attrs.getDictionary(&context);
    fused->setAttr("phase_granularities",b.getArrayAttr(dictionaries));
    assert(mlir::failed(mlir::verify(*module)));++rejections;
    fused->setAttr("phase_granularities",saved);
    ++cases;
  }
  assert(cases==5 && rejections==11);
  std::cout<<"DM exact fusion: 5 NCHW/pixel-shuffle written candidates, aliased ownership, phase arithmetic and 11 invalid geometries including an independently enumerated cross-tile shuffle PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_exact_fusion_test
