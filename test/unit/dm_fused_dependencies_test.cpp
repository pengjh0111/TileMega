// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskOwnershipGeometry.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/FlowPreparation.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <tilemega/Dialect/CouplingGraph/FusionPass.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Dialect/CouplingGraph/BoundDependency.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/Builders.h>
#include <cassert>
#include <iostream>
#include <set>

namespace tilemega::tests::dm_fused_dependencies_test {
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
  auto source=operation("source",rows,64,64);
  source.result={"source_out",{{"m",F(rows)},{"n",F(64)}}};
  source.result_map.results={IndexResult::Affine({{"m",F(1),F(1),F(16)},
      {"m",F(-rows),F(rows),F(16)}}),I("n")};
  source.operands=store.operands;
  source.operands[1].tensor={"source_weight",{{"n",F(64)},{"k",F(64)}}};
  store.operands[0]={"source",source.result,{{I("m"),I("k")}}, {}};
  auto sink=operation("sink",consumer_rows,16,48);
  sink.result={"sink_out",{{"m",F(consumer_rows)},{"n",F(16)}}};
  sink.result_map.results={I("m"),I("n")};
  sink.operands={{"consume",consume.result,{{I("m"),I("k")}},{}},
      {"",{"sink_weight",{{"n",F(16)},{"k",F(48)}}},{{I("n"),I("k")}}, {}}};
  plan.buffers.resize(9);
  for(unsigned i=5;i<9;++i)plan.buffers[i].name="buffer"+std::to_string(i);
  plan.buffers[5].constant=64*64;plan.buffers[6].per_batch=15*64;
  plan.buffers[7].constant=16*48;plan.buffers[8].per_batch=height*width*16;
  auto source_gemm=p;source_gemm.n=source_gemm.k=64;source_gemm.b=5;source_gemm.d=6;
  auto store_gemm=p;store_gemm.a=6;
  auto sink_gemm=c;sink_gemm.n=16;sink_gemm.k=48;sink_gemm.a=4;sink_gemm.b=7;sink_gemm.d=8;
  plan.gemms={source_gemm,store_gemm,c,sink_gemm};plan.stages.clear();
  fixture.bridge.nodes.resize(1);fixture.bridge.tasks.clear();
  for(unsigned stage=0;stage<4;++stage) {
    auto name=std::vector<std::string>{"source","store","consume","sink"}[stage];
    PlanStage item;item.gemm=stage;item.representative=name;item.representative_index=stage+1;
    plan.stages.push_back(item);
    FxNodeRecord node;node.name=name;node.index=stage+1;node.op="call_function";
    node.target="aten.linear.default";node.shape={"B",std::to_string(stage<2?15:height*width),
        std::to_string(plan.gemms[stage].n)};
    fixture.bridge.nodes.push_back(node);fixture.bridge.tasks.push_back(node);
  }
  fixture.bridge.outputs={"sink"};plan.outputs={{8,""}};
  fixture.lifted.sem.ops={source,store,consume,sink};fixture.lifted.has_plan=true;
  fixture.lifted.ops={{"source",OpRole::kProjection,OwnershipKind::kTilePerBlock,0,0,"source"},
      {"store",OpRole::kProjection,OwnershipKind::kTilePerBlock,1,0,"store"},
      {"consume",OpRole::kProjection,OwnershipKind::kTilePerBlock,2,0,"consume"},
      {"sink",OpRole::kProjection,OwnershipKind::kTilePerBlock,3,0,"sink"}};
  fixture.lifted.written={0,0,0,1,1};return fixture;
}
}
int TestDmFusedDependencies(int,char**) {
  using namespace tilemega;
  IslContext isl;mlir::MLIRContext context;
  unsigned cases=0,rejections=0,tables=0,external_edges=0;
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
      ImportOptions invalid;invalid.gemms={{16,16,16,2,1},{32,64,16,2,1},{16,16,16,2,1},{16,16,16,2,1}};
      invalid.phase_batch=batch;invalid.combiner_tile_per_block=true;
      auto module=TorchExportImporter{}.InstantiateForGranularity(fixture,context,invalid);
      solver::ModelDims dims;dims.seq=dims.total=1;dims.batch=batch;
      auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"crossed-fusion");
      bool rejected=false;
      try {(void)solver::DeriveLogicalFusionCandidate(model,
          {{16,16,16,2,1},{32,64,16,2,1},{16,16,16,2,1},{16,16,16,2,1}},"store","consume");}
      catch(std::invalid_argument const& e) {
        rejected=std::string(e.what())=="fusion tile constraint: consumer spans multiple producer tasks";
      }
      assert(rejected);++rejections;
      continue;
    }
    ImportOptions options;options.gemms={{16,16,16,2,1},{32,shuffle?64:32,16,2,1},{16,16,16,2,1},{16,16,16,2,1}};
    options.phase_batch=batch;options.combiner_tile_per_block=true;
    auto module=TorchExportImporter{}.InstantiateForGranularity(fixture,context,options);
    solver::ModelDims dims;dims.seq=dims.total=1;dims.batch=batch;
    auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"mapped-fusion");
    std::vector<solver::GemmConfig> geometry{{16,16,16,2,1},{32,shuffle?64:32,16,2,1},{16,16,16,2,1},{16,16,16,2,1}};
    auto candidate=solver::DeriveLogicalFusionCandidate(model,geometry,"store","consume");
    assert(candidate.producer.task.element_access && candidate.consumer.task.element_access);
    assert(candidate.producer.task.output.axes.size()==2 && candidate.producer_accesses.writes.at("mapped").RangeDimNames().size()==4);
    struct Snapshot {std::string name,src,dst;CouplingRelation relation,shared,reads;};
    std::vector<Snapshot> original;
    std::string ps,cs;
    for(auto task:module->getOps<dialect::TileSpaceOp>()) {
      if(task.getOperatorName()=="store")ps=task.getSymName().str();
      if(task.getOperatorName()=="consume")cs=task.getSymName().str();
    }
    for(auto edge:module->getOps<dialect::CouplingOp>()) {
      assert(dialect::ReadBoundTaskGeometry(edge,{}));
      original.push_back({edge.getSymName().str(),edge.getSrc().str(),edge.getDst().str(),
          edge.getRelation().getMap(),edge->getAttrOfType<dialect::CouplingMapAttr>("shared_elements").getMap(),
          edge->getAttrOfType<dialect::CouplingMapAttr>("coupled_reads").getMap()});
    }
    dialect::FuseTaskPair(*module,"store","consume");
    auto mapping=candidate.accesses.consumer_to_producer;
    auto mapped=mapping.Points();
    auto rebase=[&](std::vector<long> const& old,std::string const& endpoint) {
      std::vector<std::vector<long>> result;
      if(endpoint!=ps)return std::vector<std::vector<long>>{old};
      for(auto const& pair:mapped)if(pair.second==old)result.push_back(pair.first);
      return result;
    };
    for(auto const& old:original) {
      if(old.src==ps && old.dst==cs)continue;
      auto edge=mlir::dyn_cast<dialect::CouplingOp>(mlir::SymbolTable::lookupSymbolIn(*module,old.name));
      assert(edge);
      auto expected=old.relation;
      if(old.dst==ps)expected=mapping.ApplyRange(expected);
      if(old.src==ps)expected=expected.ApplyRange(mapping.Reverse());
      assert(equal(expected,edge.getRelation().getMap()));
      using Element=std::pair<std::vector<long>,std::vector<long>>;
      std::set<Element> shared,reads;
      unsigned consumer_rank=old.relation.DomainDimNames().size();
      for(auto const& [pair,element]:old.shared.Points()) {
        std::vector<long> c(pair.begin(),pair.begin()+consumer_rank),p(pair.begin()+consumer_rank,pair.end());
        for(auto const& nc:rebase(c,old.dst))for(auto const& np:rebase(p,old.src)) {
          auto coordinates=nc;coordinates.insert(coordinates.end(),np.begin(),np.end());
          shared.insert({coordinates,element});
        }
      }
      for(auto const& [consumer,element]:old.reads.Points())
        for(auto const& nc:rebase(consumer,old.dst))reads.insert({nc,element});
      auto points=edge->getAttrOfType<dialect::CouplingMapAttr>("shared_elements").getMap().Points();
      assert(shared==std::set<Element>(points.begin(),points.end()));
      points=edge->getAttrOfType<dialect::CouplingMapAttr>("coupled_reads").getMap().Points();
      assert(reads==std::set<Element>(points.begin(),points.end()));
      auto geometry=dialect::ReadBoundTaskGeometry(edge,{});assert(geometry);
      if(dialect::ReadBoundDependencyTable(edge,{}))++tables;
      auto event=mlir::dyn_cast<dialect::EventTensorOp>(mlir::SymbolTable::lookupSymbolIn(*module,edge.getEvent()));
      assert(event && event.getExtent().getValue().Eval({})==geometry->producers);
      ++external_edges;
    }
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
    reject([&]{auto wrong=geometry;wrong[1].tile_m=16;(void)solver::DeriveWrittenFusionCandidate(inputs.front(),model,wrong);});
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
  assert(cases==5 && rejections==11 && external_edges==10 && tables>=1);
  std::cout<<"DM fused dependencies: "<<external_edges<<" external edges, "<<tables<<" exact tables, phase-coordinate element enumeration and full event storage PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_fused_dependencies_test
