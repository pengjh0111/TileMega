// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Analysis/DramFloor.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/MLIRContext.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <stdexcept>

namespace tilemega::tests::dnn_epilogue_semantics_test {
namespace {
using namespace analysis;
using namespace frontend;
using namespace codegen;
using Points=std::set<std::pair<std::vector<long>,std::vector<long>>>;
ClosedForm C(long x) {return ClosedForm::Constant(x);}
DmBufferLayout Layout(DmWriteKind kind,unsigned columns=12) {
  DmBufferLayout l;l.rank=4;
  unsigned h=kind==DmWriteKind::kPixelShuffle?6:3;
  unsigned w=kind==DmWriteKind::kPixelShuffle?10:5;
  unsigned channels=kind==DmWriteKind::kPixelShuffle?columns/4:columns;
  if(kind==DmWriteKind::kNCHW) {
    unsigned shape[]={2,channels,h,w};
    std::copy(shape,shape+4,l.logical);std::copy(shape,shape+4,l.physical);
  }else {
    l.kind=DmLayout::kNHWC;
    unsigned logical[]={2,h,w,channels},physical[]={2,h+2,w+2,(channels+7)/8*8};
    std::copy(logical,logical+4,l.logical);std::copy(physical,physical+4,l.physical);
    l.halo_top=l.halo_bottom=l.halo_left=l.halo_right=1;
  }
  l.strides[3]=1;
  for(unsigned i=3;i-->0;)l.strides[i]=l.strides[i+1]*l.physical[i+1];
  return l;
}
ModelPlan Plan(DmWriteKind kind) {
  ModelPlan p;p.dm=p.forward=true;p.dtype="bf16";p.serving_seq=1;
  p.buffers.resize(7);
  for(unsigned i=0;i<7;++i)p.buffers[i].name="tensor"+std::to_string(i);
  p.buffers[2].layout=p.buffers[5].layout=Layout(kind);
  PlanGemm g;g.a=0;g.b=1;g.c=g.d=2;g.n=12;g.k=65;
  g.access.rows_per_batch=15;g.access.write={kind,kind==DmWriteKind::kPixelShuffle?2u:1u,2};
  g.chain.count=4;
  g.chain.operations[0].kind=DmEpilogueKind::kBias;g.chain.operations[0].parameter[0]=3;
  g.chain.operations[1].kind=DmEpilogueKind::kScale;g.chain.operations[1].parameter[0]=4;
  auto& residual=g.chain.operations[2];residual.kind=DmEpilogueKind::kResidual;
  residual.parameter[0]=5;residual.parameter[1]=6;residual.residual_map=g.access.write;
  residual.residual_map.layout=5;
  g.chain.operations[3].kind=DmEpilogueKind::kActivation;
  g.chain.operations[3].activation=DmActivation::kRelu6;
  p.gemms={g};PlanStage stage;stage.kind=PlanTaskKind::kGemm;p.stages={stage};return p;
}
std::vector<long> Owner(OperatorNode const& node,unsigned row,unsigned col,unsigned tm,unsigned tn) {
  std::vector<long> result;
  if(node.IsTiled(0))result.push_back(row/tm);
  if(node.IsTiled(1))result.push_back(col/tn);
  return result;
}
std::vector<long> Pixel(DmWriteKind kind,unsigned row,unsigned col) {
  unsigned image=row/15,y=(row%15)/5,x=row%5;
  if(kind==DmWriteKind::kNCHW)return {image,col,y,x};
  if(kind==DmWriteKind::kPixelShuffle)return {image,2*y+(col%4)/2+1,2*x+col%2+1,col/4};
  return {image,y+1,x+1,col};
}
void Equal(CouplingRelation const& actual,Points const& expected) {
  auto listed=actual.Points();assert(Points(listed.begin(),listed.end())==expected);
  auto tuple=[](std::vector<long> const& values) {
    std::string text="[";for(auto x:values){if(text.size()>1)text+=",";text+=std::to_string(x);}return text+"]";
  };
  std::string text="{ ";for(auto const& [a,b]:expected)text+=tuple(a)+" -> "+tuple(b)+"; ";
  auto oracle=CouplingRelation::FromIslText(text+" }");
  assert(Contains(actual,oracle) && Contains(oracle,actual));
}
template<class F> void Reject(F const& f) {
  bool rejected=false;try{f();}catch(std::invalid_argument const&){rejected=true;}assert(rejected);
}
void Emit(DmWriteKind kind,char const* output) {
  auto plan=Plan(kind);
  char const* names[]={"input","weight0","output","bias0","weight1","residual","bias1"};
  for(unsigned i=0;i<7;++i) {
    auto& buffer=plan.buffers[i];buffer.name=buffer.external_name=names[i];buffer.role="external";
    buffer.dtype=i==3 || i==6?"f32":"bf16";
  }
  plan.buffers[0].per_batch=15*72;plan.buffers[1].constant=plan.buffers[4].constant=20*72;
  plan.buffers[3].constant=plan.buffers[6].constant=20;
  for(auto id:{2,5}) {
    plan.buffers[id].layout=Layout(kind,20);
    plan.buffers[id].per_batch=plan.buffers[id].layout.strides[0];
  }
  auto first=plan.gemms.front();first.n=20;first.k=72;first.d=first.c=5;first.access.write.layout=5;
  first.chain.count=2;first.chain.operations[1].kind=DmEpilogueKind::kActivation;
  first.chain.operations[1].activation=DmActivation::kRelu6;first.chain.operations[1].parameter[0]=kDmNoIndex;
  auto second=first;second.b=4;second.c=second.d=2;second.access.write.layout=2;
  second.chain.count=3;second.chain.operations[0].parameter[0]=6;
  second.chain.operations[1].kind=DmEpilogueKind::kResidual;
  second.chain.operations[1].parameter[0]=5;second.chain.operations[1].residual_map=first.access.write;
  second.chain.operations[2]=first.chain.operations[1];
  plan.gemms={first,second};plan.stages.resize(2);plan.stages[1].gemm=1;
  for(unsigned i=0;i<2;++i) {
    plan.stages[i].representative="projection"+std::to_string(i);
    plan.stages[i].representative_index=5+i;
  }
  plan.outputs={{2,""}};
  auto node=[](int index,std::string name,char const* op,char const* target,
      std::initializer_list<llvm::json::Value> inputs,std::initializer_list<llvm::json::Value> shape,
      char const* dtype="torch.bfloat16") {
    return llvm::json::Object{{"index",index},{"name",name},{"op",op},{"target",target},
        {"inputs",llvm::json::Array(inputs)},{"shape",llvm::json::Array(shape)},{"dtype",dtype}};
  };
  llvm::json::Array nodes{
    node(0,"input","placeholder","input",{}, {"s0","15","72"}),
    node(1,"weight0","placeholder","weight0",{}, {"20","72"}),
    node(2,"bias0","placeholder","bias0",{}, {"20"},"torch.float32"),
    node(3,"weight1","placeholder","weight1",{}, {"20","72"}),
    node(4,"bias1","placeholder","bias1",{}, {"20"},"torch.float32"),
    node(5,"projection0","call_function","aten.linear.default",{"input","weight0","bias0"},{"s0","15","20"}),
    node(6,"projection1","call_function","aten.linear.default",{"input","weight1","bias1","projection0"},{"s0","15","20"})};
  llvm::json::Array inputs;
  for(auto name:{"input","weight0","bias0","weight1","bias1"})
    inputs.push_back(llvm::json::Object{{"name",name},{"kind",std::string(name)=="input"?"USER_INPUT":"PARAMETER"},{"target",name}});
  plan.node_buffer={{"input",0},{"weight0",1},{"bias0",3},{"weight1",4},{"bias1",6},
      {"projection0",5},{"projection1",2}};
  llvm::json::Value json=llvm::json::Object{{"schema","tilemega.exported_program.v1"},
      {"guards",llvm::json::Array{}},{"range_constraints",llvm::json::Object{{"s0","VR[1, 64]"}}},
      {"nodes",std::move(nodes)},{"signature",llvm::json::Object{{"inputs",std::move(inputs)},
      {"outputs",llvm::json::Array{llvm::json::Object{{"name","projection1"},{"kind","USER_OUTPUT"}}}}}}};
  int fd;llvm::SmallString<128> filename;
  assert(!llvm::sys::fs::createTemporaryFile("dm-epilogue-cg","json",fd,filename));
  {llvm::raw_fd_ostream stream(fd,true);stream<<llvm::formatv("{0:2}",json);}
  mlir::MLIRContext context;ImportOptions options;
  options.phase_batch=2;options.combiner_tile_per_block=true;
  options.gemms={{16,16,16,3,1},{16,16,16,3,3}};
  auto module=TorchExportImporter{}.ImportPlan(filename.str().str(),plan,context,nullptr,options);
  solver::ModelDims dims{1,0,1};dims.batch=2;
  auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"finite-epilogue");
  auto graph=solver::InstantiateModelTasks(model,{{16,16,16,3,1},{16,16,16,3,3}});
  auto const& partial=*graph.Find("dnn.s1");
  auto const& final=*graph.Find("dnn.s1.combine");
  assert(partial.operands.size()==2 && final.operands.size()==3);
  auto combined=solver::DeriveCombineTaskInput(model,1,{16,16,16,3,3},graph,128,true,true);
  assert(combined.work.write_elements.SumDomain().Eval(model.MetricBindings())==600);
  auto source=codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,1,1}});
  std::error_code error;llvm::raw_fd_ostream stream(output,error);assert(!error);stream<<source;
  assert(!llvm::sys::fs::remove(filename));
  std::cout<<"DNN_EPILOGUE_CG split_final_reads map="<<unsigned(kind)<<" PASS\n";
}
}
int TestDnnEpilogueSemantics(int argc,char** argv) {
  IslContext isl;LiftOptions options;options.forward=true;options.batch_symbol="B";options.static_seq=1;
  if(argc==3) {
    std::string flag=argv[1];
    if(flag=="--emit-dense") {Emit(DmWriteKind::kDense,argv[2]);return 0;}
    if(flag=="--emit-nchw") {Emit(DmWriteKind::kNCHW,argv[2]);return 0;}
    if(flag=="--emit-shuffle") {Emit(DmWriteKind::kPixelShuffle,argv[2]);return 0;}
  }
  ParamBinding known;known.Bind("B",2);unsigned cases=0;
  for(auto kind:{DmWriteKind::kDense,DmWriteKind::kNCHW,DmWriteKind::kPixelShuffle}) {
    auto plan=Plan(kind);auto lifted=LiftSemantics(plan,options);
    auto& op=lifted.sem.ops.front();assert(op.operands.size()==2 && op.epilogue_operands.size()==4);
    op.epilogue_operands[2].producer="residual.writer";
    auto producer=op;producer.name="residual.writer";producer.kind=OperatorKind::kPointwise;
    producer.arithmetic="mul";producer.reduction={};producer.operands.clear();producer.epilogue_operands.clear();
    producer.domain.erase(std::remove_if(producer.domain.begin(),producer.domain.end(),
        [](auto const& dim){return dim.type==IteratorType::kReduction;}),producer.domain.end());
    producer.result=op.epilogue_operands[2].tensor;producer.result_map=op.epilogue_operands[2].map;
    auto payload=EncodeSemanticOp(op);assert(EncodeSemanticOp(DecodeSemanticOp(payload))==payload);
    DramFloorOptions floor_options;floor_options.dram_gbps=100;floor_options.tc_gflops=100;
    floor_options.element_bytes={{"tensor3",4},{"tensor4",4},{"tensor6",4}};
    auto floor=DeriveDramFloor(SemanticGraph{{producer,op}},floor_options,known);
    for(auto name:{"tensor3","tensor4","tensor6"})
      assert(floor.tensors.at(name).read_bytes.Eval({})==48);
    assert(floor.tensors.at("tensor5").read_bytes.Eval({})==0);
    assert(!floor.tensors.at("tensor5").output);
    auto before=lifted.sem.Serialize();
    for(unsigned tm:{16,32})for(unsigned tn:{16,32})for(unsigned split:{1,3}) {
      auto g=LaunchGranularity(lifted,plan,{{int(tm),int(tn),16,3,int(split)}});
      g.Tile(producer.name,"m",C(tm)).Tile(producer.name,"n",C(tn));
      auto graph=Instantiate(SemanticGraph{{producer,op}},g);
      assert(lifted.sem.Serialize()==before);
      auto const& final=*graph.Find(split==1?op.name:op.reduction.combiner);
      auto const& access=*final.element_access;
      assert(access.semantic.epilogue_operands.empty());
      assert(final.operands.size()==(split==1?6:5));
      assert(final.operands[split==1?4:3].producer==producer.name);
      if(split>1) {
        auto const& partial=*graph.Find(op.name);
        assert(partial.operands.size()==2 && partial.element_access->semantic.epilogue_operands.empty());
        assert(final.operands.front().producer==op.name);
        solver::ModelDescription model;model.dm=true;model.dtype=solver::ScalarType::kBF16;
        model.dims.batch=2;
        model.metric_bindings.Bind("B",2);model.gemms={{12,65,0,2}};
        model.stages={{solver::StageKind::kGemm,0}};model.task_semantics={{op,{},0,false}};
        model.buffer_element_bytes={{"tensor3",4},{"tensor4",4},{"tensor5",2},{"tensor6",4}};
        auto combined=solver::DeriveCombineTaskInput(model,0,{int(tm),int(tn),16,3,3},graph,128,true,true);
        long count=(30+tm-1)/tm;
        assert(combined.work.write_elements.SumDomain().Eval(known)==360);
        assert(combined.physical_read_bytes->SumDomain().Eval(known)==360*3*4+360*2+count*12*3*4);
      }
      Points expected;
      for(unsigned row=0;row<30;++row)for(unsigned col=0;col<12;++col)
        expected.insert({Owner(final,row,col,tm,tn),Pixel(kind,row,col)});
      auto writes=ProjectTaskWrite(access.semantic,final,access.partition,
          access.semantic.result,access.semantic.result_map,{},known);
      Equal(writes,expected);
      auto const& residual=access.semantic.operands[split==1?4:3];
      auto reads=ProjectTaskRead(access.semantic,final,access.partition,
          residual.tensor,residual.map,{},known);
      Equal(reads,expected);
      for(unsigned operand=split==1?2:1;operand<final.operands.size();++operand) {
        if(operand==(split==1?4:3))continue;
        auto const& read=access.semantic.operands[operand];Points columns;
        for(unsigned row=0;row<30;++row)for(unsigned col=0;col<12;++col)
          columns.insert({Owner(final,row,col,tm,tn),{long(col)}});
        Equal(ProjectTaskRead(access.semantic,final,access.partition,read.tensor,read.map,{},known),columns);
      }
      auto edges=CouplingDerivation{}.Derive(graph,known);bool residual_edge=false;
      for(auto const& edge:edges)if(edge.src.name==producer.name) {
        assert(edge.dst.name==final.name);residual_edge=true;
      }
      assert(residual_edge);
      auto work=DeriveTaskWork(op,final,known);
      assert(work.write_elements.SumDomain().Eval(known)==360);++cases;
    }
    auto invalid=op;invalid.epilogue_operands.front().map.results={IndexResult::Dim("k")};
    Reject([&]{DecodeSemanticOp(EncodeSemanticOp(invalid));});
    Reject([&]{Instantiate(SemanticGraph{{invalid}},{});});
    invalid=op;invalid.element_reads={{op.operands[0].tensor,op.operands[0].map,{}}};
    Reject([&]{DecodeSemanticOp(EncodeSemanticOp(invalid));});
    Reject([&]{Instantiate(SemanticGraph{{invalid}},{});});
    plan.gemms[0].chain.side_count=1;
    Reject([&]{LiftSemantics(plan,options);});
  }
  auto legacy=Plan(DmWriteKind::kDense);legacy.gemms[0].chain={};
  auto text=EncodeSemanticOp(LiftSemantics(legacy,options).sem.ops.front());
  assert(text.find("epilogue_operands")==std::string::npos);
  std::cout<<"DNN_EPILOGUE exact_read_write_split_finalization cases="<<cases<<" PASS\n";
  return 0;
}
} // namespace tilemega::tests::dnn_epilogue_semantics_test
