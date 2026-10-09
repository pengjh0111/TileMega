// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DnnModelPlan.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Analysis/ISLContext.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/MLIRContext.h>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace tilemega::tests::dnn_model_plan_test {
using namespace frontend;
FxArgument N(char const* name) {FxArgument a;a.kind=FxArgument::Kind::kNode;a.text=name;return a;}
FxArgument I(long value) {FxArgument a;a.kind=FxArgument::Kind::kInt;a.integer=value;return a;}
FxArgument F(double value) {FxArgument a;a.kind=FxArgument::Kind::kFloat;a.real=value;return a;}
FxArgument L(std::initializer_list<long> values) {
  FxArgument a;a.kind=FxArgument::Kind::kList;for(auto value:values)a.items.push_back(I(value));return a;
}
ExportBridge Fixture() {
  ExportBridge b;
  auto parameter=[&](char const* name,std::vector<std::string> shape) {
    FxNodeRecord n;n.name=n.target=name;n.op="placeholder";n.index=b.nodes.size();n.shape=shape;n.dtype="torch.bfloat16";
    b.nodes.push_back(n);b.inputs.push_back({name,"PARAMETER",std::string(name)+".weight"});
  };
  parameter("w0",{"8","3","3","3"});parameter("w1",{"8","8","1","1"});parameter("wf",{"10","8"});
  for(auto name:{"gamma","beta","mean","variance"})parameter(name,{"8"});parameter("bias",{"10"});
  FxNodeRecord input;input.name=input.target="x";input.op="placeholder";input.index=b.nodes.size();
  input.shape={"B","3","9","11"};input.dtype="torch.bfloat16";b.nodes.push_back(input);b.inputs.push_back({"x","USER_INPUT"});
  auto call=[&](char const* name,char const* op,std::vector<FxArgument> args) {
    FxNodeRecord n;n.name=name;n.target=std::string("aten.")+op+".default";n.op="call_function";
    n.index=b.nodes.size();n.has_arguments=true;n.args=std::move(args);
    for(auto const& a:n.args)if(a.kind==FxArgument::Kind::kNode)n.inputs.push_back(a.text);
    b.nodes.push_back(n);
  };
  call("conv0","conv2d",{N("x"),N("w0"),{},L({1,1}),L({1,1})});
  FxArgument no;no.kind=FxArgument::Kind::kBool;no.boolean=false;
  call("bn","batch_norm",{N("conv0"),N("gamma"),N("beta"),N("mean"),N("variance"),no,F(.1),F(1e-5)});
  call("relu","relu",{N("bn")});
  call("pool","max_pool2d",{N("relu"),L({3,3}),L({2,2}),L({1,1})});
  call("conv1","conv2d",{N("pool"),N("w1")});
  call("residual","add",{N("conv1"),N("pool")});
  call("relu1","relu",{N("residual")});
  call("average","mean",{N("relu1"),L({-1,-2})});
  call("output","linear",{N("average"),N("wf"),N("bias")});b.outputs={"output"};return b;
}
int TestDnnModelPlan(int argc,char** argv) {
  analysis::IslContext isl;
  auto bridge=argc>1?ReadExportBridge(argv[1]):Fixture();
  DnnPlanOptions options;options.batch=2;
  auto plan=BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  assert(plan.dm && plan.forward && !plan.serving && plan.serving_seq==1);
  assert(!plan.stages.empty() && plan.stages.front().kind==PlanTaskKind::kLayoutConvert);
  assert(plan.outputs.size()==bridge.outputs.size());
  for(auto const& stage:plan.stages)assert(stage.kind!=PlanTaskKind::kElementwise);
  if(argc==1) {
    assert(plan.stages.size()==6 && plan.gemms.size()==3 && plan.convolutions.size()==3);
    assert(plan.gemms[0].chain.count==2 && plan.gemms[1].chain.count==2);
    assert(plan.gemms[1].chain.side_count==1);
    assert(plan.buffers[plan.stages[4].operands[2]].dtype=="bf16");
    auto invalid=bridge;invalid.nodes.back().target="aten.unknown.default";
    bool rejected=false;try{BuildDnnModelPlan(invalid.nodes,invalid.inputs,invalid.outputs,options);}
    catch(std::invalid_argument const& e){rejected=std::string(e.what()).find("unsupported DNN target")!=std::string::npos;}
    assert(rejected);
    invalid=bridge;invalid.outputs.insert(invalid.outputs.begin(),"conv0");
    rejected=false;try{BuildDnnModelPlan(invalid.nodes,invalid.inputs,invalid.outputs,options);}
    catch(std::invalid_argument const& e){rejected=std::string(e.what()).find("observable use")!=std::string::npos;}
    assert(rejected);
  }
  LiftOptions lift;lift.forward=true;lift.static_seq=1;lift.batch_symbol=argc>1?"s77":"B";
  auto lifted=LiftDnnSemantics(plan,lift);assert(lifted.degraded.empty() && lifted.sem.ops.size()==plan.stages.size());
  if(argc>2) {
    ImportOptions imported;imported.phase_batch=2;imported.combiner_tile_per_block=true;
    mlir::MLIRContext context;
    auto module=TorchExportImporter{}.ImportPlan(argv[1],plan,context,nullptr,imported);
    assert(module);
    std::error_code ec;llvm::raw_fd_ostream output(argv[2],ec);assert(!ec);
    output<<codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,1,1}});
  }
  std::cout<<"{\"passed\":true,\"stages\":"<<plan.stages.size()<<",\"gemms\":"<<plan.gemms.size()
      <<",\"convolutions\":"<<plan.convolutions.size()<<",\"scope\":\"DNN graph planning and semantic lifting\"}\n";
  return 0;
}
} // namespace tilemega::tests::dnn_model_plan_test
