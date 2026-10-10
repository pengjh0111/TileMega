// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DnnModelPlan.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Target/TargetSpec.h>
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
  options.workspace_budget_bytes=TargetSpec::FromJson(
      std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/sm_89.json").res.max_dynamic_smem_per_cta;
  auto plan=BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  assert(plan.dm && plan.forward && !plan.serving);
  assert(!plan.stages.empty() && (plan.stages.front().kind==PlanTaskKind::kLayoutConvert ||
                                plan.stages.front().kind==PlanTaskKind::kEmbeddingSum));
  assert(plan.outputs.size()==bridge.outputs.size());
  for(auto const& stage:plan.stages)assert(stage.kind!=PlanTaskKind::kElementwise);
  if(argc==1) {
    assert(plan.stages.size()==6 && plan.gemms.size()==3 && plan.convolutions.size()==3);
    assert(plan.gemms[0].chain.count==3 && plan.gemms[1].chain.count==2);
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
  if(plan.serving_seq!=1) {
    assert(plan.gemms.back().access.rows_per_batch==1);
    assert(plan.gemms.back().access.a_row_stride==unsigned(plan.serving_seq));
    auto reject=[&](ExportBridge const& bad,std::string const& diagnostic) {
      bool rejected=false;
      try{BuildDnnModelPlan(bad.nodes,bad.inputs,bad.outputs,options);}
      catch(std::invalid_argument const& e){rejected=std::string(e.what()).find(diagnostic)!=std::string::npos;}
      assert(rejected);
    };
    auto invalid=bridge;
    auto positions=std::find_if(invalid.nodes.begin(),invalid.nodes.end(),[](auto const& n){return n.immutable_buffer_value.present;});
    assert(positions!=invalid.nodes.end());positions->immutable_buffer_value.byte_order="big";
    reject(invalid,"immutable value proof");
    invalid=bridge;
    auto attention=std::find_if(invalid.nodes.begin(),invalid.nodes.end(),[](auto const& n){return n.target=="aten.scaled_dot_product_attention.default";});
    if(attention!=invalid.nodes.end()) {
      attention->args.resize(6);attention->args[5].kind=FxArgument::Kind::kBool;attention->args[5].boolean=true;
      reject(invalid,"noncausal");
      invalid=bridge;
      attention=std::find_if(invalid.nodes.begin(),invalid.nodes.end(),[](auto const& n){return n.target=="aten.scaled_dot_product_attention.default";});
      attention->kwargs["scale"]=F(.5);reject(invalid,"scale differs");
    }else {
      auto scaling=std::find_if(invalid.nodes.begin(),invalid.nodes.end(),[](auto const& n){
        return n.target=="aten.mul.Scalar" && n.args.size()>1 && n.args[1].kind==FxArgument::Kind::kFloat;});
      assert(scaling!=invalid.nodes.end());scaling->args[1]=F(.25);reject(invalid,"scaling differs");
      invalid=bridge;
      auto guard=std::find_if(invalid.nodes.begin(),invalid.nodes.end(),[](auto const& n){return n.target=="aten.full_like.default";});
      assert(guard!=invalid.nodes.end());guard->args[1]=I(1);reject(invalid,"fully masked");
    }
  }
  auto gated=std::count_if(plan.stages.begin(),plan.stages.end(),[](auto const& stage) {
    if(stage.kind!=PlanTaskKind::kDepthwiseConv)return false;
    for(unsigned i=0;i<stage.chain.count;++i)if(stage.chain.operations[i].kind==codegen::DmEpilogueKind::kGatePair)return true;
    return false;
  });
  if(gated==36 && plan.outputs.size()==1) {
    auto too_small=options;too_small.workspace_budget_bytes=options.workspace_budget_bytes/2;
    bool rejected=false;
    try{BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,too_small);}
    catch(std::invalid_argument const& e) {
      rejected=std::string(e.what()).find("workspace budget")!=std::string::npos;
    }
    assert(rejected);
    for(auto const& stage:plan.stages)if(stage.kind==PlanTaskKind::kDepthwiseConv) {
      auto const& c=plan.convolutions.at(stage.conv);
      auto bytes=std::uint64_t((stage.group-1)*c.stride_h+(c.r-1)*c.dilation_h+1)*
          plan.buffers.at(c.input_layout).layout.physical[2]*stage.width*4;
      assert(bytes<=options.workspace_budget_bytes);
    }
    assert(plan.stages.size()==335 && plan.gemms.size()==190 && plan.convolutions.size()==226);
    assert(std::count_if(plan.stages.begin(),plan.stages.end(),[](auto const& stage) {
      return stage.kind==PlanTaskKind::kLayerNorm;})==72);
    assert(std::count_if(plan.gemms.begin(),plan.gemms.end(),[](auto const& gemm) {
      return gemm.access.a_scale!=codegen::kDmNoIndex;})==36);
    assert(std::count_if(plan.gemms.begin(),plan.gemms.end(),[](auto const& gemm) {
      return gemm.access.write.kind==codegen::DmWriteKind::kPixelShuffle;})==4);
    assert(plan.gemms.back().access.write.kind==codegen::DmWriteKind::kNCHW);
    auto reject=[&](ExportBridge const& bad) {
      bool rejected=false;try{BuildDnnModelPlan(bad.nodes,bad.inputs,bad.outputs,options);}
      catch(std::invalid_argument const&){rejected=true;}assert(rejected);
    };
    auto invalid=bridge;
    auto split=std::find_if(invalid.nodes.begin(),invalid.nodes.end(),[](auto const& node) {
      return node.target=="aten.chunk.default" || node.target=="aten.split_with_sizes.default";
    });
    assert(split!=invalid.nodes.end());split->args[2]=I(2);reject(invalid);
    invalid=bridge;
    auto gate=std::find_if(invalid.nodes.begin(),invalid.nodes.end(),[](auto const& node) {
      return node.target=="<built-in function getitem>" && node.args.size()>1 && node.args[1].integer==1;
    });
    assert(gate!=invalid.nodes.end());invalid.outputs.push_back(gate->name);reject(invalid);
    invalid=bridge;
    auto scaled=std::find_if(invalid.inputs.begin(),invalid.inputs.end(),[](auto const& input) {
      return input.target.find(".beta")!=std::string::npos;
    });
    assert(scaled!=invalid.inputs.end());
    auto parameter=std::find_if(invalid.nodes.begin(),invalid.nodes.end(),[&](auto const& node) {
      return node.name==scaled->name;
    });
    assert(parameter!=invalid.nodes.end());parameter->shape[0]="2";reject(invalid);
  }
  if(plan.serving_seq==1 && plan.stages.back().kind==PlanTaskKind::kLayerNorm) {
    assert(plan.stages.size()==3 && plan.gemms.size()==1);
    assert(plan.buffers[plan.stages.back().operands[3]].layout.kind==codegen::DmLayout::kNHWC);
    auto reject=[&](ExportBridge const& bad) {
      bool rejected=false;try{BuildDnnModelPlan(bad.nodes,bad.inputs,bad.outputs,options);}
      catch(std::invalid_argument const&){rejected=true;}assert(rejected);
    };
    auto invalid=bridge;
    auto epsilon=std::find_if(invalid.nodes.begin(),invalid.nodes.end(),[](auto const& n) {
      return n.target=="aten.add.Tensor" && n.args.size()>1 && n.args[1].kind==FxArgument::Kind::kFloat;
    });
    assert(epsilon!=invalid.nodes.end());epsilon->args[1]=F(0);reject(invalid);
    invalid=bridge;auto mean=std::find_if(invalid.nodes.begin(),invalid.nodes.end(),[](auto const& n) {
      return n.target=="aten.mean.dim";
    });
    assert(mean!=invalid.nodes.end());auto mean_name=mean->name;mean->args[1]=L({2});reject(invalid);
    invalid=bridge;invalid.outputs.push_back(mean_name);reject(invalid);
  }
  LiftOptions lift;lift.forward=true;lift.static_seq=plan.serving_seq;lift.batch_symbol="B";
  for(auto const& input:bridge.inputs)if(input.kind=="USER_INPUT") {
    auto node=std::find_if(bridge.nodes.begin(),bridge.nodes.end(),[&](auto const& n){return n.name==input.name;});
    lift.batch_symbol=node->shape.at(0);break;
  }
  auto lifted=LiftDnnSemantics(plan,lift);assert(lifted.degraded.empty() && lifted.sem.ops.size()==plan.stages.size());
  if(argc>2) {
    ImportOptions imported;imported.phase_batch=2;imported.combiner_tile_per_block=true;
    mlir::MLIRContext context;
    auto module=TorchExportImporter{}.ImportPlan(argv[1],plan,context,nullptr,imported);
    assert(module);
    std::error_code ec;llvm::raw_fd_ostream output(argv[2],ec);assert(!ec);
    output<<codegen::CouplingGraphToCUDA{}.LowerVariants(
        {{*module,unsigned(plan.serving_seq),unsigned(plan.serving_seq)}});
  }
  std::cout<<"{\"passed\":true,\"stages\":"<<plan.stages.size()<<",\"gemms\":"<<plan.gemms.size()
      <<",\"convolutions\":"<<plan.convolutions.size()<<",\"scope\":\"DNN graph planning and semantic lifting\"}\n";
  return 0;
}
} // namespace tilemega::tests::dnn_model_plan_test
