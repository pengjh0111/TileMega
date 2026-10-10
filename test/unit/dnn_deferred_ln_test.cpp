// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DnnDeferredNorm.h>
#include <tilemega/Frontend/DnnModelPlan.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskStorage.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/FormatVariadic.h>
#include <fstream>
#include <iostream>
#include <cassert>

namespace tilemega::tests::dnn_deferred_ln_test {
namespace {
using namespace frontend;using namespace codegen;
ModelPlan Fixture() {
  ModelPlan p;p.dm=p.forward=true;p.dtype="bf16";p.serving_seq=5;
  auto matrix=[&](char const* name,unsigned width,bool external=false) {
    PlanBuffer b;b.name=name;b.per_batch=5*width;b.layout.rank=2;
    b.layout.logical[0]=b.layout.physical[0]=10;b.layout.logical[1]=b.layout.physical[1]=width;
    b.layout.strides[0]=width;b.layout.strides[1]=1;
    if(external){b.role="external";b.external_name=name;}
    p.buffers.push_back(b);return unsigned(p.buffers.size()-1);
  };
  auto parameter=[&](char const* name,unsigned count,char const* dtype="bf16") {
    PlanBuffer b;b.name=b.external_name=name;b.constant=count;b.dtype=dtype;
    b.source=PlanBuffer::Source::kWeight;b.role="external";
    b.pack_json=llvm::formatv("{0}",llvm::json::Value(llvm::json::Object{
        {"kind","alias"},{"source",name}})).str();
    p.buffers.push_back(b);return unsigned(p.buffers.size()-1);
  };
  auto x=matrix("deferred_ln_input",32,true),raw=matrix("raw",32),normalized=matrix("normalized",32);
  auto expanded=matrix("expanded",64),out=matrix("deferred_ln_output",32,true);
  auto gamma=parameter("gamma",32),beta=parameter("beta",32);
  for(unsigned index=0;index<3;++index) {
    unsigned n=index==1?64:32,k=index==2?64:32;
    auto w=parameter(("w"+std::to_string(index)).c_str(),n*k);
    auto bias=parameter(("b"+std::to_string(index)).c_str(),n,"f32");
    PlanGemm g;g.n=n;g.k=k;g.a=index==0?x:index==1?normalized:expanded;
    g.b=w;g.c=g.d=index==0?raw:index==1?expanded:out;
    g.access.rows_per_batch=5;g.access.write.layout=g.d;
    g.chain.count=1;g.chain.operations[0].parameter[0]=bias;
    if(index==2) {
      g.chain.count=2;auto& r=g.chain.operations[1];r.kind=DmEpilogueKind::kResidual;
      r.parameter[0]=normalized;r.residual_map.layout=normalized;
    }
    p.gemms.push_back(g);PlanStage s;s.gemm=index;s.operands.fill(kDmNoIndex);
    s.representative=index==0?"raw":index==1?"expanded":"deferred_ln_output";
    s.representative_index=10+index;p.stages.push_back(s);
    if(index==0) {
      PlanStage norm;norm.kind=PlanTaskKind::kLayerNorm;norm.width=32;norm.group=4;
      norm.rows_per_batch=5;norm.norm_epsilon=1e-12f;norm.operands.fill(kDmNoIndex);
      norm.operands[0]=raw;norm.operands[1]=gamma;norm.operands[2]=beta;norm.operands[3]=normalized;
      norm.representative="normalized";norm.representative_index=9;p.stages.push_back(norm);
    }
  }
  p.node_buffer={{"deferred_ln_input",x},{"raw",raw},{"normalized",normalized},
      {"expanded",expanded},{"deferred_ln_output",out}};
  p.outputs={{out,{}}};return p;
}
}
int TestDnnDeferredLN(int argc,char** argv) {
  analysis::IslContext isl;auto plan=Fixture();
  assert(ApplyDnnDeferredLayerNorm(plan)==1 && plan.stages.size()==3);
  auto const& norm=plan.gemms[1].chain.operations[0];
  auto const& residual=plan.gemms[2].chain.operations[1];
  assert(norm.kind==DmEpilogueKind::kDeferredLayerNorm && norm.norm_width==32 &&
      norm.norm_epsilon==1e-12f && residual.kind==DmEpilogueKind::kResidualLN &&
      residual.norm_width==32 && plan.gemms[2].k==64 && residual.norm_epsilon==1e-12f);
  auto observed=Fixture();observed.outputs.push_back({2,{}});
  assert(!ApplyDnnDeferredLayerNorm(observed) && observed.stages.size()==4);
  auto strided=Fixture();strided.gemms[1].access.a_row_stride=2;
  assert(!ApplyDnnDeferredLayerNorm(strided));
  auto alias=Fixture();alias.gemms[2].chain.operations[1].parameter[1]=5;
  assert(!ApplyDnnDeferredLayerNorm(alias));
  LiftOptions options;options.forward=true;options.batch_symbol="B";options.static_seq=5;
  auto lifted=LiftDnnSemantics(plan,options);
  auto g=LaunchGranularity(lifted,plan,{{16,16,16,2,1},{16,32,16,2,1},{16,16,16,2,1}});
  auto concrete=analysis::MaterializeTaskStorage(lifted.sem,g);
  assert(concrete.ops.size()==3 && concrete.ops[0].additional_writes.size()==2);
  for(unsigned consumer:{1u,2u}) {
    unsigned stats=0;
    for(auto const& read:concrete.ops[consumer].epilogue_operands)
      if(read.tensor.name==plan.buffers[norm.parameter[0]].name) {
        assert(read.producer==concrete.ops[0].name && read.tensor.axes.size()==3);
        ++stats;
      }
    assert(stats==2);
  }
  if(argc==3 && std::string(argv[1])=="--emit") {
    llvm::json::Array nodes,inputs;
    auto node=[&](std::string const& name,char const* op,std::vector<std::string> shape,
                  char const* dtype="torch.bfloat16") {
      llvm::json::Array dimensions;for(auto const& extent:shape)dimensions.push_back(extent);
      nodes.push_back(llvm::json::Object{{"index",int(nodes.size())},{"name",name},{"op",op},
          {"target",std::string(op)=="placeholder"?name:"aten.linear.default"},
          {"inputs",llvm::json::Array{}},{"shape",std::move(dimensions)},{"dtype",dtype}});
    };
    for(auto const& buffer:plan.buffers)if(buffer.role=="external") {
      bool input=buffer.name=="deferred_ln_input";
      if(!input && buffer.pack_json.empty())continue;
      node(buffer.name,"placeholder",input?std::vector<std::string>{"B","5","32"}:
          std::vector<std::string>{std::to_string(buffer.constant)},
          buffer.dtype=="f32"?"torch.float32":"torch.bfloat16");
      inputs.push_back(llvm::json::Object{{"name",buffer.name},{"kind",input?"USER_INPUT":"PARAMETER"},
          {"target",buffer.name}});
    }
    for(auto& stage:plan.stages) {
      stage.representative_index=nodes.size();node(stage.representative,"call_function",
          {"B","5",std::to_string(plan.gemms[stage.gemm].n)});
    }
    llvm::json::Value bridge=llvm::json::Object{{"schema","tilemega.exported_program.v1"},
        {"guards",llvm::json::Array{}},{"range_constraints",llvm::json::Object{{"B","VR[1, 64]"}}},
        {"nodes",std::move(nodes)},{"signature",llvm::json::Object{{"inputs",std::move(inputs)},
        {"outputs",llvm::json::Array{llvm::json::Object{{"name","deferred_ln_output"},{"kind","USER_OUTPUT"}}}}}}};
    auto path=std::string(argv[2])+".bridge.json";
    {std::ofstream stream(path);stream<<llvm::formatv("{0:2}",bridge).str();}
    mlir::MLIRContext context;ImportOptions geometry;geometry.phase_batch=2;
    geometry.gemms={{16,16,16,2,1},{16,32,16,2,1},{16,16,16,2,1}};
    auto module=TorchExportImporter{}.ImportPlan(path,plan,context,nullptr,geometry);
    auto cu=CouplingGraphToCUDA{}.LowerVariants({{*module,5,5}});
    std::ofstream stream(argv[2]);stream<<cu;assert(stream);
    std::cout<<"Deferred LN generated CUDA with K=64 residual normalization width=32 PASS\n";
  }
  if(argc==2) {
    auto bridge=ReadExportBridge(argv[1]);DnnPlanOptions dnn;dnn.batch=2;
    auto upstream=BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,dnn);
    auto count=ApplyDnnDeferredLayerNorm(upstream);
    auto model_options=options;model_options.static_seq=upstream.serving_seq;
    auto sem=LiftDnnSemantics(upstream,model_options);
    assert(sem.sem.ops.size()==upstream.stages.size());
    std::cout<<"Upstream deferred LN: "<<count<<" stages elided, "<<sem.sem.ops.size()<<" semantic stages PASS\n";
  }
  std::cout<<"Deferred LN: complete row proof, observed/strided/scaled exclusions and unequal residual K PASS\n";
  return 0;
}
}
