// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/FormatVariadic.h>
#include <cassert>
#include <fstream>
#include <iostream>

namespace tilemega::tests::dnn_memory_codegen_test {
int TestDnnMemoryCodegen(int argc,char** argv) {
  using namespace frontend;using namespace codegen;
  analysis::IslContext isl;ModelPlan plan;
  plan.dm=plan.forward=true;plan.dtype="bf16";plan.serving_seq=7;
  plan.memory_reuse="greedy";
  llvm::json::Array nodes,inputs;
  auto node=[&](std::string name,char const* op,std::vector<std::string> shape) {
    llvm::json::Array dimensions;for(auto const& extent:shape)dimensions.push_back(extent);
    nodes.push_back(llvm::json::Object{{"index",int(nodes.size())},{"name",name},{"op",op},
        {"target",std::string(op)=="placeholder"?name:"aten.linear.default"},
        {"inputs",llvm::json::Array{}},{"shape",std::move(dimensions)},{"dtype","torch.bfloat16"}});
  };
  auto matrix=[&](std::string name,unsigned width,bool external) {
    PlanBuffer b;b.name=name;b.per_batch=7*width;b.layout.rank=2;
    b.layout.logical[0]=b.layout.physical[0]=14;b.layout.logical[1]=b.layout.physical[1]=width;
    b.layout.strides[0]=width;b.layout.strides[1]=1;
    if(external){b.role="external";b.external_name=name;}
    unsigned id=plan.buffers.size();plan.buffers.push_back(b);plan.node_buffer[name]=id;return id;
  };
  auto input=matrix("reuse_input",64,true);node("reuse_input","placeholder",{"B","7","64"});
  inputs.push_back(llvm::json::Object{{"name","reuse_input"},{"kind","USER_INPUT"},{"target","reuse_input"}});
  unsigned previous=input,k=64;
  std::vector<GemmGranularity> geometry;
  for(unsigned index=0;index<7;++index) {
    // Alternating live allocations also leave a large/small/large slot history.
    unsigned n=index%3==1?32:64;
    std::string name=index==6?"reuse_output":"reuse_value"+std::to_string(index);
    auto output=matrix(name,n,index==6);
    PlanBuffer weight;weight.name=weight.external_name="reuse_weight"+std::to_string(index);
    weight.constant=n*k;weight.role="external";weight.source=PlanBuffer::Source::kWeight;
    weight.pack_json=llvm::formatv("{0}",llvm::json::Value(llvm::json::Object{
        {"kind","alias"},{"source",weight.name}})).str();
    auto w=unsigned(plan.buffers.size());plan.buffers.push_back(weight);
    node(weight.name,"placeholder",{std::to_string(n),std::to_string(k)});
    inputs.push_back(llvm::json::Object{{"name",weight.name},{"kind","PARAMETER"},{"target",weight.name}});
    PlanGemm gemm;gemm.n=n;gemm.k=k;gemm.a=previous;gemm.b=w;gemm.c=gemm.d=output;
    gemm.access.rows_per_batch=7;gemm.access.write.layout=output;
    gemm.chain.count=1;gemm.chain.operations[0].kind=DmEpilogueKind::kActivation;
    gemm.chain.operations[0].activation=DmActivation::kRelu;
    PlanStage stage;stage.gemm=index;stage.operands.fill(kDmNoIndex);
    stage.representative=name;stage.representative_index=nodes.size();
    node(name,"call_function",{"B","7",std::to_string(n)});
    plan.gemms.push_back(gemm);plan.stages.push_back(stage);
    geometry.push_back({16,16,16,2,1});previous=output;k=n;
  }
  plan.outputs={{previous,{}}};
  llvm::json::Value bridge=llvm::json::Object{{"schema","tilemega.exported_program.v1"},
      {"guards",llvm::json::Array{}},{"range_constraints",llvm::json::Object{{"B","VR[1, 64]"}}},
      {"nodes",std::move(nodes)},{"signature",llvm::json::Object{{"inputs",std::move(inputs)},
      {"outputs",llvm::json::Array{llvm::json::Object{{"name","reuse_output"},{"kind","USER_OUTPUT"}}}}}}};
  if(argc!=3 || std::string(argv[1])!="--emit")
    throw std::invalid_argument("dnn_memory_codegen requires --emit OUTPUT.cu");
  auto path=std::string(argv[2])+".bridge.json";
  {std::ofstream stream(path);stream<<llvm::formatv("{0:2}",bridge).str();}
  mlir::MLIRContext context;ImportOptions options;options.phase_batch=2;options.gemms=geometry;
  auto module=TorchExportImporter{}.ImportPlan(path,plan,context,nullptr,options);
  auto stored=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  assert(stored.getAs<mlir::IntegerAttr>("dm_memory_arena_bytes").getInt()>0);
  auto hazards=(*module)->getAttrOfType<mlir::IntegerAttr>("tilemega.memory_hazard_count").getInt();
  assert(hazards>0);
  auto source=CouplingGraphToCUDA{}.LowerVariants({{*module,7,7}});
  std::ofstream output(argv[2]);output<<source;assert(output);
  std::cout<<"Reusable storage generated CUDA: "<<hazards<<" exact anti-dependencies PASS\n";
  return 0;
}
}
