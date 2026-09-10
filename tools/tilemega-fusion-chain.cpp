// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/FusionPass.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <mlir/IR/MLIRContext.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <filesystem>
#include <iostream>
#include <limits>

int main(int argc,char** argv) try {
  using namespace tilemega;
  using namespace frontend;
  if (argc!=5) throw std::invalid_argument("usage: tilemega-fusion-chain EXPORT add|norm TILE_M OUTDIR");
  analysis::IslContext isl;
  auto bridge=ReadExportBridge(argv[1]);
  auto source=BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
  auto found=std::find_if(source.gemms.begin(),source.gemms.end(),[](auto const& g) { return g.beta==1; });
  if (found==source.gemms.end()) throw std::invalid_argument("source model has no residual projection");
  std::string mode=argv[2];
  if (mode!="add" && mode!="norm") throw std::invalid_argument("unsupported chain consumer");
  int tile_m=std::stoi(argv[3]);
  if (tile_m<=0) throw std::invalid_argument("positive tile M required");
  ModelPlan plan;
  plan.dtype=source.dtype;
  auto buffer=[&](std::string name,unsigned constant,unsigned per_seq,bool input) {
    PlanBuffer b; b.name=name; b.constant=constant; b.per_seq=per_seq;
    b.source=input ? PlanBuffer::Source::kFixture : PlanBuffer::Source::kZero;
    b.file=input ? name+".bin" : "";
    plan.buffers.push_back(b);
  };
  buffer("input",0,found->k,true);
  buffer("weight",found->n*found->k,0,true);
  buffer("intermediate",0,found->n,false);
  buffer("consumer_input",mode=="norm" ? found->n : 0,mode=="add" ? found->n : 0,true);
  buffer("output",0,found->n,false);
  plan.gemms.push_back({found->n,found->k,0,1,2,2,0});
  PlanStage producer; producer.kind=PlanTaskKind::kGemm;
  producer.representative="chain_gemm"; producer.operands.fill(std::numeric_limits<unsigned>::max());
  PlanStage consumer; consumer.kind=mode=="add" ? PlanTaskKind::kAdd : PlanTaskKind::kRMSNorm;
  consumer.representative="chain_consumer"; consumer.width=consumer.extent=found->n;
  consumer.operands.fill(std::numeric_limits<unsigned>::max());
  consumer.operands[0]=2; consumer.operands[1]=3; consumer.operands[2]=4;
  plan.stages={producer,consumer}; plan.outputs.push_back({4,"reference.bin"});
  ImportOptions options;
  options.gemms.push_back({tile_m,mode=="norm" ? int(found->n) : GemmGranularity{}.tile_n,
                           GemmGranularity{}.tile_k,GemmGranularity{}.stages,1});
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  auto module=TorchExportImporter{}.ImportPlan(argv[1],plan,context,nullptr,options);
  auto description=solver::ModelDescription::FromCouplingGraph(*module,{4,3,7},argv[1]);
  if (description.task_semantics.size()!=2) throw std::runtime_error("calibration chain is not two tasks");
  std::filesystem::path out(argv[4]);
  if (!std::filesystem::create_directory(out)) throw std::invalid_argument("output directory already exists");
  auto write=[&](std::string const& name,std::string const& text) {
    std::error_code error; llvm::raw_fd_ostream stream((out/name).string(),error,llvm::sys::fs::CD_CreateNew);
    if (error) throw std::runtime_error(error.message());
    stream<<text; stream.flush();
    if (stream.has_error()) throw std::runtime_error("chain output write failed");
  };
  auto emit=[&](std::string const& name) {
    std::string text; llvm::raw_string_ostream stream(text); module->print(stream);
    write(name+".mlir",text);
    write(name+".cu",codegen::CouplingGraphToCUDA{}.Lower(*module));
  };
  emit("separate");
  dialect::FuseTaskPair(*module,description.task_semantics[0].op.name,description.task_semantics[1].op.name);
  emit("fused");
  write("shape.json","{\"n\":"+std::to_string(found->n)+",\"k\":"+std::to_string(found->k)+
      ",\"tile_m\":"+std::to_string(tile_m)+",\"consumer\":\""+mode+"\",\"dtype\":\""+plan.dtype+"\"}\n");
  if (isl.ReferenceCount()) throw std::runtime_error("chain generation retained ISL objects");
  std::cout<<"FUSION_CHAIN consumer="<<mode<<" n="<<found->n<<" k="<<found->k
           <<" replacement_cg_verified=1 remaining=0\n";
} catch (std::exception const& error) { std::cerr<<error.what()<<'\n'; return 2; }
