#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Support/Json.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void Write(std::filesystem::path const& path,std::string const& text) {
  std::ofstream out(path);out<<text<<'\n';if(!out)throw std::runtime_error("Cannot write "+path.string());
}
int main(int argc,char** argv) {
  try {
    if(argc!=5 || std::string(argv[1])!="inspect")
      throw std::invalid_argument("usage: as2_compile_llm inspect exported_directory new_output_directory batch");
    namespace fs=std::filesystem;
    auto root=fs::path("/home/d50064638/junhuip/TileMega/runs/as2");
    auto input=fs::weakly_canonical(argv[2]),output=fs::weakly_canonical(argv[3]);
    auto controlled=[&](auto const& p){auto r=p.lexically_relative(root);return p!=root && !r.empty() && *r.begin()!="..";};
    if(!controlled(input)||!controlled(output)||fs::exists(output))
      throw std::invalid_argument("Use existing AS-2 exports and a new AS-2 output directory");
    int batch=std::stoi(argv[4]);if(batch!=1 && batch!=4 && batch!=16)throw std::invalid_argument("Unsupported batch");
    auto receipt=tilemega::json::ParseFile((input/"as2_export.json").string());
    auto phase=receipt.At("phase").AsString("phase");
    if(phase!="prefill" && phase!="decode")throw std::invalid_argument("Unknown phase");
    tilemega::frontend::ServingOptions serving;
    serving.phase=phase=="prefill"?decltype(serving.phase)::kPrefill:decltype(serving.phase)::kDecode;
    serving.seq=phase=="prefill"?64:1;serving.capacity=1088;
    serving.deferred_norm=false;serving.kv_block=128;serving.argmax_tile_n=64;
    auto start=std::chrono::steady_clock::now();
    tilemega::analysis::IslContext isl;
    auto bridge=tilemega::frontend::ReadExportBridge((input/"export.json").string());
    auto plan=tilemega::frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,serving);
    tilemega::frontend::ImportOptions options;options.phase_batch=batch;
    options.rope_tile_per_block=options.kv_tile_per_block=options.activation_tile_per_block=options.combiner_tile_per_block=true;
    options.gemms.assign(plan.gemms.size(),{32,64,64,2,1});
    mlir::MLIRContext context;tilemega::frontend::ImportSummary summary;
    tilemega::frontend::TorchExportImporter importer;
    auto semantic=importer.ImportSemantics((input/"export.json").string(),plan,context);
    if(!semantic.lifted.degraded.empty())throw std::invalid_argument("Serving semantics contain degraded operators");
    for(auto const& op:semantic.lifted.ops)
      if(op.role==tilemega::frontend::OpRole::kGeneric)
        throw std::invalid_argument("Generic semantics cannot certify an imported LLM");
    auto module=importer.InstantiateForGranularity(semantic,context,options,nullptr,&summary);
    if(!module || mlir::failed(mlir::verify(*module)))
      throw std::runtime_error("LLM import verification failed");
    tilemega::solver::ModelDims dims;dims.batch=batch;dims.seq=serving.seq;
    dims.past=phase=="prefill"?0:64;dims.total=dims.seq+dims.past;
    auto model=tilemega::solver::ModelDescription::FromCouplingGraph(*module,dims,"as2-llm-inspection");
    fs::create_directories(output);
    std::string ir;{llvm::raw_string_ostream stream(ir);module->print(stream);}Write(output/"ctg.mlir",ir);
    tilemega::json::Array stages;
    tilemega::json::Array raw_unclassified;
    for(auto const& name:bridge.unsupported)raw_unclassified.emplace_back(name);
    for(unsigned i=0;i<model.stages.size();++i)
      stages.emplace_back(tilemega::json::Object{{"stage",int(i)},{"kind",int(model.stages[i].kind)}});
    auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    tilemega::json::Value result(tilemega::json::Object{
      {"status","verified_import_only"},{"model_id",receipt.At("model_id").AsString("model_id")},
      {"phase",phase},{"batch",batch},{"seq",dims.seq},{"past",dims.past},{"capacity",serving.capacity},
      {"stages",std::move(stages)},{"gemms",int(model.gemms.size())},
      {"task_spaces",int(summary.task_spaces)},{"couplings",int(summary.couplings)},
      {"buffers",int(plan.buffers.size())},{"import_seconds",seconds},
      {"semantic_generic_stages",0},{"raw_unclassified_aten",std::move(raw_unclassified)},
      {"solver_executed",false},{"device_executed",false}});
    Write(output/"import.json",result.Dump());std::cout<<result.Dump()<<std::endl;return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
