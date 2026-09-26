// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <mlir/Parser/Parser.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace tilemega;
int main(int argc,char** argv) try {
  if(argc!=8)throw std::invalid_argument("usage: tilemega-flow-grid export target fixture resources legacy_root model out");
  analysis::IslContext isl;mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();context.getOrLoadDialect<dialect::ExecDialect>();
  auto target=TargetSpec::FromJson(argv[2]);solver::SolverTiming import_timing;
  frontend::TorchExportImporter importer;
  auto imported=[&]{solver::SolverPhase phase(&import_timing,"import");
    auto bridge=frontend::ReadExportBridge(argv[1]);auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
    return importer.ImportSemantics(argv[1],plan,context);}();
  std::filesystem::path root=argv[7];std::filesystem::create_directories(root);
  std::ofstream imports(root/(std::string(argv[6])+".import.tsv"));import_timing.Write(imports,"theta-grid",argv[6],0);imports.flush();
  for(int seq:{1,2,4,8,16,32,64}) {
    int seed_seq=1;for(int available:{1,4,16,64})if(available>=seq){seed_seq=available;break;}
    auto directory=root/(std::string(argv[6])+"_s"+std::to_string(seq));std::filesystem::create_directories(directory);
    auto prefix=(directory/"selected.cu").string();
    if(std::filesystem::exists(prefix+".search.tsv"))throw std::runtime_error("refusing to overwrite theta evidence");
    auto seed=mlir::parseSourceFile<mlir::ModuleOp>((std::filesystem::path(argv[5])/(std::string(argv[6])+"_s"+std::to_string(seed_seq))/"selected.mlir").string(),&context);
    if(!seed)throw std::runtime_error("missing theta seed");
    auto runtime=codegen::ReadRuntimePlan(*seed);auto const& g=runtime.gemms.at(0);
    solver::SolverTiming timing;solver::SkeletonSearchOptions options;
    options.common.timing=&timing;options.common.placement.target=target;options.common.placement.dims={seq,3,seq+3};
    options.seed={g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k};options.search_only=true;
    options.fixture=argv[3];options.artifact_prefix=prefix;
    options.kappa=(*seed)->getAttrOfType<mlir::IntegerAttr>("tmexec.solved_kappa").getInt();
    options.seed_residency=(*seed)->getAttrOfType<mlir::IntegerAttr>("tmexec.solved_residency").getInt();
    std::string error;if(!solver::HopCurve::FromTsv(std::string(TILEMEGA_SOURCE_DIR)+"/docs/experiments/SIMULATOR/hop_ns.tsv",&options.common.placement.hop,&error))throw std::runtime_error(error);
    options.variant_probe=[&](auto const&,solver::GemmConfig const* tile,auto){
      auto name=tile?std::to_string(tile->tile_m)+"_"+std::to_string(tile->tile_n)+"_"+std::to_string(tile->tile_k)+"_"+std::to_string(tile->stages):"nongemm";
      auto file=llvm::MemoryBuffer::getFile((std::filesystem::path(argv[4])/(name+".json")).string());
      if(!file)throw std::runtime_error("missing cached resource "+name);
      auto json=llvm::json::parse(file.get()->getBuffer());auto* value=json?json->getAsObject():nullptr;
      if(!value)throw std::runtime_error("invalid resource JSON");
      return solver::VariantResources{int(*value->getInteger("registers")),int(*value->getInteger("shared_bytes")),int(*value->getInteger("threads")),false};
    };
    std::ofstream evidence(prefix+".search.tsv");auto start=std::chrono::steady_clock::now();
    auto result=solver::SolveSkeletonImported(imported,context,options,nullptr,evidence);
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream times(prefix+".timing.tsv");timing.Write(times,"theta-grid",argv[6],seq);
    auto const& best=result.evaluated.front();
    std::ofstream done(directory/"completed.tsv");done<<std::setprecision(17)<<"seq\tseed_seq\tevaluations\trounds\twall_seconds\tflow_ns\tkey\n"<<seq<<'\t'<<seed_seq<<'\t'<<result.evaluated.size()<<'\t'<<result.rounds<<'\t'<<elapsed<<'\t'<<best.score<<'\t'<<best.key<<'\n';
    std::cout<<"THETA_COMPLETE seq="<<seq<<" seconds="<<elapsed<<std::endl;
  }
  return 0;
}catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 1;}
