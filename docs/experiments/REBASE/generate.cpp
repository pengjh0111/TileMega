// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <mlir/Parser/Parser.h>
#include <llvm/Support/raw_ostream.h>
#include <fstream>
#include <filesystem>
#include <iostream>
using namespace tilemega;
int main(int argc,char** argv)try{
 if(argc!=4)throw std::invalid_argument("generate SOLVED_CG TARGET OUT_DIR");
 analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
 auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);if(!module)throw std::invalid_argument("invalid CG");
 auto attr=[&](char const* name){auto a=module->getOperation()->getAttrOfType<mlir::IntegerAttr>(name);if(!a)throw std::invalid_argument(name);return int(a.getInt());};
 dialect::PlacementSolveOptions options;options.target=TargetSpec::FromJson(argv[2]);
 options.dims={attr("tilemega.solved_seq"),attr("tilemega.solved_past"),0};options.dims.total=options.dims.seq+options.dims.past;
 options.kappa=attr("tilemega.solved_kappa");options.residency=attr("tilemega.solved_residency");options.verified_resident_limit=options.residency;
 std::string error;if(!solver::HopCurve::FromTsv(std::getenv("R6_REBASE_HOP") ? std::getenv("R6_REBASE_HOP") : "docs/experiments/SIMULATOR/hop_ns.tsv",&options.hop,&error))throw std::runtime_error(error);
 auto solved=dialect::SolveAndWritePlacement(*module,options);std::filesystem::path out=argv[3];std::filesystem::create_directories(out);
 std::ofstream rows(out/"placements.tsv");rows<<"placement\tkappa\tresidency\tgrid\tzero_sync_floor_ns\tsemantic_cp_ns\tqueue_lb_ns\tpredicted_ns\tsource\n";
 for(auto const& candidate:solved.candidates){
  if(!candidate.error.empty())throw std::runtime_error(candidate.name+": "+candidate.error);
  auto clone=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(module->clone()));dialect::WriteSolvedPlacement(*clone,candidate,options);
  std::vector<codegen::RuntimeVariantModule> variants{{*clone,1u,unsigned(options.dims.seq)}};
  auto stem=out/candidate.name;std::ofstream(stem.string()+".cu")<<codegen::CouplingGraphToCUDA{}.LowerVariants(variants);
  std::error_code ec;llvm::raw_fd_ostream cg(stem.string()+".mlir",ec);if(ec)throw std::runtime_error("cannot write CG");(*clone).print(cg);
  rows<<candidate.name<<'\t'<<options.kappa<<'\t'<<options.residency<<'\t'<<solved.grid<<'\t'<<candidate.bounds.lower_bound_ns<<'\t'<<candidate.bounds.critical_path_ns<<'\t'<<candidate.bounds.queue_lb_ns<<'\t'<<candidate.predicted_ns<<'\t'<<std::filesystem::absolute(stem).string()<<".cu\n";
 }
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
