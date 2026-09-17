// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <mlir/Parser/Parser.h>
#include <fstream>
#include <iostream>
using namespace tilemega;
int main(int argc,char** argv)try{
 if(argc!=8)throw std::invalid_argument("seqscan INPUT_JSON CHOSEN_CG TARGET SEQ PAST OUTPUT_CU HOP");
 analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
 auto selected=mlir::parseSourceFile<mlir::ModuleOp>(argv[2],&context);if(!selected)throw std::invalid_argument("invalid chosen CG");
 auto runtime=codegen::ReadRuntimePlan(*selected);frontend::ImportOptions io;
 for(auto const& g:runtime.gemms)io.gemms.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
 io.rope_tile_per_block=io.kv_tile_per_block=io.activation_tile_per_block=io.combiner_tile_per_block=true;
 auto module=frontend::TorchExportImporter{}.Import(argv[1],context,nullptr,io);
 auto integer=[&](char const* name){auto a=selected->getOperation()->getAttrOfType<mlir::IntegerAttr>(name);if(!a)throw std::invalid_argument(name);return int(a.getInt());};
 auto placement=selected->getOperation()->getAttrOfType<mlir::StringAttr>("tilemega.solved_placement");if(!placement)throw std::invalid_argument("no chosen placement");
 dialect::PlacementSolveOptions options;options.target=TargetSpec::FromJson(argv[3]);options.dims={std::stoi(argv[4]),std::stoi(argv[5]),0};options.dims.total=options.dims.seq+options.dims.past;
 options.kappa=integer("tilemega.solved_kappa");options.residency=integer("tilemega.solved_residency");options.verified_resident_limit=options.residency;
 std::string error;if(!solver::HopCurve::FromTsv(argv[7],&options.hop,&error))throw std::runtime_error(error);
 auto solved=dialect::SolveAndWritePlacement(*module,options);auto it=std::find_if(solved.candidates.begin(),solved.candidates.end(),[&](auto const& c){return c.name==placement.getValue();});
 if(it==solved.candidates.end() || !it->error.empty())throw std::runtime_error("chosen family did not materialize at new theta");
 dialect::WriteSolvedPlacement(*module,*it,options);
 std::vector<codegen::RuntimeVariantModule> variants{{*module,1u,unsigned(options.dims.seq)}};
 std::ofstream(argv[6])<<codegen::CouplingGraphToCUDA{}.LowerVariants(variants);
 std::string text;llvm::raw_string_ostream out(text);module->print(out);out.flush();std::ofstream(std::string(argv[6])+".mlir")<<text;
 std::cout<<"SEQSCAN_SOLVED seq="<<options.dims.seq<<" past="<<options.dims.past<<" placement="<<it->name<<" grid="<<solved.grid<<'\n';
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
