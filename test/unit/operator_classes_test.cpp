// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/OperatorClasses.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <iostream>
#include <regex>
#include <fstream>
using namespace tilemega;
int main(int argc,char** argv) {
 try {
  analysis::IslContext isl;mlir::MLIRContext context;
  std::string path=std::string(TILEMEGA_SOURCE_DIR)+"/docs/experiments/E2E_GEN/raw/export_bridge.json";
  auto bridge=frontend::ReadExportBridge(path);auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
  frontend::TorchExportImporter importer;auto imported=importer.ImportSemantics(path,plan,context);
  auto classes=solver::BuildOperatorClasses(imported);
  if(classes.size()<2)throw std::runtime_error("fixture needs distinct semantic classes");
  std::vector<solver::GemmConfig> selected(classes.size(),{32,16,16,2,1});selected.back()={32,32,32,2,1};
  auto options=solver::ClassGranularity(imported,classes,selected);
  analysis::CouplingCache cache;auto module=importer.InstantiateForGranularity(imported,context,options,&cache);
  auto source=codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,1u,4u}});
  if(source.find("#define TILEMEGA_GEMM_VARIANT_COUNT 2\n")==std::string::npos)throw std::runtime_error("variant count mismatch");
  auto begin=source.find("constexpr GemmRuntimeDesc kRuntimeGemms0[]");
  if(begin==std::string::npos)throw std::runtime_error("missing GEMM invocation table");
  auto end=source.find("};",begin);auto table=source.substr(begin,end-begin);
  std::regex row(R"(\{(\d+)u, (\d+)u, (\d+)u, (\d+)u, (\d+)u, (\d+)u\})");
  std::map<std::tuple<int,int,int,int>,int> index;std::size_t i=0;
  for(std::sregex_iterator it(table.begin(),table.end(),row),stop;it!=stop;++it,++i) {
    auto const& g=options.gemms.at(i);auto key=std::make_tuple(g.tile_m,g.tile_n,g.tile_k,g.stages);
    auto found=index.emplace(key,index.size()).first;auto const& match=*it;
    if(std::stoi(match[1])!=found->second || std::stoi(match[2])!=g.split_k ||
       std::stoi(match[3])!=g.tile_m || std::stoi(match[4])!=g.tile_n ||
       std::stoi(match[5])!=g.tile_k || std::stoi(match[6])!=g.stages)
      throw std::runtime_error("wrong invocation variant");
  }
  if(i!=plan.gemms.size())throw std::runtime_error("incomplete invocation verification");
  if(argc>1)std::ofstream(argv[1])<<source;
  std::cout<<"VARIANTS count=2 classes="<<classes.size()<<" invocations="<<i<<" PASS\n";
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
