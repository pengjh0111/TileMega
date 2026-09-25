// SPDX-License-Identifier: BSD-3-Clause
// R9b supersedes the R9 fork-equivalence test: the search is single-threaded,
// deterministic, imports once, and never materializes or simulates in its loop.
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/SkeletonSearch.h>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace tilemega;
int main() try {
  analysis::IslContext isl;std::string root=TILEMEGA_SOURCE_DIR;
  auto dir=std::filesystem::temp_directory_path()/"tilemega-flow-search-contract";
  std::filesystem::create_directories(dir);
  std::string expected;
  for(int repeat=0;repeat<2;++repeat) {
    mlir::MLIRContext context;solver::SolverTiming timing;solver::SkeletonSearchOptions options;
    options.common.placement.target=TargetSpec::FromJson(root+"/docs/experiments/SOLVER_R9B/fit/target.json");
    options.common.placement.target.res.num_sms=4;
    options.common.placement.dims={4,3,7};options.common.timing=&timing;
    options.seed={32,16,32,2,1};options.common.geometry_domain={options.seed};
    options.variant_probe=[](auto const&,auto const*,auto){return solver::VariantResources{32,65536,128,false};};
    options.jobs=1;options.passes=1;options.search_only=true;
    options.artifact_prefix=(dir/std::to_string(repeat)).string();
    std::ostringstream evidence;
    auto result=solver::SolveSkeletonExport(root+"/docs/experiments/SEQSCAN/raw/export/gqa2.json",context,options,nullptr,evidence);
    if(result.evaluated.size()<6 || !result.top.empty())throw std::runtime_error("search-only must evaluate candidates without top-3 materialization");
    if(timing.phases.at("import").count!=1 || timing.phases.at("cache_hit").count==0)
      throw std::runtime_error("import/cache accounting changed");
    for(auto name:{"materialize","simulate","megakernel_compile"})
      if(timing.phases.count(name) && timing.phases.at(name).count)
        throw std::runtime_error("outer loop performed final-only work");
    if(repeat==0)expected=evidence.str();
    else if(expected!=evidence.str())throw std::runtime_error("repeated coordinate search changed scores or ordering");
    std::cout<<"FLOW_SEARCH repeat="<<repeat<<" evaluations="<<result.evaluated.size()<<" imports=1 outer_materialize=0 outer_simulate=0 PASS\n";
    options.jobs=2;bool rejected=false;
    try {solver::SolveSkeletonExport("not-read.json",context,options,nullptr,evidence);}
    catch(std::invalid_argument const& e){rejected=std::string(e.what()).find("single-threaded")!=std::string::npos;}
    if(!rejected)throw std::runtime_error("parallel ISL search was not rejected before import");
  }
  std::filesystem::remove_all(dir);
  std::cout<<"FLOW_SEARCH deterministic=1 parallel_rejected=1 PASS\n";
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
