// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/SkeletonSearch.h>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace tilemega;
int main(int argc,char** argv) {
 try {
  int parallel=argc>1?std::stoi(argv[1]):3;
  if(parallel<2 || parallel>64)throw std::invalid_argument("parallel jobs must be in 2..64");
  bool wide=argc>2 && std::string(argv[2])=="--wide-domain";
  analysis::IslContext isl;std::string root=TILEMEGA_SOURCE_DIR;
  auto dir=std::filesystem::temp_directory_path()/("tilemega-search-isolation-test-"+std::to_string(parallel));
  std::filesystem::create_directories(dir);
  std::string expected;std::vector<std::string> top;
  for(int jobs:{1,parallel}) {
    mlir::MLIRContext context;solver::SolverTiming timing;solver::SkeletonSearchOptions options;
    options.common.placement.target=TargetSpec::FromJson(root+"/docs/experiments/COSTMODEL/event_fit/target.json");
    // A CPU fixture with one resident level; these are not GPU occupancy measurements.
    options.common.placement.target.res.num_sms=4;
    
    options.common.placement.dims={4,3,7};options.common.timing=&timing;
    options.seed={32,16,32,2,1};options.common.geometry_domain={options.seed};
    if(wide)options.common.geometry_domain={options.seed,{32,16,16,2,1},
      {32,16,64,2,1},{64,16,16,2,1},{16,16,16,2,1},{16,32,16,2,1}};
    options.variant_probe=[](auto const&,auto const*,auto){return solver::VariantResources{32,65536,128,false};};
    options.common.query_residency=[](auto,int){return 1;};
    options.jobs=jobs;options.passes=1;options.artifact_prefix=(dir/std::to_string(jobs)).string();
    std::ostringstream evidence;
    auto result=solver::SolveSkeletonExport(root+"/docs/experiments/SEQSCAN/raw/export/gqa2.json",context,options,nullptr,evidence);
    if(result.evaluated.size()<6 || result.top.size()!=5)throw std::runtime_error("fixture did not exercise candidate batches and top-K");
    if(timing.phases.at("import").count!=1 || timing.phases.at("cache_hit").count==0)
      throw std::runtime_error("import/cache accounting changed");
    if(jobs==1){expected=evidence.str();for(auto const& candidate:result.top)top.push_back(candidate.key);}
    else {
      if(expected!=evidence.str())throw std::runtime_error("parallel coordinate evaluations differ from serial");
      for(std::size_t i=0;i<top.size();++i)if(top[i]!=result.top.at(i).key)throw std::runtime_error("parallel top-K differs");
      for(int rank=1;rank<=5;++rank) {
        auto read=[&](int j){std::ifstream in(dir/(std::to_string(j)+".final"+std::to_string(rank)+".tasks.tsv"));return std::string(std::istreambuf_iterator<char>(in),{});};
        if(read(1)!=read(parallel))throw std::runtime_error("parallel final schedule differs");
      }
    }
    std::cout<<"SEARCH_ISOLATION jobs="<<jobs<<" evaluations="<<result.evaluated.size()<<" import_count="<<timing.phases.at("import").count<<" PASS\n";
  }
  std::filesystem::remove_all(dir);
  std::cout<<"SEARCH_ISOLATION candidate_records_equal=1 top5_schedules_equal=1 PASS\n";
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
