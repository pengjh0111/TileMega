// SPDX-License-Identifier: BSD-3-Clause
// R9b supersedes the R9 fork-equivalence test: the search is single-threaded,
// deterministic, imports once, and never materializes or simulates in its loop.
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace tilemega::tests::skeleton_search_isolation_test {
using namespace tilemega;
int TestServingSearchRejection(int, char**) try {
  analysis::IslContext isl;mlir::MLIRContext context;
  std::string root=TILEMEGA_SOURCE_DIR;
  auto export_path=root+"/test/fixtures/serving_rejection_export.json";
  auto bridge=frontend::ReadExportBridge(export_path);
  frontend::ServingOptions serving;serving.phase=frontend::ServingOptions::Phase::kDecode;
  serving.seq=1;serving.kv_block=256;serving.argmax_tile_n=32;
  auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,serving);
  frontend::TorchExportImporter importer;
  auto imported=importer.ImportSemantics(export_path,plan,context);
  auto classes=solver::BuildOperatorClasses(imported);
  std::vector<solver::GemmConfig> valid(classes.size(),{16,32,128,7,1});
  std::size_t head=classes.size();
  for(std::size_t c=0;c<classes.size();++c)
    for(auto id:classes[c].gemms)
      if(plan.gemms[id].epilogue==frontend::PlanGemm::Epilogue::kArgmaxPartial)head=c;
  if(head==classes.size())throw std::runtime_error("regression fixture lacks lm_head");
  auto invalid=valid;invalid[head]={16,8,64,2,1,1};
  auto alternate=valid;alternate[head]={16,64,128,4,1};
  auto dir=std::filesystem::temp_directory_path()/"tilemega-serving-rejected-structure";
  std::filesystem::create_directories(dir);
  for(bool incremental:{false,true}) {
    solver::SkeletonSearchOptions options;
    options.common.placement.target=TargetSpec::FromJson(root+"/test/fixtures/serving_rejection_target.json");
    options.common.placement.dims={1,1,575};options.search_only=true;
    options.incremental_prepare=incremental;options.artifact_prefix=(dir/std::to_string(incremental)).string();
    options.variant_probe=[](auto const&,auto const*,auto){return solver::VariantResources{32,86016,128,false};};
    options.evaluation_cases={{valid,1,1},{alternate,1,1},{valid,1,1}};
    std::ostringstream control_log;
    auto control=solver::SolveSkeletonImported(imported,context,options,nullptr,control_log);
    options.evaluation_cases={{valid,1,1},{invalid,1,1},{alternate,1,1},{valid,1,1}};
    std::ostringstream evidence;
    auto recovered=solver::SolveSkeletonImported(imported,context,options,nullptr,evidence);
    if(recovered.evaluated.size()!=4 || recovered.evaluated[1].error.find("argmax partial tile")==std::string::npos)
      throw std::runtime_error("illegal head tile was not rejected in place");
    for(auto [actual,expected]:{std::pair{0,0},std::pair{2,1},std::pair{3,2}}) {
      auto const& result=recovered.evaluated[actual];
      if(!result.error.empty() || !std::isfinite(result.score) || result.score!=control.evaluated[expected].score)
        throw std::runtime_error("rejected structure changed later legal candidate scores");
    }
    std::cout<<"SERVING_REJECTION incremental="<<incremental<<" invalid_rejected=1 later_scores_unchanged=1 PASS\n";
  }
  std::filesystem::remove_all(dir);return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}

int TestSkeletonSearchIsolation(int argc, char** argv) try {
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
    options.jobs=repeat?3:1;options.passes=1;options.search_only=true;
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
    options.jobs=0;bool rejected=false;
    try {solver::SolveSkeletonExport("not-read.json",context,options,nullptr,evidence);}
    catch(std::invalid_argument const& e){rejected=std::string(e.what()).find("positive")!=std::string::npos;}
    if(!rejected)throw std::runtime_error("invalid compilation jobs were not rejected before import");
  }
  std::filesystem::remove_all(dir);
  std::cout<<"FLOW_SEARCH deterministic=1 compile_jobs_3_equivalent=1 invalid_jobs_rejected=1 PASS\n";
  return 0;

  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}

}  // namespace tilemega::tests::skeleton_search_isolation_test
