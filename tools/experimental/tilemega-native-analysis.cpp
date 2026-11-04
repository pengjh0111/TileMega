// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Analysis/ISLContext.h>
#include <fstream>
#include <iostream>
#include <numeric>
using namespace tilemega;
using namespace tilemega::solver;
int main(int argc,char** argv) {try {
  if(argc!=5) {
    std::cerr<<"usage: tilemega-native-analysis target.json source-root cells.json output.jsonl\n";
    return 2;
  }
  analysis::IslContext isl;
  auto target=TargetSpec::FromJson(argv[1]);
  auto constants=Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]),argv[2]);
  Sm90DenseCostModel cost(target,constants);
  std::ifstream existing(argv[4]);
  if(existing)throw std::invalid_argument("retain previous analysis output; choose a new path");
  std::ofstream output(argv[4]);if(!output)throw std::runtime_error("cannot open analysis output");
  auto cells=json::ParseFile(argv[3]);
  for(auto const& cell:cells.AsArray("cells")) {
    auto receipt=json::ParseFile(cell.At("receipt").AsString("receipt"));
    auto const& candidates=receipt.At("evaluated").AsArray("evaluated");
    if(candidates.empty())throw std::invalid_argument("candidate set is empty");
    auto seed=Sm90DenseCoordinates::FromJson(candidates.front().At("coordinates"));
    mlir::MLIRContext context;
    auto imported=ImportSm90Dense(cell.At("export").AsString("export"),seed,context);
    auto work=Sm90DenseWorkloadFromPlan(imported.plan,cell.At("batch").AsNumber("batch"));
    Sm90NativeCtgCache reused;
    int index=0;
    for(auto const& candidate:candidates) {
      auto coordinate=Sm90DenseCoordinates::FromJson(candidate.At("coordinates"));
      std::string identity;double reference=0;
      for(auto const* mode:{"reuse","rederive","materialize"}) {
        Sm90NativeCtgCache fresh;fresh.materialize=std::string(mode)=="materialize";
        auto& cache=std::string(mode)=="reuse"?reused:fresh;
        auto old_hits=cache.coupling.hits;
        auto old_release_hits=cache.release_hits,old_space_hits=cache.space_hits;
        auto result=Sm90NativeCtgScore(imported,work,coordinate,cost,context,cache);
        auto encoded=Sm90NativeFlowIdentity(result.problem);
        if(std::string(mode)=="reuse") {identity=encoded;reference=result.step_ns;}
        else if(identity!=encoded || reference!=result.step_ns)
          throw std::runtime_error("AB5 changed the exact FlowProblem or PoolScore at candidate "+std::to_string(index));
        auto tasks=std::accumulate(result.problem.spaces.begin(),result.problem.spaces.end(),
            std::size_t(0),[](auto n,auto const& s){return n+s.count;});
        if(fresh.materialize && tasks!=result.enumerated_tasks)
          throw std::runtime_error("materialization did not enumerate every task");
        json::Value row=json::Object{{"input",cell},{"candidate",index},{"analysis",mode},
            {"coordinates",coordinate.ToJson()},{"pool_ns",result.step_ns},
            {"analysis_ms",result.analysis_ms},{"release_ms",result.release_ms},
            {"pricing_ms",result.pricing_ms},{"evaluation_ms",result.evaluation_ms},
            {"derivations",double(result.derivations)},
            {"coupling_hits",double(cache.coupling.hits-old_hits)},
            {"space_hits",double(cache.space_hits-old_space_hits)},
            {"release_hits",double(cache.release_hits-old_release_hits)},
            {"enumerated_tasks",double(result.enumerated_tasks)},
            {"enumerated_edges",double(result.enumerated_edges)},
            {"tasks",double(tasks)},{"spaces",double(result.problem.spaces.size())},
            {"couplings",double(result.problem.edges.size())},
            {"flow_identity_bytes",double(encoded.size())},{"bitwise_flow_equal",true}};
        output<<row.Dump(0);output.flush();
      }
      std::cout<<cell.At("id").AsString("id")<<" candidate="<<index++
          <<"/"<<candidates.size()<<" three-way bitwise equality passed\n"<<std::flush;
    }
  }
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
