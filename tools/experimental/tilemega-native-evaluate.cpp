// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90DenseResources.h>
#include <tilemega/Analysis/ISLContext.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>

using namespace tilemega;
using namespace tilemega::solver;
struct EvaluationState {
  mlir::MLIRContext context;
  frontend::ImportedSemantics imported;
  Sm90NativeCtgCache cache;
  EvaluationState(std::string const& graph,Sm90DenseCoordinates const& c):
      imported(ImportSm90Dense(graph,c,context)) {}
};
int main(int argc,char** argv) {try {
  if(argc!=5) {
    std::cerr<<"usage: tilemega-native-evaluate target.json source-root cells.json output.jsonl\n";
    return 2;
  }
  analysis::IslContext isl;
  auto target=TargetSpec::FromJson(argv[1]);
  auto constants=Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]),argv[2]);
  Sm90DenseCostModel cost(target,constants);
  std::map<std::string,std::unique_ptr<EvaluationState>> states;
  std::ifstream existing(argv[4]);
  if(existing)throw std::invalid_argument("retain previous evaluation output; choose a new path");
  std::ofstream output(argv[4]);if(!output)throw std::runtime_error("cannot open evaluation output");
  auto cells=json::ParseFile(argv[3]);
  for(auto const& cell:cells.AsArray("cells")) {
    auto graph=cell.At("export").AsString("export");
    auto c=Sm90DenseCoordinates::FromJson(json::ParseFile(cell.At("coordinates").AsString("coordinates")));
    auto started=std::chrono::steady_clock::now();
    bool cold=!states.count(graph);
    if(cold)states.emplace(graph,std::make_unique<EvaluationState>(graph,c));
    auto& state=*states.at(graph);
    int batch=cell.At("batch").AsNumber("batch");
    auto work=Sm90DenseWorkloadFromPlan(state.imported.plan,batch);
    c.ValidateSchedule(work.shapes,batch,work.head_dim,work.queries,1088,
        Sm90DenseSharedBudget(target,c.residency));
    auto evaluate_start=std::chrono::steady_clock::now();
    auto flow=Sm90NativeCtgScore(state.imported,work,c,cost,state.context,state.cache);
    auto flow_end=std::chrono::steady_clock::now();
    std::vector<int> offsets(flow.nodes.size());
    for(unsigned i=1;i<flow.nodes.size();++i)
      if(c.expert_fuse && flow.nodes[i].price.name=="gemm7" && flow.nodes[i-1].price.name=="gemm6")
        offsets[i]=flow.problem.spaces[i-1].count;
    auto replay=RefineAndReplaySm90Native(flow.problem,Sm90NativeCandidateTracks(flow.problem,offsets));
    auto done=std::chrono::steady_clock::now();
    auto milliseconds=[](auto a,auto b) {return std::chrono::duration<double,std::milli>(b-a).count();};
    auto additive=cost.Predict(work,c);
    json::Array spaces;
    for(unsigned i=0;i<flow.nodes.size();++i) {
      auto const& node=flow.nodes[i];auto const& measured=flow.evaluated.spaces.at(i);
      spaces.emplace_back(json::Object{{"name",flow.problem.spaces[i].name},{"kind",node.price.name},
          {"tasks",node.price.tasks},{"sync",Sm90SyncKindName(node.sync)},
          {"first_start_ns",measured.first_start},{"last_end_ns",measured.last_end},
          {"fixed_ns",measured.fixed_ns},{"mainloop_ns",measured.mainloop_ns},
          {"wait_ns",measured.wait_ns},{"approximation",node.price.approximation}});
    }
    json::Array stale;
    for(auto const& header:constants.StaleHeaders(argv[2]))stale.emplace_back(header);
    json::Value row=json::Object{{"input",cell},{"coordinates",c.ToJson()},{"cold_import",cold},
        {"import_ms",milliseconds(started,evaluate_start)},
        {"pool_evaluation_ms",milliseconds(evaluate_start,flow_end)},
        {"replay_ms",milliseconds(flow_end,done)},{"pool_ns",flow.step_ns},
        {"replay_ns",replay.replay.makespan_ns},{"additive_ns",additive.step_ns},
        {"additive_stages",additive.ToJson()},{"spaces",spaces},{"approximation",flow.approximation},
        {"constants_from_older_body",stale},{"space_hits",double(state.cache.space_hits)},
        {"space_misses",double(state.cache.space_misses)},
        {"release_hits",double(state.cache.release_hits)},{"release_misses",double(state.cache.release_misses)}};
    output<<row.Dump(0);output.flush();
    std::cout<<cell.At("id").AsString("id")<<" pool_ms="<<flow.step_ns/1e6
        <<" replay_ms="<<replay.replay.makespan_ns/1e6<<" additive_ms="<<additive.step_ns/1e6<<'\n'<<std::flush;
  }
  return 0;
}catch(std::exception const& error) {std::cerr<<error.what()<<'\n';return 1;}}
