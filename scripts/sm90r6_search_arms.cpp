// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Analysis/ISLContext.h>
#include <fstream>
#include <iostream>
using namespace tilemega;
using namespace tilemega::solver;

int main(int argc,char** argv) {try {
  if(argc!=5)throw std::invalid_argument("usage: search-arms target.json source-root cells.json output.jsonl");
  analysis::IslContext isl;
  auto target_json=json::ParseFile(argv[1]);auto target=TargetSpec::FromJson(argv[1]);
  auto constants=Sm90NativeConstants::FromTarget(target_json,argv[2]);
  Sm90DenseCostModel cost(target,constants);
  if(std::ifstream(argv[4]))throw std::invalid_argument("retain previous arm search output");
  std::ofstream output(argv[4]);if(!output)throw std::runtime_error("cannot open arm search output");
  auto cells=json::ParseFile(argv[3]);
  for(auto const& cell:cells.AsArray("cells")) {
    auto seed=Sm90DenseCoordinates::FromJson(json::ParseFile(cell.At("seed").AsString("seed")));
    mlir::MLIRContext context;
    auto imported=ImportSm90Dense(cell.At("export").AsString("export"),seed,context);
    auto work=Sm90DenseWorkloadFromPlan(imported.plan,cell.At("batch").AsNumber("batch"));
    Sm90DenseSearchSpace space{work.shapes,work.batch,work.head_dim,work.queries,target,work.hkv};
    space.moe=work.moe;space.experts=work.experts;space.top_k=work.top_k;
    space.qk_norm_row=work.qk_norm_row;space.tp=work.tp;
    auto const& decisions=target_json.At("sm90_native_decisions");
    space.weight_stream=decisions.At("weight_stream").AsBool("weight_stream");
    space.layer_loop=decisions.At("layer_loop").AsBool("layer_loop");
    Sm90NativePipelineOptions options;
    options.budget_ms=cell.At("budget_ms").AsNumber("budget_ms");
    if(!(options.budget_ms>0 && options.budget_ms<=600000))throw std::invalid_argument("invalid arm search budget");
    auto arm=cell.At("arm").AsString("arm");
    if(arm=="AB2")options.operator_sum=true;
    else if(arm=="AB4a")options.retile_only=true;
    else if(arm=="AB4b")options.realize_aggregate=true;
    else if(arm=="AB1a" || arm=="AB1b")
      options.pinned_theta=Sm90DenseCoordinates::FromJson(json::ParseFile(cell.At("theta").AsString("theta")));
    else throw std::invalid_argument("unknown independently solved ablation arm");
    Sm90NativeCtgCache cache;
    auto evaluate=[&](Sm90DenseCoordinates const& c,bool isolated) {
      return Sm90NativeCtgScore(imported,work,c,cost,context,cache,isolated);
    };
    auto result=SolveSm90NativePipeline(space,work,cost,options,evaluate,{seed});
    auto rows=[](auto const& candidates) {
      json::Array values;
      for(auto const& c:candidates)values.emplace_back(json::Object{{"coordinates",c.coordinates.ToJson()},
          {"pool_ns",c.pool_score_ns},{"replay_ns",c.replay_ns},{"evaluation_ms",c.evaluation_ms},
          {"discovered_ms",c.discovered_ms},{"origin",c.origin},{"structural_class",c.structural_class}});
      return values;
    };
    json::Array trajectories;
    for(auto const& path:result.descent) {json::Array values;for(auto x:path)values.emplace_back(x);trajectories.emplace_back(values);}
    json::Value row=json::Object{{"input",cell},{"analysis","reuse"},
        {"scope","offline Algorithm 1 under the requested arm constraints; no hardware selection"},
        {"objective",options.operator_sum?"sum of isolated FlowSpace evaluations":"PoolScore then template replay"},
        {"evaluated",rows(result.evaluated)},{"shortlist",rows(result.shortlist)},
        {"finalists",rows(result.finalists)},{"descent_scores",trajectories},
        {"seconds",result.seconds},{"evaluations",double(result.evaluations)},
        {"memo_hits",double(result.memo_hits)}};
    output<<row.Dump(0);output.flush();
    std::cout<<cell.At("id").AsString("id")<<' '<<arm<<" evaluated="<<result.evaluations<<'\n'<<std::flush;
  }
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
