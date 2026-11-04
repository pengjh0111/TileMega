// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
using namespace tilemega;
using namespace tilemega::solver;
int main(int argc,char** argv) {try {
  if(argc!=5) {
    std::cerr<<"usage: tilemega-native-search target.json source-root cells.json output.jsonl\n";
    return 2;
  }
  analysis::IslContext isl;
  auto target_json=json::ParseFile(argv[1]);auto target=TargetSpec::FromJson(argv[1]);
  auto constants=Sm90NativeConstants::FromTarget(target_json,argv[2]);
  Sm90DenseCostModel cost(target,constants);
  std::ifstream existing(argv[4]);
  if(existing)throw std::invalid_argument("retain prior search results; choose a new output path");
  std::ofstream output(argv[4]);if(!output)throw std::runtime_error("cannot open search output");
  auto cells=json::ParseFile(argv[3]);
  for(auto const& cell:cells.AsArray("cells")) {
    auto seed=Sm90DenseCoordinates::FromJson(json::ParseFile(cell.At("seed").AsString("seed")));
    auto inspected=Sm90DenseCoordinates::FromJson(json::ParseFile(cell.At("inspection").AsString("inspection")));
    mlir::MLIRContext context;
    auto imported=ImportSm90Dense(cell.At("export").AsString("export"),seed,context);
    int batch=cell.At("batch").AsNumber("batch");
    auto work=Sm90DenseWorkloadFromPlan(imported.plan,batch);
    Sm90DenseSearchSpace space{work.shapes,work.batch,work.head_dim,work.queries,target,work.hkv};
    space.moe=work.moe;space.experts=work.experts;space.top_k=work.top_k;
    space.qk_norm_row=work.qk_norm_row;space.tp=work.tp;
    auto const& decisions=target_json.At("sm90_native_decisions");
    space.weight_stream=decisions.At("weight_stream").AsBool("weight_stream");
    space.layer_loop=decisions.At("layer_loop").AsBool("layer_loop");
    if(!space.Legal(seed) || !space.Legal(inspected))throw std::invalid_argument("illegal seed or inspected coordinate");
    Sm90NativeCtgCache cache;
    std::string mode="reuse";
    if(auto const* setting=cell.Find("analysis"))mode=setting->AsString("analysis");
    if(mode!="reuse" && mode!="rederive" && mode!="materialize")
      throw std::invalid_argument("invalid analysis mode");
    std::map<std::string,std::string> identities;
    json::Array analysis_calls;
    auto evaluate=[&](Sm90DenseCoordinates const& c,bool isolated) {
      auto flow=Sm90NativeAnalyze(imported,work,c,cost,context,cache,mode,isolated);
      identities[c.ToJson().Dump(0)]=Sm90NativeFlowIdentity(flow.problem);
      analysis_calls.emplace_back(json::Object{{"coordinates",c.ToJson()},
          {"analysis_ms",flow.analysis_ms},{"release_ms",flow.release_ms},
          {"pricing_ms",flow.pricing_ms},{"evaluation_ms",flow.evaluation_ms},
          {"derivations",double(flow.derivations)},
          {"enumerated_tasks",double(flow.enumerated_tasks)},
          {"enumerated_edges",double(flow.enumerated_edges)}});
      return flow;
    };
    Sm90NativePipelineOptions options;options.budget_ms=cell.At("budget_ms").AsNumber("budget_ms");
    if(!std::isfinite(options.budget_ms) || options.budget_ms<=0)
      throw std::invalid_argument("search budget must be finite and positive");
    auto result=SolveSm90NativePipeline(space,work,cost,options,evaluate,{seed});
    auto verification_start=std::chrono::steady_clock::now();
    if(mode!="reuse") {
      Sm90NativeCtgCache reference;
      // Verification is outside the search budget and cannot alter E or Q_short.
      for(auto const& c:result.evaluated) {
        auto flow=Sm90NativeCtgScore(imported,work,c.coordinates,cost,context,reference);
        if(Sm90NativeFlowIdentity(flow.problem)!=identities.at(c.coordinates.ToJson().Dump(0)))
          throw std::runtime_error("analysis mode changed a searched candidate's FlowProblem");
      }
    }
    double verification_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-verification_start).count();
    auto identity=inspected.ToJson().Dump(0);
    auto contains=[&](auto const& candidates) {
      return std::any_of(candidates.begin(),candidates.end(),[&](auto const& c) {
        return c.coordinates.ToJson().Dump(0)==identity;
      });
    };
    // Inspection follows the frozen search. It neither adds a seed nor changes Q_short.
    Sm90NativeCtgCache inspection_cache;
    auto probe=Sm90NativeCtgScore(imported,work,inspected,cost,context,inspection_cache);
    int better=0;for(auto const& candidate:result.evaluated)better+=candidate.pool_score_ns<probe.step_ns;
    auto rows=[](auto const& candidates) {
      json::Array values;
      for(auto const& c:candidates)values.emplace_back(json::Object{{"coordinates",c.coordinates.ToJson()},
          {"pool_ns",c.pool_score_ns},{"replay_ns",c.replay_ns},{"evaluation_ms",c.evaluation_ms},
          {"discovered_ms",c.discovered_ms},
          {"origin",c.origin},{"structural_class",c.structural_class}});
      return values;
    };
    json::Array trajectories;
    for(auto const& path:result.descent) {json::Array points;for(auto x:path)points.emplace_back(x);trajectories.emplace_back(points);}
    json::Value row=json::Object{{"input",cell},{"scope","offline Algorithm 1 through finalists; no compilation or occupancy"},
        {"analysis",mode},{"analysis_calls",analysis_calls},{"verification_ms",verification_ms},
        {"bitwise_flow_verified_against_reuse",mode!="reuse"},
        {"evaluated",rows(result.evaluated)},{"shortlist",rows(result.shortlist)},
        {"finalists",rows(result.finalists)},{"descent_scores",trajectories},
        {"seconds",result.seconds},{"evaluations",double(result.evaluations)},
        {"memo_hits",double(result.memo_hits)},{"space_hits",double(cache.space_hits)},
        {"release_hits",double(cache.release_hits)},
        {"inspection",json::Object{{"coordinates",inspected.ToJson()},{"pool_ns",probe.step_ns},
          {"present_in_E",contains(result.evaluated)},{"comparison_rank",better+1},
          {"rank_is_counterfactual",!contains(result.evaluated)},{"in_shortlist",contains(result.shortlist)}}}};
    output<<row.Dump(0);output.flush();
    std::cout<<cell.At("id").AsString("id")<<" evaluated="<<result.evaluations
        <<" inspection_rank="<<better+1<<" in_E="<<contains(result.evaluated)
        <<" shortlisted="<<contains(result.shortlist)<<'\n'<<std::flush;
  }
  return 0;
}catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}}
