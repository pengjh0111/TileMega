// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
using namespace tilemega;
using namespace tilemega::solver;
int main(int argc,char** argv) {try {
  if(argc<5 || (argc-5)%2!=0) {
    std::cerr<<"usage: tilemega-native-search target.json source-root cells.json output.jsonl "
                 "[--search-profile sq2 --native-cost-table FILE --extra-seeds FILE "
                 "--objective completion|operator-sum]\n";
    return 2;
  }
  std::map<std::string,std::string> flags;
  for (int i = 5; i < argc; i += 2) {
    std::string flag = argv[i];
    if (flag!="--search-profile" && flag!="--native-cost-table" &&
        flag!="--extra-seeds" && flag!="--objective" && flag!="--diagnose-owners")
      throw std::invalid_argument("unknown native search option " + flag);
    if (!flags.emplace(flag, argv[i+1]).second) throw std::invalid_argument("duplicate " + flag);
  }
  bool sq2=flags.count("--search-profile");
  if (!flags.empty() && (!sq2 || flags.at("--search-profile")!="sq2" ||
      !flags.count("--native-cost-table") || !flags.count("--extra-seeds")))
    throw std::invalid_argument("SQ2 requires the cost table and extra seed files");
  std::string objective=flags.count("--objective")?flags.at("--objective"):"completion";
  if (objective!="completion" && objective!="operator-sum")
    throw std::invalid_argument("unknown native search objective");
  bool diagnose=flags.count("--diagnose-owners");
  if (diagnose && (flags.at("--diagnose-owners")!="1" || objective!="completion"))
    throw std::invalid_argument("owner diagnostics require 1 and the completion objective");
  analysis::IslContext isl;
  auto target_json=json::ParseFile(argv[1]);auto target=TargetSpec::FromJson(argv[1]);
  auto constants=Sm90NativeConstants::FromTarget(target_json,argv[2]);
  if (sq2 && !constants.StaleHeaders(argv[2]).empty())
    throw std::invalid_argument("stage cost table requires current target task-body hashes");
  Sm90DenseCostModel cost(target,constants,sq2?flags.at("--native-cost-table"):"");
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
    if (sq2) {
      if (cells.AsArray("cells").size()!=1 || options.budget_ms!=600000)
        throw std::invalid_argument("SQ2 takes one case with the fixed 600-second budget");
      options.search_profile="sq2";
      options.operator_sum=objective=="operator-sum";
      auto sample=json::ParseFile(flags.at("--extra-seeds"));
      for (auto const& row : sample.At("random_search_seeds").AsArray("random_search_seeds")) {
        auto c=Sm90DenseCoordinates::FromJson(row);
        if (!space.Legal(c)) throw std::invalid_argument("SQ2 random seed is illegal");
        options.extra_seeds.push_back(std::move(c));
      }
    }
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
    auto probe=Sm90NativeCtgScore(imported,work,inspected,cost,context,inspection_cache,options.operator_sum);
    int better=0;for(auto const& candidate:result.evaluated)better+=candidate.pool_score_ns<probe.step_ns;
    auto rows=[sq2](auto const& candidates) {
      json::Array values;
      for(auto const& c:candidates) {
        json::Value row=json::Object{{"coordinates",c.coordinates.ToJson()},
            {"pool_ns",c.pool_score_ns},{"replay_ns",c.replay_ns},{"evaluation_ms",c.evaluation_ms},
            {"discovered_ms",c.discovered_ms},
            {"origin",c.origin},{"structural_class",c.structural_class}};
        if (sq2) row.Set("implementation_signature",c.implementation_signature);
        values.push_back(std::move(row));
      }
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
    if (sq2) {
      row.Set("search_profile","sq2");
      row.Set("objective",objective);
      row.Set("owner_mode","fixed-skeleton diagnostic; no arbitrary GPU ownership emitted");
      row.Set("pooled_seconds",result.pooled_seconds);
      row.Set("shortlist_replay_seconds",result.replay_seconds);
    }
    if (diagnose) {
      json::Array diagnostics;
      for (auto const& candidate : result.shortlist) {
        auto flow=Sm90NativeCtgScore(imported,work,candidate.coordinates,cost,context,cache);
        auto chosen=DiagnoseSm90NativeSq2Owners(flow.problem);
        auto native=Sm90NativeCandidateTracks(flow.problem);
        json::Value first;
        std::size_t count=0, moved=0, moves_in_body=0, split_stripes=0, excess_stripes=0;
        for (std::size_t s=0; s<native.size(); ++s) {
          auto const& category=flow.problem.spaces[s].category;
          bool typed_body=category=="gemm" || category=="attention";
          std::vector<std::set<int>> requested_by_stripe(flow.problem.workers);
          std::vector<int> requested_load(flow.problem.workers);
          for (std::size_t t=0; t<native[s].size(); ++t) {
            ++count;
            int requested=chosen.placement.worker_of_task[s][t];
            if (typed_body) {
              requested_by_stripe[native[s][t]].insert(requested);
              ++requested_load[requested];
            }
            if (requested==native[s][t]) continue;
            ++moved;
            if (typed_body) {
              ++moves_in_body;
              if (moves_in_body==1)
                first=json::Object{{"space",flow.problem.spaces[s].name},{"task",double(t)},
                    {"compiled_owner",native[s][t]},{"eq5_owner",requested}};
            }
          }
          auto max_tasks=(native[s].size()+flow.problem.workers-1)/flow.problem.workers;
          for (int w=0; w<flow.problem.workers; ++w) {
            if (requested_by_stripe[w].size()>1) ++split_stripes;
            if (std::size_t(requested_load[w])>max_tasks) ++excess_stripes;
          }
        }
        diagnostics.emplace_back(json::Object{{"coordinates",candidate.coordinates.ToJson()},
            {"tasks",double(count)},{"changed_owners",double(moved)},
            {"changed_body_owners",double(moves_in_body)},{"first_body_mismatch",first},
            {"split_native_stripes",double(split_stripes)},
            {"track_load_exceeds_body_stripe",double(excess_stripes)},
            {"native_replay_ns",candidate.replay_ns},{"hypothetical_replay_ns",chosen.replay.makespan_ns}});
      }
      row.Set("eq5_diagnostic",diagnostics);
    }
    output<<row.Dump(0);output.flush();
    std::cout<<cell.At("id").AsString("id")<<" evaluated="<<result.evaluations
        <<" inspection_rank="<<better+1<<" in_E="<<contains(result.evaluated)
        <<" shortlisted="<<contains(result.shortlist)<<'\n'<<std::flush;
  }
  return 0;
}catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}}
