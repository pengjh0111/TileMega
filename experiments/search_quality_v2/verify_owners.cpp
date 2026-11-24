// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace tilemega;
using namespace tilemega::solver;

int main(int argc, char** argv) {
  try {
    if (argc != 6)
      throw std::invalid_argument("usage: tilemega-sq2-verify-owners target source-root cost-table search-receipt output");
    if (std::ifstream(argv[5])) throw std::invalid_argument("retain previous owner verification");
    auto record = json::ParseFile(argv[4]);
    auto const& input = record.At("input");
    auto const& seed_path = input.At("seed").AsString("seed");
    auto seed = Sm90DenseCoordinates::FromJson(json::ParseFile(seed_path));
    analysis::IslContext isl;
    mlir::MLIRContext context;
    auto imported = ImportSm90Dense(input.At("export").AsString("export"), seed, context);
    auto work = Sm90DenseWorkloadFromPlan(imported.plan, 1);
    auto target = TargetSpec::FromJson(argv[1]);
    auto constants = Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]), argv[2]);
    if (!constants.StaleHeaders(argv[2]).empty()) throw std::runtime_error("stale calibration");
    Sm90DenseCostModel cost(target, constants, argv[3]);
    Sm90NativeCtgCache cache;
    json::Array rows;
    bool all = true;
    double replay_seconds = 0;
    for (auto const& row : record.At("shortlist").AsArray("shortlist")) {
      auto begin = std::chrono::steady_clock::now();
      auto q = Sm90DenseCoordinates::FromJson(row.At("coordinates"));
      auto result = Sm90NativeCtgScore(imported, work, q, cost, context, cache);
      auto const& problem = result.problem;
      auto native = Sm90NativeCandidateTracks(problem);
      auto refined = DiagnoseSm90NativeSq2Owners(problem);
      double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      replay_seconds += elapsed;
      auto fixed = RefineAndReplaySm90Native(problem, native);
      bool same_owner = refined.placement.worker_of_task == native;
      bool same_replay_order = refined.placement.order == fixed.placement.order;
      bool same_typed_order = true;
      int first_track = -1;
      for (std::size_t w = 0; w < refined.placement.order.size(); ++w) {
        std::vector<std::pair<int, int>> sequence;
        for (auto [s, t] : refined.placement.order[w]) {
          auto const& category = problem.spaces[s].category;
          if (category == "gemm" || category == "attention") sequence.emplace_back(s, t);
        }
        if (!std::is_sorted(sequence.begin(), sequence.end())) {
          same_typed_order = false;
          if (first_track < 0) first_track = int(w);
        }
      }
      bool same_score = refined.replay.makespan_ns == fixed.replay.makespan_ns;
      bool supported = same_owner && same_typed_order && same_replay_order && same_score;
      all = all && supported;
      rows.emplace_back(json::Object{{"coordinates", q.ToJson()}, {"same_owner", same_owner},
          {"same_typed_order", same_typed_order}, {"same_replay_order", same_replay_order},
          {"same_replay_score", same_score}, {"first_unordered_track", first_track},
          {"materialization_and_eq5_seconds", elapsed},
          {"eq5_replay_ns", refined.replay.makespan_ns}, {"native_replay_ns", fixed.replay.makespan_ns},
          {"compatible_with_native_template", supported}});
    }
    std::ofstream(argv[5]) << json::Value(json::Object{
        {"all_shortlist_plans_match", all}, {"shortlist", rows},
        {"materialization_and_eq5_seconds", replay_seconds},
        {"scope", "Eq. (5) owners and order under compact native CTG completion contracts. Typed bodies execute the compiled cyclic template."}}).Dump(2);
    std::cout << "shortlist plans=" << rows.size() << " all match=" << all << '\n';
    return all ? 0 : 2;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
