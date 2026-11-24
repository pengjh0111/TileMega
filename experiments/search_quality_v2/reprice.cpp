// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace tilemega;
using namespace tilemega::solver;

int main(int argc, char** argv) {
  try {
    if (argc != 8)
      throw std::invalid_argument("usage: tilemega-sq2-reprice target root cost-table export coordinate actual output");
    if (std::ifstream(argv[7])) throw std::invalid_argument("retain previous capacity receipt");
    analysis::IslContext isl;
    auto target = TargetSpec::FromJson(argv[1]);
    auto constants = Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]), argv[2]);
    if (!constants.StaleHeaders(argv[2]).empty()) throw std::runtime_error("stale calibration");
    Sm90DenseCostModel cost(target, constants, argv[3]);
    auto seed = Sm90DenseCoordinates::FromJson(json::ParseFile(argv[5]));
    mlir::MLIRContext context;
    auto imported = ImportSm90Dense(argv[4], seed, context);
    auto work = Sm90DenseWorkloadFromPlan(imported.plan, 1);
    Sm90DenseSearchSpace space{work.shapes, work.batch, work.head_dim, work.queries, target, work.hkv};
    space.qk_norm_row = work.qk_norm_row;
    Sm90NativeCtgCache cache;
    auto revised = RepriceSm90NativeResidency(space, seed, std::stoi(argv[6]), false,
        [&](auto const& c, bool isolated) {
          auto result = Sm90NativeCtgScore(imported, work, c, cost, context, cache, isolated);
          auto native = RefineAndReplaySm90Native(result.problem, Sm90NativeCandidateTracks(result.problem));
          auto refined = DiagnoseSm90NativeSq2Owners(result.problem);
          if (native.placement.worker_of_task != refined.placement.worker_of_task ||
              native.placement.order != refined.placement.order ||
              native.replay.makespan_ns != refined.replay.makespan_ns)
            throw std::runtime_error("capacity feedback requires a different owner plan from the executor");
          return result;
        });
    json::Value out = json::Object{{"legal", bool(revised)}};
    if (revised) {
      out.Set("coordinates", revised->coordinates.ToJson());
      out.Set("pool_ns", revised->pool_score_ns);
      out.Set("replay_ns", revised->replay_ns);
      out.Set("evaluation_ms", revised->evaluation_ms);
    }
    std::ofstream(argv[7]) << out.Dump(2);
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
