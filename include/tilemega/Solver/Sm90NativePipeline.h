// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/Sm90NativeFlow.h>
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <tilemega/Solver/Sm90DenseSearch.h>
#include <string>
#include <optional>
#include <functional>
#include <vector>

namespace tilemega::solver {

// P3: Algorithm 1 of the draft on the native path. Coordinate descent from
// several seeds under Eq. (8)'s PoolScore, a shortlist that keeps structural
// variety, then Eq. (10)'s replay over the skeleton's template placement.
struct Sm90NativePipelineOptions {
  int shortlist = 6;              // continuation 1 section 3: top 6 by score
  int structural_extra = 4;       // plus the best of each unseen structural class
  int finalists = 4;              // ranked by replay
  int per_class = 2;              // at most this many finalists share a class
  double budget_ms = 600000;      // --search-budget-ms
  // AB-2: sum isolated evaluations with the same prices and evaluator.
  bool operator_sum = false;
  // AB-4a: forbid the structural rows of Table 1 other than Retile.
  bool retile_only = false;
  // AB-4b: every coupling becomes an aggregate (V3) boundary.
  bool realize_aggregate = false;
  std::optional<Sm90DenseCoordinates> pinned_theta;
};

struct Sm90NativeReplayResult {
  FlowPlacement placement;
  std::vector<std::vector<double>> estimated_start_ns,estimated_finish_ns;
  FlowResult replay;
};

// Eq. (9)'s candidate set is the native skeleton's singleton track. Offsets
// account for families sharing a fused stage's grid-stride task list.
std::vector<std::vector<int>> Sm90NativeCandidateTracks(
    FlowProblem const&,std::vector<int> const& offsets={});
Sm90NativeReplayResult RefineAndReplaySm90Native(FlowProblem const&,
    std::vector<std::vector<int>> const& candidate_tracks);

struct Sm90NativePipelineCandidate {
  Sm90DenseCoordinates coordinates;
  double pool_score_ns = 0;
  double replay_ns = 0;
  double evaluation_ms = 0;
  std::string origin;            // which seed or move produced it
  std::string structural_class;  // continuation 1 section 3's structural key
};

struct Sm90NativePipelineResult {
  std::vector<Sm90NativePipelineCandidate> evaluated;  // E, in discovery order
  std::vector<Sm90NativePipelineCandidate> shortlist;  // Q_short
  std::vector<Sm90NativePipelineCandidate> finalists;  // S after replay
  // Algorithm 1's trajectory, one list per seed: the score after each accepted
  // move. Monotone within a seed; a later seed starts wherever it starts.
  std::vector<std::vector<double>> descent;
  std::size_t memo_hits = 0, evaluations = 0;
  double seconds = 0;
};

// Everything but the per-family geometry: the coordinates that change which
// tasks exist or how they synchronize (continuation 1 section 3).
std::string Sm90NativeStructuralClass(Sm90DenseCoordinates const&);

Sm90NativePipelineResult SolveSm90NativePipeline(
    Sm90DenseSearchSpace const& space, Sm90DenseWorkload const& work,
    Sm90DenseCostModel const& cost, Sm90NativePipelineOptions const& options,
    std::function<Sm90NativeFlowResult(Sm90DenseCoordinates const&,bool)> const& evaluate,
    std::vector<Sm90DenseCoordinates> const& seeds = {});
}  // namespace tilemega::solver
