// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.7.1 (Plan = (pi, sigma, W, policy, sync, kappa)),
//                §5.7.3 (legality L-a and L-c), §5.7.4 (the host materializes,
//                the solver decides), §8.11.
//
// The materialized form of a Plan's (pi, sigma) for one bound theta.  The host
// calls this after binding seq/past, the split-K rewrite and the resident grid;
// it is the same routine the solver uses offline, so a template mode is
// evaluated in exactly one place.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Dialect/CouplingGraph/PlacementPlan.h>

namespace tilemega::solver {

/// One queue entry, in sigma order.
struct PlanQueueItem {
  std::uint32_t stage;
  int logical;
};

struct PlanBalancedStats {
  int max_queue = 0;
  int baseline_max_queue = 0;
  long same_worker_edges = 0;
  long fence_free_producers = 0;
};

/// pi as `owner`, sigma as `slot`, and the queues sigma already implies.
/// `slot` is dense per worker, so `queue[owner[s][t]][slot[s][t]]` is `(s, t)`.
struct MaterializedPlan {
  std::vector<std::vector<int>> owner;
  std::vector<std::vector<int>> slot;
  std::vector<std::vector<PlanQueueItem>> queue;
  /// `rotate` only: the per-stage prefix-sum base, kept so the host can dump
  /// the same numbers the offline check recomputes by hand.
  std::vector<int> rotate_base;
  PlanBalancedStats balanced;
  bool has_balanced_stats = false;
};

struct PlanRequest {
  dialect::PlacementMode mode = dialect::PlacementMode::kLegacyGridStride;
  std::vector<std::int64_t> params;
  int grid = 0;
  /// Active task count per stage, after the split-K rewrite.
  std::vector<int> counts;
  /// Stage execution order; `rotate` walks its prefix sums and the stage-major
  /// modes number sigma along it.
  std::vector<std::uint32_t> stage_order;
  /// Logical worker -> physical CTA, i.e. the Placement.cuh map already applied.
  std::vector<int> physical_worker;
  /// Required by the modes that read the task DAG; ignored by the rest.
  codegen::RuntimeTaskGraph const* graph = nullptr;
};

/// False, with `*error` set, for a mode this build cannot materialize or a
/// request that contradicts itself.  Callers must treat that as fatal (H5).
bool MaterializePlanPlacement(PlanRequest const& request, MaterializedPlan* out,
                              std::string* error);

/// §5.7.3 L-a (the union of task edges and same-worker queue edges is acyclic)
/// and L-c (a same-worker producer holds the smaller sigma).  L-b is checked
/// before the grid is fixed (`ResidentScheduleLegal`), L-d collapses to the
/// current hoisting rules at W = 1, and L-e needs the event tables, so those
/// three stay with the caller.
bool CheckPlanLegality(codegen::RuntimeTaskGraph const& graph,
                       MaterializedPlan const& plan, std::string* error);

}  // namespace tilemega::solver
