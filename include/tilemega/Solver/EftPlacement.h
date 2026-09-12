// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.7.1 (the solver decides (pi, sigma)), §5.7.2 (W = 1, so a
//                worker's order is its queue), §5.7.3 L-b/L-c, §8.11.
//
// EX-S2: earliest-finish-time list scheduling over the runtime task DAG.
//
// Why this and not the round-one `ListScheduler`: that one ranks by edge count
// and only balances queue lengths, which is the axis modes 0 and 5 already sit
// at the two ends of (F-139).  This one ranks in ns and chooses the worker that
// finishes the task soonest, so the same objective trades locality against
// balance instead of picking one: a cross-worker predecessor costs a hop, a
// long queue costs waiting, and whichever is larger loses.  No locality
// penalty term is added on top -- the first question is what plain EFT does.
//
// The schedule is a decision about one bound theta: durations come from
// `CostModel::TaskInstanceNs` at that theta, so a schedule computed for one
// (seq, past) is not a schedule for another.  Callers pin it (§5.7.4).
#pragma once

#include <string>
#include <vector>

#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/HopCurve.h>

namespace tilemega::solver {

struct EftRequest {
  codegen::RuntimeTaskGraph const* graph = nullptr;
  /// Solo duration of each runtime node, one resident CTA per SM, indexed by
  /// the graph's flat node id.  Same input the simulator prices a plan with.
  std::vector<double> task_ns;
  /// §4.4.1 demand per node.  Empty falls back to proportional sharing, which
  /// is what whole models get: only GEMM stages expose a lane vector.
  std::vector<ResourceVector> task_lanes;
  int grid = 0;
  int sms = 0;            ///< 0 gives every worker its own SM
  int ctas_per_sm = 1;
  std::vector<int> worker_sm;  ///< empty means `w % sms`
  HopCurve hop;
};

struct EftSchedule {
  std::vector<int> worker;      ///< pi, per node
  std::vector<int> slot;        ///< sigma, per node, dense per worker
  std::vector<double> start_ns;
  std::vector<double> end_ns;
  /// The greedy's own estimate.  It is a by-product, not the cost model: the
  /// simulator prices the resulting plan, and the two disagree wherever the
  /// greedy's causal approximation of co-residency does.
  double makespan_ns = 0.0;
};

/// False, with `*error` set, on a malformed request or a cyclic task DAG.
/// `request.grid` is the resident grid, so L-b holds by construction; the
/// caller still checks `ResidentScheduleLegal` before fixing that grid.
bool ScheduleByEarliestFinish(EftRequest const& request, EftSchedule* out,
                              std::string* error);

}  // namespace tilemega::solver
