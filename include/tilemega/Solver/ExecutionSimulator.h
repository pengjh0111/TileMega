// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.7.2 (executor semantics: q[0..n) by ascending sigma, head,
//                W = 1 is strict FIFO), §4.4.1 (the nine-lane resource vector),
//                §4.3 (resident-only).
//
// EX-S1: the L2 cost model is an execution simulator, not a closed form.  Round
// one priced a schedule as `max(work_lb, queue_lb, critical_path)` and
// underestimated mode 5 by 1.7-3.7x (F-139), because none of those three bounds
// can express the thing that actually costs: a worker that is idle because the
// task at its head is not ready yet while a task it could have run sits behind
// it.  Only replaying the queue discipline shows that.
//
// What is simulated, precisely:
//   * each worker runs `plan.queue[w]` in sigma order, one task at a time
//     (W = 1, so a blocked head blocks the worker -- this is the whole point);
//   * a task starts at max(worker free, every predecessor's end + hop), where
//     the hop is zero for a same-worker predecessor;
//   * `hop_ns(N, R)` comes from the measured curve in
//     docs/experiments/SIMULATOR/ -- the primitive the runtime executes,
//     backoff included;
//   * co-resident workers on one SM share that SM's busiest pipe, so a task's
//     duration is stretched by the aggregate nine-lane demand of the set it
//     runs with.
//
// What is not simulated, and must not be read into its output: DRAM and L2 are
// charged per task through the lane vector, never as a device-wide queue; there
// is no instruction-level model; W > 1 is out of scope this round (EX-E2) and
// `Simulate` rejects a window other than 1 rather than quietly running FIFO.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/HopCurve.h>
#include <tilemega/Solver/PlanMaterialize.h>

namespace tilemega::solver {

struct SimulatorOptions {
  int sms = 0;             ///< 0 maps every worker to its own SM
  int ctas_per_sm = 1;     ///< resident CTAs per SM, so `sms * ctas_per_sm >= grid`
  int window = 1;          ///< §5.7.2 W; anything but 1 is rejected (EX-E2)
  /// Price hops at `c0` alone.  Used to show that the choice of (N, R) context
  /// below moves the makespan by less than the curve's own uncertainty.
  bool flat_hop = false;
  /// Ignore the lane vectors and stretch a co-resident set by its size.  This
  /// is what a cost model that priced every task at `ctas_per_sm` already
  /// assumes, kept as the control arm for the lane-stretch model.
  bool proportional_sharing = false;
};

struct SimulatorInput {
  codegen::RuntimeTaskGraph const* graph = nullptr;
  /// Solo duration of each runtime node, one resident CTA per SM.  Indexed by
  /// the graph's flat node id, i.e. `stage_offsets[stage] + task`.
  std::vector<double> task_ns;
  /// §4.4.1 demand of each node.  Empty, or an all-zero row, falls back to
  /// proportional sharing for that node.
  std::vector<ResourceVector> task_lanes;
  /// Physical worker -> SM.  Empty means `w % sms`.
  std::vector<int> worker_sm;
};

struct SimulatedTask {
  double start_ns = 0.0;
  double end_ns = 0.0;
  /// Time this task's worker sat at this head with the task not yet ready.
  /// Summed over tasks this is the schedule's cost of being wrong, and it is
  /// the quantity round one's three bounds could not see.
  double block_ns = 0.0;
  double stretch = 1.0;
  int worker = -1;
};

struct SimulatorResult {
  double makespan_ns = 0.0;
  double total_work_ns = 0.0;      ///< sum of stretched durations
  double solo_work_ns = 0.0;       ///< sum of `task_ns`, the work lower bound
  double total_block_ns = 0.0;
  double busiest_worker_ns = 0.0;  ///< the queue lower bound, stretched
  int busiest_worker = -1;
  double critical_path_ns = 0.0;   ///< longest end-to-end chain in the result
  long cross_worker_edges = 0;
  long same_worker_edges = 0;
  std::vector<SimulatedTask> tasks;
};

/// False, with `*error` set, on a malformed plan, a window other than 1, or a
/// queue order that deadlocks.  A deadlock is a real finding, not a simulator
/// bug: it means the (pi, sigma) under test violates §5.7.3 L-a, and it is
/// reported as such rather than by running until a step budget expires.
bool SimulateExecution(SimulatorInput const& input, MaterializedPlan const& plan,
                       SimulatorOptions const& options, HopCurve const& hop,
                       SimulatorResult* out, std::string* error);

}  // namespace tilemega::solver
