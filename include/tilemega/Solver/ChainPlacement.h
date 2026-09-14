// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.7.1 (the solver decides (pi, sigma)), §5.7.2, §5.7.3 L-a/L-c.
//
// EX-S2c: place the critical chain of the runtime task DAG on one worker.
//
// Why this and not `ScheduleByEarliestFinish`: EFT minimises each task's own
// finish time, which is a local objective -- it will hand a task to whichever
// worker frees up first even when that costs a hop on the critical path.  The
// round-two measurement is that the two cancel almost exactly (net 0.2-2.7%
// against rotate), because EFT buys balance and pays it back in hops.  The
// lever this round is the *number of hops on the critical path*, which is the
// one term that falls on both sm_89 and sm_120: on sm_120 the per-hop cost is
// already at a ~400 ns floor, so nothing but fewer hops can move it.
//
// The chain that carries the makespan is therefore placed whole on a single
// worker, where its internal edges cost no hop at all, and the remaining graph
// is clustered the same way until a chain is no longer longer than the work an
// average free worker still has to do.
//
// Like EFT this is a decision about one bound theta: durations come from
// `CostModel::TaskInstanceNs` (or from a round-one trace) at that theta, so a
// schedule computed for one (seq, past) is not a schedule for another.
#pragma once

#include <string>
#include <vector>

#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Solver/HopCurve.h>

namespace tilemega::solver {

struct ChainRequest {
  codegen::RuntimeTaskGraph const* graph = nullptr;
  /// Solo duration of each runtime node, indexed by the graph's flat node id.
  /// §6.2 asks for both sources -- the cost model and the round-one trace --
  /// so the caller chooses and the difference is reported, not averaged.
  std::vector<double> task_ns;
  int grid = 0;
  int sms = 0;                 ///< 0 gives every worker its own SM
  int ctas_per_sm = 1;
  std::vector<int> worker_sm;  ///< empty means `w % sms`
  HopCurve hop;
  /// Stop extracting chains when the next one is no longer than this multiple
  /// of the work an average still-empty worker would otherwise carry.  At 1.0
  /// extraction stops as soon as chaining stops shortening the longest queue,
  /// which is the point of the exercise: past it, a chain is just a long queue.
  double chain_stop_ratio = 1.0;
  /// Re-extract after measuring.  §6.1 ranks a path in work and hops; the path
  /// that ends up critical is a different, longer one manufactured by queueing,
  /// so each extra round re-scores every node by the delay the previous pass
  /// measured it spending waiting for its worker rather than for its data.  The
  /// best pass by makespan wins, so a round cannot make the answer worse.  Zero
  /// is one plain pass and is the default (H2).
  int feedback_rounds = 0;
  /// Cap a filled worker in queue count and in work as well as in time.  The
  /// caps predate the earliest-finish fill and answer pathologies it does not
  /// have: that fill prices a queue position by when the task ahead of it
  /// finishes, which is the thing a count cannot see.  Kept as a switch so the
  /// round reports both arms rather than asserting one -- with the caps on, the
  /// count cap binds at the cells where chaining loses (736 forced placements
  /// at mha4 s128, 4640 at real s128, 0 wherever it wins), and a forced
  /// placement is one that ignored the time estimate.
  bool cap_fill = true;
};

struct ChainSchedule {
  std::vector<int> worker;   ///< pi, per node
  std::vector<int> slot;     ///< sigma, per node, dense per worker
  std::vector<double> start_ns;
  std::vector<double> end_ns;
  double makespan_ns = 0.0;
  /// §6.4.  `spine_length_ns` is the extracted critical chain, which is also
  /// the capacity cap the fill step works to -- not round two's
  /// `baseline_max_queue`, whose pathology is F-129.  `max_queue_ns` is time,
  /// not a task count, because a queue of 400 elementwise tasks and a queue of
  /// three GEMMs are not the same load.
  double spine_length_ns = 0.0;
  double max_queue_ns = 0.0;
  int chain_count = 0;
  /// Which extracted chain holds each node, -1 for one the fill step placed.
  /// Chain 0 is the spine.  §6.4's per-chain lengths come from this, and it is
  /// what separates "the spine is not contiguous" from "the spine is contiguous
  /// and the simulated critical path runs somewhere else entirely".
  std::vector<int> chain_of;
  /// How often a task that is not a member of a worker's chain sorts inside
  /// that chain's span, so the chain is no longer contiguous on its queue.  At
  /// a window of one that is what puts a foreign task in front of a chain
  /// member, and it is a performance property, not a legality one.
  int chain_interleaves = 0;
  /// §6.3 asks for the split count, and it is always zero here -- reported so,
  /// not omitted.  A chain is a path in the task DAG, so the queue edges
  /// between its consecutive members are task edges already, and every queue is
  /// ordered by topological index, so the union of task and queue edges is a
  /// subset of one topological order: no chain placement can close a cycle and
  /// there is nothing for a split to repair.  §6.3's hazard is a cycle in the
  /// *contracted* graph, which is strictly stronger than L-a and not the
  /// condition that gates a Plan.
  int split_count = 0;
  /// Tasks the fill step had to hand a worker already at the queue-count cap,
  /// which is max(spine nodes, ceil(nodes / grid)).  The cap is the spine in
  /// *count* and not only in work because at a window of one a queue is strict
  /// FIFO: a long queue of small tasks serialises waiting even when its total
  /// work is small.  Measured at mha4 s128 with a time-only cap: the 362813 ns
  /// spine admitted 864 of the 419.84 ns filled tasks onto one worker, and 565
  /// of the 614 critical-path edges became queue steps, against rotate's 32.
  /// The count cap alone still lost (158 queue edges, 602997 ns against rotate's
  /// 437124), so a filled worker is bounded in work by total work / grid rather
  /// than by the spine, which only the extracted chains earn.
  int fill_overflows = 0;
};

/// False, with `*error` set, on a malformed request or a cyclic task DAG.
/// The result still has to pass `CheckPlanLegality`; the caller runs it rather
/// than trusting this one, exactly as the EFT path does.
bool ScheduleByCriticalChain(ChainRequest const& request, ChainSchedule* out,
                             std::string* error);

}  // namespace tilemega::solver
