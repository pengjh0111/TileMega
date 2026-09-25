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
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/HopCurve.h>
#include <tilemega/Solver/PlanMaterialize.h>

namespace tilemega::solver {

struct SimulatorOptions {
  bool dram_fluid = false;
  double dram_gbps = 0;
  double dram_floor_ns = 0;
  bool all_external_miss = false;
  bool no_external_dram = false;
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
  /// Once per publishing producer, in addition to consumer visibility delay.
  /// Zero preserves the historical simulator exactly. Measured node times
  /// already include co-resident stretching when observed_task_times is true.
  double publication_ns = 0.0;
  double consumer_wait_ns = 0.0;
  bool observed_task_times = false;
};

// Consecutive node runs use half-open intervals; sparse rows retain their
// original storage when interval pairs would be larger. Traversal preserves
// order and duplicate entries, including non-topological node numbering.
class SuccessorIntervals {
 public:
  explicit SuccessorIntervals(std::vector<int> const& nodes) : size_(nodes.size()) {
    std::size_t runs = 0;
    bool encodable = true;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      if (i == 0 || std::int64_t(nodes[i]) != std::int64_t(nodes[i - 1]) + 1) ++runs;
      encodable &= nodes[i] != std::numeric_limits<int>::max();
    }
    intervals_ = encodable && 2 * runs < nodes.size();
    if (!intervals_) { storage_ = nodes; return; }
    storage_.reserve(2 * runs);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      if (i == 0 || std::int64_t(nodes[i]) != std::int64_t(nodes[i - 1]) + 1) {
        storage_.push_back(nodes[i]);
        storage_.push_back(nodes[i] + 1);
      } else storage_.back() = nodes[i] + 1;
    }
  }
  std::size_t size() const { return size_; }
  std::size_t stored_ints() const { return storage_.size(); }
  template<class Visitor> void Visit(Visitor&& visitor) const {
    if (intervals_) {
      for (std::size_t i = 0; i < storage_.size(); i += 2)
        for (int node = storage_[i]; node < storage_[i + 1]; ++node) visitor(node);
    } else for (int node : storage_) visitor(node);
  }
  bool Equals(std::vector<int> const& nodes) const {
    if (nodes.size() != size_) return false;
    bool equal = true;
    std::size_t i = 0;
    Visit([&](int node) { equal &= node == nodes[i++]; });
    return equal;
  }
 private:
  std::vector<int> storage_;
  std::size_t size_ = 0;
  bool intervals_ = false;
};

/// Identical successor sets are shared dependency groups. Preparation is
/// reusable while the source graph is immutable; queue placement remains free.
struct PreparedExecutionGraph {
  codegen::RuntimeTaskGraph const* source = nullptr;
  std::vector<SuccessorIntervals> successors;
  std::vector<std::vector<int>> producers;
  std::vector<int> group_of_node, producer_count;
  bool forward_node_order=false;
};
bool PrepareExecutionGraph(codegen::RuntimeTaskGraph const& graph,
    PreparedExecutionGraph* out, std::string* error);

// Immutable queue readiness can be reused across different calibrated task
// prices. Rebuild this object whenever the graph or any worker queue changes.
struct PreparedExecutionPlan {
  codegen::RuntimeTaskGraph const* graph=nullptr;
  MaterializedPlan const* plan=nullptr;
  std::vector<std::vector<int>> queue;
  std::vector<int> owner,unmet,cross_fanout,queue_next;
  std::vector<unsigned char> cross_input;
  long cross_edges=0,same_edges=0;
  bool forward_node_order=false;
};
bool PrepareExecutionPlan(PreparedExecutionGraph const& graph,MaterializedPlan const& plan,
    PreparedExecutionPlan* out,std::string* error);

struct SimulatorInput {
  std::vector<TaskPriceParts> task_price_parts;
  codegen::RuntimeTaskGraph const* graph = nullptr;
  /// Solo duration of each runtime node, one resident CTA per SM.  Indexed by
  /// the graph's flat node id, i.e. `stage_offsets[stage] + task`.
  std::vector<double> task_ns;
  /// §4.4.1 demand of each node.  Empty, or an all-zero row, falls back to
  /// proportional sharing for that node.
  std::vector<ResourceVector> task_lanes;
  /// Part of `task_ns` a node spends on its read-only frontier (§5.3.1's
  /// Prefetch phase).  Empty means no node pipelines, which is what a build
  /// without the mechanism reports.  A same-worker adjacency is pipelinable
  /// only where this is positive; the rest of the queue edges cost the full
  /// duration, which is the distinction §5.2's edge cost has to make.
  std::vector<double> prefetch_ns;
  /// Physical worker -> SM.  Empty means `w % sms`.
  std::vector<int> worker_sm;
  /// Actual runtime publication requirement per task. Empty uses the minimal
  /// graph requirement (at least one cross-worker consumer). Runtime stage-wide
  /// event flags may publish more tasks; a dump-backed caller supplies them.
  std::vector<unsigned char> publication_required;
  std::vector<unsigned char> consumer_wait_required;
  PreparedExecutionGraph const* prepared_graph = nullptr;
  PreparedExecutionPlan const* prepared_plan = nullptr;
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
bool SimulateFluidExecution(SimulatorInput const& input,MaterializedPlan const& plan,
    SimulatorOptions const& options,HopCurve const& hop,SimulatorResult* out,std::string* error);

/// False, with `*error` set, on a malformed plan, a window other than 1, or a
/// queue order that deadlocks.  A deadlock is a real finding, not a simulator
/// bug: it means the (pi, sigma) under test violates §5.7.3 L-a, and it is
/// reported as such rather than by running until a step budget expires.
bool SimulateExecution(SimulatorInput const& input, MaterializedPlan const& plan,
                       SimulatorOptions const& options, HopCurve const& hop,
                       SimulatorResult* out, std::string* error);

}  // namespace tilemega::solver

namespace tilemega::solver {

/// Immutable task costs and semantic-DAG bounds shared by placement candidates.
/// Preparation walks the DAG once; its time is reported separately from a
/// placement evaluation. Queue edges are deliberately not semantic edges.
struct PreparedPlanBounds {
  std::vector<int> stage_offsets;
  std::vector<double> task_ns,prefetch_ns;
  double work_ns = 0;
  double critical_path_ns = 0;
  PreparedExecutionGraph graph;
};
struct PlanBounds {
  double work_lb_ns = 0;
  double queue_lb_ns = 0;
  double critical_path_ns = 0;
  double lower_bound_ns = 0;
  double binding_path_ns = 0;  ///< CG plus FIFO queue edges; semantic CP stays separate.
};
struct RankedPlan {
  std::size_t index = 0;
  PlanBounds bounds;
  bool simulated = false;
  double makespan_ns = 0;
};

/// Which nodes of `plan` overlap their frontier fetch with the slot before them
/// on the same worker: exactly the adjacencies the two bounds above discount,
/// so the flags written back to CG cannot drift from the price the plan was
/// chosen at.  All zero when `input.prefetch_ns` is empty.
std::vector<unsigned char> PipelinedSlots(SimulatorInput const& input,
                                          MaterializedPlan const& plan);

bool PreparePlanBounds(SimulatorInput const& input, PreparedPlanBounds* out,
                       std::string* error);
bool EvaluatePlanBounds(PreparedPlanBounds const& input,
                        MaterializedPlan const& plan, PlanBounds* out,
                        std::string* error);
/// Coarse sort by max(work, DAG, queue). Boundary ties remain eligible, since
/// a zero-synchronization bound cannot distinguish tied placements. Only the
/// eligible top-k set is fully simulated; a measured rank is never invented
/// for an unmeasured candidate. The caller retains normal Plan legality checks.
bool RankPlans(SimulatorInput const& input, PreparedPlanBounds const& prepared,
               std::vector<MaterializedPlan const*> const& plans,
               SimulatorOptions const& options, HopCurve const& hop,
               std::size_t top_k, std::vector<RankedPlan>* out,
               std::string* error);

}  // namespace tilemega::solver
