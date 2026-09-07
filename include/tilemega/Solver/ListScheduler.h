// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §4.3 / P4.8 Place -- list scheduling on the layered DAG.
//
// The queue-driven L2 path uses this order as the compact, variant-exact input
// from which the host materializes each physical worker's TaskRef queue.  L1
// still executes one stage at a time with a grid barrier.  Both paths require a
// topological stage order, but independent stages can be adjacent in either
// order; `levels` measures the DAG's critical-path depth.
//
// Priority inside a level is critical-path height, the standard list-scheduling
// rule: a node with a longer remaining path is scheduled first, so the nodes
// that can absorb a wait are the ones left over.
#pragma once

#include <vector>

namespace tilemega::solver {

struct ScheduleStats {
  int nodes = 0;
  int levels = 0;          ///< critical-path depth in stages
  int widest_level = 0;    ///< most stages available at one topological level
  int barriers_saved = 0;  ///< legacy L1 interpretation: nodes - levels
};

struct ScheduleSafety {
  int max_dependency_span = 0;
};

class ListScheduler {
 public:
  /// `successors[i]` lists the nodes that depend on `i`.  Throws
  /// std::invalid_argument on a cycle or an out-of-range successor rather than
  /// returning a partial order that would look like a schedule.
  std::vector<int> Levels(std::vector<std::vector<int>> const& successors) const;

  /// Longest path from each node to a sink.  This is the priority function.
  std::vector<int> Heights(std::vector<std::vector<int>> const& successors) const;

  /// Earliest level first, greatest height inside a level, index to break the
  /// remaining ties so the schedule is deterministic.
  std::vector<int> Schedule(std::vector<std::vector<int>> const& successors,
                            ScheduleStats* stats = nullptr) const;

  /// Verify that `order` is a complete topological permutation and compute
  /// each coupling's producer-to-consumer span in that launch order.  A cycle,
  /// duplicate/missing stage, or backwards wait is rejected at generation
  /// time rather than becoming a persistent-kernel hang.
  ScheduleSafety Validate(
      std::vector<std::vector<int>> const& successors,
      std::vector<int> const& order) const;
};

}  // namespace tilemega::solver
