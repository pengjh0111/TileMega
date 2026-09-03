// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §4.3 / P4.8 Place -- list scheduling on the layered DAG.
//
// The megakernel executes one stage at a time with a grid barrier between
// consecutive stages.  That order is a topological order of the stage DAG, but
// it is not the only one, and it is not the shortest: two stages with no path
// between them can share a barrier interval.  The number of intervals -- the
// DAG's critical path in stages -- is therefore the floor on how many barriers
// the model can possibly execute, and `levels` measures it.
//
// Priority inside a level is critical-path height, the standard list-scheduling
// rule: a node with a longer remaining path is scheduled first, so the nodes
// that can absorb a wait are the ones left over.
#pragma once

#include <vector>

namespace tilemega::solver {

struct ScheduleStats {
  int nodes = 0;
  int levels = 0;          ///< barrier intervals after packing; the DAG's depth
  int widest_level = 0;    ///< most stages that could share one interval
  int barriers_saved = 0;  ///< nodes - levels, the whole prize P4.8 plays for
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
};

}  // namespace tilemega::solver
