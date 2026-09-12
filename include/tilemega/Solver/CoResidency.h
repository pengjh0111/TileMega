// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §4.4.1 (the nine-lane resource vector), §4.3 (residency).
//
// The co-residency interference model, shared so that the simulator that
// replays a plan and the scheduler that chooses one charge a shared SM the
// same way.
#pragma once

#include <algorithm>
#include <vector>

#include <tilemega/Solver/CostModel.h>

namespace tilemega::solver {

/// The SM's busiest pipe is shared, so a co-resident set cannot finish its pipe
/// work faster than the set's aggregate demand on that pipe.  `stretch` is that
/// aggregate over the largest single member's solo bottleneck, which is 1 for a
/// set of one and |S| for a set of identical tasks bottlenecked on one lane.
inline double LaneStretch(std::vector<ResourceVector> const& lanes,
                          std::vector<int> const& members) {
  double worst_sum = 0.0, worst_solo = 0.0;
  for (int lane = 0; lane < ResourceVector::kLaneCount; ++lane) {
    double sum = 0.0;
    for (int m : members) sum += lanes[m][static_cast<ResourceVector::Lane>(lane)];
    worst_sum = std::max(worst_sum, sum);
  }
  for (int m : members) worst_solo = std::max(worst_solo, lanes[m].Bottleneck());
  if (worst_solo <= 0.0) return static_cast<double>(members.size());
  return std::max(1.0, worst_sum / worst_solo);
}

}  // namespace tilemega::solver
