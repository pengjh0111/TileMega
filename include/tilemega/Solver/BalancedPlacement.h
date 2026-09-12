// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <vector>

namespace tilemega::solver {
struct TaskPlacement {
  std::vector<int> worker;
  int max_queue = 0, max_worker_span = 0;
  long same_worker_edges = 0, fence_free_producers = 0;
};
// The task DAG must come from the shared logical-to-runtime projection.
// All producers precede consumers in order; queue edges are validated too.
TaskPlacement BalanceTaskPlacement(std::vector<std::vector<int>> const& successors,
    std::vector<int> const& order, std::vector<int> const& preferred_worker,
    int workers, int max_queue);
}  // namespace tilemega::solver
