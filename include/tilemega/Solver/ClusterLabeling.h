// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §4.3 cluster-label assignment (Phase 3 stub).
#pragma once
#include <vector>
namespace tilemega::solver {
class ClusterLabeling { public: std::vector<int> Assign(int node_count, int max_cluster_size) const; };
}  // namespace tilemega::solver
