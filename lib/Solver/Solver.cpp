// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ClusterLabeling.h>
#include <algorithm>
namespace tilemega::solver {
std::vector<int> ClusterLabeling::Assign(int count, int max_cluster_size) const {
  std::vector<int> labels(std::max(0, count));
  int width = std::max(1, max_cluster_size);
  for (int i = 0; i < count; ++i) labels[i] = i / width;
  return labels;
}
}  // namespace tilemega::solver
