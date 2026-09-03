// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ClusterLabeling.h>
#include <tilemega/Solver/ListScheduler.h>
#include <algorithm>
namespace tilemega::solver {
std::vector<int> ClusterLabeling::Assign(int count, int max_cluster_size) const {
  std::vector<int> labels(std::max(0, count));
  int width = std::max(1, max_cluster_size);
  for (int i = 0; i < count; ++i) labels[i] = i / width;
  return labels;
}
std::vector<int> ListScheduler::Schedule(std::vector<std::vector<int>> const& graph) const {
  std::vector<int> order(graph.size());
  for (std::size_t i = 0; i < graph.size(); ++i) order[i] = static_cast<int>(i);
  return order;
}
}  // namespace tilemega::solver
