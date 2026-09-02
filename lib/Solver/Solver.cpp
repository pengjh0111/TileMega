// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/AlignmentPropagation.h>
#include <tilemega/Solver/ChainDP.h>
#include <tilemega/Solver/ClusterLabeling.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/ListScheduler.h>
#include <algorithm>
namespace tilemega::solver {
bool AlignmentPropagation::Compatible(TileCandidate const& a, TileCandidate const& b) const {
  return a.n == 0 || b.m == 0 || a.n == b.m;
}
double CostModel::Evaluate(TileCandidate const& c, TargetSpec const& target) const {
  // TODO(P4.2): calibrated latency plus coupling-aware communication cost.
  return static_cast<double>(c.m) * c.n * c.k /
         std::max(1, target.res.num_sms);
}
std::vector<TileCandidate> ChainDP::Solve(std::vector<std::vector<TileCandidate>> const& candidates) const {
  std::vector<TileCandidate> result;
  for (auto const& set : candidates) if (!set.empty()) result.push_back(set.front());
  return result;
}
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
