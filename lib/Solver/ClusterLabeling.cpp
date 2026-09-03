// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ClusterLabeling.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace tilemega::solver {
namespace {

/// A cluster under construction.  `first`/`last` are the stage span, which is
/// what the temporal constraint is checked against: two clusters may merge
/// only if the union still fits inside `max_stage_distance`, so the reach is a
/// property of the whole group and not of the pair that happened to propose
/// the merge.
struct Group {
  std::vector<int> members;
  int first = 0;
  int last = 0;
  double smem = 0;
};

}  // namespace

ClusterPlan ClusterLabeling::Assign(std::vector<ClusterNode> const& nodes,
                                    std::vector<ClusterEdge> const& edges,
                                    ClusterConstraints const& limits) const {
  ClusterPlan plan;
  int const n = static_cast<int>(nodes.size());
  plan.labels.assign(n, 0);
  std::iota(plan.labels.begin(), plan.labels.end(), 0);
  plan.clusters = n;
  for (auto const& edge : edges) {
    if (edge.a < 0 || edge.a >= n || edge.b < 0 || edge.b >= n)
      throw std::invalid_argument("cluster labeling: edge names no node");
    plan.total_weight += edge.weight;
  }
  if (n == 0) return plan;

  int const cap = std::max(1, limits.max_cluster_size);
  if (cap == 1) {
    plan.limited_by = "target has no clusters";
    return plan;
  }

  std::vector<Group> groups(n);
  for (int i = 0; i < n; ++i)
    groups[i] = {{i}, nodes[i].stage, nodes[i].stage, nodes[i].smem_bytes};

  // Pair weights are accumulated once; agglomeration then works on group
  // pairs, so a merged group inherits the sum of its members' edges to every
  // other group -- the standard coarsening contraction.
  std::vector<std::vector<double>> weight(n, std::vector<double>(n, 0.0));
  for (auto const& edge : edges) {
    if (edge.a == edge.b) continue;
    weight[edge.a][edge.b] += edge.weight;
    weight[edge.b][edge.a] += edge.weight;
  }

  std::vector<int> live(n);
  std::iota(live.begin(), live.end(), 0);
  bool smem_blocked = false, stage_blocked = false;
  while (true) {
    double best = 0;
    int best_a = -1, best_b = -1;
    for (std::size_t i = 0; i < live.size(); ++i)
      for (std::size_t j = i + 1; j < live.size(); ++j) {
        int const a = live[i], b = live[j];
        if (weight[a][b] <= 0) continue;
        if (static_cast<int>(groups[a].members.size() +
                             groups[b].members.size()) > cap)
          continue;
        int const span = std::max(groups[a].last, groups[b].last) -
                         std::min(groups[a].first, groups[b].first);
        if (span > limits.max_stage_distance) { stage_blocked = true; continue; }
        if (limits.smem_budget_bytes > 0 &&
            groups[a].smem + groups[b].smem > limits.smem_budget_bytes) {
          smem_blocked = true;
          continue;
        }
        // Ties break on the lowest (a, b) so the plan is reproducible.
        if (weight[a][b] > best) { best = weight[a][b]; best_a = a; best_b = b; }
      }
    if (best_a < 0) break;
    plan.internal_weight += weight[best_a][best_b];
    groups[best_a].members.insert(groups[best_a].members.end(),
                                  groups[best_b].members.begin(),
                                  groups[best_b].members.end());
    groups[best_a].first = std::min(groups[best_a].first, groups[best_b].first);
    groups[best_a].last = std::max(groups[best_a].last, groups[best_b].last);
    groups[best_a].smem += groups[best_b].smem;
    for (int other : live) {
      if (other == best_a || other == best_b) continue;
      weight[best_a][other] += weight[best_b][other];
      weight[other][best_a] = weight[best_a][other];
      weight[best_b][other] = weight[other][best_b] = 0;
    }
    weight[best_a][best_b] = weight[best_b][best_a] = 0;
    live.erase(std::find(live.begin(), live.end(), best_b));
  }

  std::sort(live.begin(), live.end());
  plan.clusters = static_cast<int>(live.size());
  plan.largest = 1;
  for (std::size_t id = 0; id < live.size(); ++id) {
    auto const& group = groups[live[id]];
    plan.largest = std::max(plan.largest, static_cast<int>(group.members.size()));
    for (int member : group.members) plan.labels[member] = static_cast<int>(id);
  }
  if (plan.largest < cap) {
    if (smem_blocked) plan.limited_by = "shared memory budget";
    else if (stage_blocked) plan.limited_by = "temporal reach";
    else plan.limited_by = "coupling graph has no heavier admissible pair";
  }
  return plan;
}

std::vector<int> ClusterLabeling::Assign(int count, int max_cluster_size) const {
  std::vector<int> labels(std::max(0, count));
  int width = std::max(1, max_cluster_size);
  for (int i = 0; i < count; ++i) labels[i] = i / width;
  return labels;
}

}  // namespace tilemega::solver
