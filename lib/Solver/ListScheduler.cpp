// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ListScheduler.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace tilemega::solver {
namespace {

std::vector<int> InDegrees(std::vector<std::vector<int>> const& successors) {
  std::vector<int> degree(successors.size(), 0);
  for (auto const& list : successors)
    for (int next : list) {
      if (next < 0 || next >= static_cast<int>(successors.size()))
        throw std::invalid_argument("list scheduler: successor out of range");
      ++degree[next];
    }
  return degree;
}

}  // namespace

std::vector<int> ListScheduler::Levels(
    std::vector<std::vector<int>> const& successors) const {
  std::vector<int> degree = InDegrees(successors);
  std::vector<int> level(successors.size(), 0);
  std::vector<int> ready;
  for (std::size_t i = 0; i < degree.size(); ++i)
    if (degree[i] == 0) ready.push_back(static_cast<int>(i));
  std::size_t settled = 0;
  while (!ready.empty()) {
    int const node = ready.back();
    ready.pop_back();
    ++settled;
    for (int next : successors[node]) {
      level[next] = std::max(level[next], level[node] + 1);
      if (--degree[next] == 0) ready.push_back(next);
    }
  }
  if (settled != successors.size())
    throw std::invalid_argument("list scheduler: stage graph has a cycle");
  return level;
}

std::vector<int> ListScheduler::Heights(
    std::vector<std::vector<int>> const& successors) const {
  // Reverse topological order, obtained by ranking on the forward levels: a
  // node's level is strictly greater than every predecessor's, so descending
  // level visits every successor before the node itself.
  std::vector<int> const level = Levels(successors);
  std::vector<int> order(successors.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return level[a] > level[b]; });
  std::vector<int> height(successors.size(), 0);
  for (int node : order)
    for (int next : successors[node])
      height[node] = std::max(height[node], height[next] + 1);
  return height;
}

std::vector<int> ListScheduler::Schedule(
    std::vector<std::vector<int>> const& successors,
    ScheduleStats* stats) const {
  std::vector<int> const level = Levels(successors);
  std::vector<int> const height = Heights(successors);
  std::vector<int> order(successors.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (level[a] != level[b]) return level[a] < level[b];
    if (height[a] != height[b]) return height[a] > height[b];
    return a < b;
  });
  if (stats) {
    stats->nodes = static_cast<int>(successors.size());
    stats->levels = successors.empty() ? 0 : *std::max_element(level.begin(),
                                                               level.end()) + 1;
    std::vector<int> width(stats->levels, 0);
    for (int value : level) ++width[value];
    stats->widest_level =
        width.empty() ? 0 : *std::max_element(width.begin(), width.end());
    stats->barriers_saved = stats->nodes - stats->levels;
  }
  return order;
}

}  // namespace tilemega::solver
