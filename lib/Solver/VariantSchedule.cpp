// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/VariantSchedule.h>
#include <tilemega/Solver/ListScheduler.h>

#include <algorithm>
#include <stdexcept>

namespace tilemega::solver {

VariantStageSchedule BuildVariantStageSchedule(
    std::vector<codegen::DependencyRecord> const& dependencies,
    std::size_t stage_count) {
  std::vector<std::vector<int>> successors(stage_count);
  for (auto const& edge : dependencies) {
    if (edge.producer >= stage_count || edge.consumer >= stage_count)
      throw std::invalid_argument("dependency names a stage outside the model");
    successors[edge.producer].push_back(static_cast<int>(edge.consumer));
  }
  ListScheduler scheduler;
  std::vector<int> const order = scheduler.Schedule(successors);
  ScheduleSafety const safety = scheduler.Validate(successors, order);
  VariantStageSchedule result;
  result.max_dependency_span =
      static_cast<std::uint32_t>(safety.max_dependency_span);
  result.schedule.reserve(order.size());
  for (int stage : order) {
    auto const first = std::lower_bound(
        dependencies.begin(), dependencies.end(), stage,
        [](codegen::DependencyRecord const& edge, int consumer) {
          return edge.consumer < static_cast<std::uint32_t>(consumer);
        });
    auto const last = std::upper_bound(
        first, dependencies.end(), stage,
        [](int consumer, codegen::DependencyRecord const& edge) {
          return static_cast<std::uint32_t>(consumer) < edge.consumer;
        });
    result.schedule.push_back(
        {static_cast<std::uint32_t>(stage),
         static_cast<std::uint32_t>(first - dependencies.begin()),
         static_cast<std::uint32_t>(last - first)});
  }
  return result;
}

}  // namespace tilemega::solver
