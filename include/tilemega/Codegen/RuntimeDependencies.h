// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/RuntimeWindow.h>
#include <cstdint>
#include <limits>

namespace tilemega::codegen {
#if defined(__CUDACC__)
#define TILEMEGA_DEPENDENCY_HD __host__ __device__
#else
#define TILEMEGA_DEPENDENCY_HD
#endif
struct RuntimeDependencyInterval {
  std::uint32_t begin = 0, count = 0;
};
struct RuntimeDependencyTableView {
  RuntimeDependencyInterval const* intervals = nullptr;
  std::uint32_t rows = 0, stride = 0;
};
// Each edge has one fixed row stride. Padding has count zero and contributes
// no dependency; sparse rows remain sparse in every executor and host graph.
template <class Visit>
TILEMEGA_DEPENDENCY_HD bool VisitDependencyTable(
    RuntimeDependencyTableView table, std::uint32_t consumer,
    std::uint32_t producers, Visit const& visit) {
  if (consumer >= table.rows || !table.stride || !table.intervals) return false;
  auto const* row = table.intervals + std::uint64_t(consumer) * table.stride;
  for (std::uint32_t interval = 0; interval < table.stride; ++interval) {
    auto bounds = row[interval];
    if (!bounds.count) continue;
    auto end = std::uint64_t(bounds.begin) + bounds.count;
    if (end > producers || end > std::uint64_t(std::numeric_limits<int>::max())) return false;
    visit(RuntimeWindowBounds{int(bounds.begin), int(end)});
  }
  return true;
}
TILEMEGA_DEPENDENCY_HD inline bool CountedDependencyTarget(
    std::uint32_t count, std::uint64_t iteration, std::uint64_t* target) {
  if (!count || !target || iteration == std::numeric_limits<std::uint64_t>::max() ||
      iteration + 1 > std::numeric_limits<std::uint64_t>::max() / count) return false;
  *target = (iteration + 1) * count;
  return true;
}
#undef TILEMEGA_DEPENDENCY_HD
}  // namespace tilemega::codegen
