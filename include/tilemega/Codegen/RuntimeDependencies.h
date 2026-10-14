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
  std::uint32_t row_period = 0;
};
TILEMEGA_DEPENDENCY_HD constexpr bool ValidDependencyRowPeriod(
    std::uint32_t rows, std::uint32_t period) {
  return !period || (period<=rows && rows%period==0);
}
// Each edge has one fixed row stride. Padding has count zero and contributes
// no dependency. An optional exact period stores repeated rows only once;
// logical consumer IDs and sparse dependencies are unchanged.
template <class Visit>
TILEMEGA_DEPENDENCY_HD bool VisitDependencyTable(
    RuntimeDependencyTableView table, std::uint32_t consumer,
    std::uint32_t producers, Visit const& visit) {
  if (consumer >= table.rows) return false;
  if (!ValidDependencyRowPeriod(table.rows,table.row_period)) return false;
  if (!table.stride) return true;
  if (!table.intervals) return false;
  auto stored_row=table.row_period ? consumer%table.row_period : consumer;
  auto const* row = table.intervals + std::uint64_t(stored_row) * table.stride;
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
