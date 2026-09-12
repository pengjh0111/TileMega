// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/PlacementPlan.h>

#include <cstring>
#include <vector>

namespace tilemega::dialect {
namespace {
struct ModeRow {
  PlacementMode mode;
  char const* name;
  std::size_t params;
};
// One table, read by the verifier, by codegen and by the host materializer.
constexpr ModeRow kModes[] = {
    {PlacementMode::kLegacyGridStride, "legacy_grid_stride", 0},
    {PlacementMode::kRotate, "rotate", 0},
    {PlacementMode::kBalanced, "balanced", 0},
    {PlacementMode::kEft, "eft", 0},
    {PlacementMode::kTemplate, "template", 1},
};
}  // namespace

char const* PlacementModeName(PlacementMode mode) {
  for (auto const& row : kModes)
    if (row.mode == mode) return row.name;
  return nullptr;
}

bool ParsePlacementMode(char const* name, std::size_t length, PlacementMode* mode) {
  for (auto const& row : kModes) {
    if (std::strlen(row.name) != length) continue;
    if (std::memcmp(row.name, name, length) != 0) continue;
    if (mode) *mode = row.mode;
    return true;
  }
  return false;
}

bool ValidatePlacementTable(PlacementTable const& table, std::string* error) {
  auto fail = [&](std::string message) {
    if (error) *error = std::move(message);
    return false;
  };
  if (table.worker.empty())
    return fail("the placement table is empty; absent and empty are different "
                "states and only absent means legacy_grid_stride");
  if (table.worker.size() != table.slot.size())
    return fail("the placement table carries " + std::to_string(table.worker.size()) +
                " pi entries and " + std::to_string(table.slot.size()) + " sigma entries");
  if (table.grid <= 0)
    return fail("the placement table names grid " + std::to_string(table.grid));
  if (table.seq <= 0 || table.past < 0)
    return fail("the placement table names seq " + std::to_string(table.seq) +
                " and past " + std::to_string(table.past));
  // sigma is counted per worker rather than sorted: a dense [0, n) is exactly
  // "n slots and none of them out of range", which the counts already say.
  std::vector<long long> queue(static_cast<std::size_t>(table.grid), 0);
  for (std::size_t node = 0; node < table.worker.size(); ++node) {
    int const worker = table.worker[node];
    if (worker < 0 || worker >= table.grid)
      return fail("the placement table sends node " + std::to_string(node) +
                  " to worker " + std::to_string(worker) + ", outside its grid of " +
                  std::to_string(table.grid));
    ++queue[static_cast<std::size_t>(worker)];
  }
  std::vector<std::vector<bool>> seen(static_cast<std::size_t>(table.grid));
  for (std::size_t worker = 0; worker < seen.size(); ++worker)
    seen[worker].assign(static_cast<std::size_t>(queue[worker]), false);
  for (std::size_t node = 0; node < table.slot.size(); ++node) {
    auto& worker_seen = seen[static_cast<std::size_t>(table.worker[node])];
    int const slot = table.slot[node];
    if (slot < 0 || static_cast<std::size_t>(slot) >= worker_seen.size())
      return fail("sigma on worker " + std::to_string(table.worker[node]) +
                  " puts node " + std::to_string(node) + " at slot " +
                  std::to_string(slot) + " of a queue of " +
                  std::to_string(worker_seen.size()));
    if (worker_seen[static_cast<std::size_t>(slot)])
      return fail("sigma on worker " + std::to_string(table.worker[node]) +
                  " uses slot " + std::to_string(slot) + " twice");
    worker_seen[static_cast<std::size_t>(slot)] = true;
  }
  return true;
}

std::size_t PlacementModeParamCount(PlacementMode mode) {
  for (auto const& row : kModes)
    if (row.mode == mode) return row.params;
  return 0;
}

}  // namespace tilemega::dialect
