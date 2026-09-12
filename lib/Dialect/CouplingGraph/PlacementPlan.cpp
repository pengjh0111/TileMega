// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/PlacementPlan.h>

#include <cstring>

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

std::size_t PlacementModeParamCount(PlacementMode mode) {
  for (auto const& row : kModes)
    if (row.mode == mode) return row.params;
  return 0;
}

}  // namespace tilemega::dialect
