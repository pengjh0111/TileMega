// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §4.3 resource-aware list scheduling (Phase 3 stub).
#pragma once
#include <vector>
namespace tilemega::solver {
class ListScheduler { public: std::vector<int> Schedule(std::vector<std::vector<int>> const&) const; };
}  // namespace tilemega::solver
