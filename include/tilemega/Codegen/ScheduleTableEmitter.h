// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3 static schedule table emission (Phase 2 stub).
#pragma once
#include <cstdint>
#include <vector>
namespace tilemega::codegen {
struct ScheduleEntry { std::uint32_t task_kind = 0; std::uint32_t logical_tile = 0; };
class ScheduleTableEmitter { public: std::vector<ScheduleEntry> Emit(std::vector<int> const& order) const; };
}  // namespace tilemega::codegen
