// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 4–5 (Phase 1 stub).
#pragma once
#include <string>
namespace tilemega::dialect {
struct ScheduleAttr { std::string mapping; std::string local_order; };
struct TierAttr { std::string value; };
}  // namespace tilemega::dialect
