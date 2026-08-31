// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 1–3 (Phase 1 stub).
#pragma once
#include <string>
namespace tilemega::dialect {
struct TaskOp { std::string name; std::string iteration_domain; };
struct CouplingOp { std::string producer; std::string consumer; std::string relation; };
}  // namespace tilemega::dialect
