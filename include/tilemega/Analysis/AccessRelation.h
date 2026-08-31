// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2 Definition 2 tensor access relations.
#pragma once
#include <string>
namespace tilemega::analysis {
struct AccessRelation { std::string producer; std::string consumer; std::string presburger_map; };
}  // namespace tilemega::analysis
