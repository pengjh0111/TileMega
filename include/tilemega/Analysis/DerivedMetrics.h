// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2 Definition 4 fan-in/locality/reuse metrics.
#pragma once
#include <cstddef>
namespace tilemega::analysis {
struct DerivedMetrics { std::size_t fan_in = 0; std::size_t fan_out = 0; double locality = 0; double reuse = 0; };
}  // namespace tilemega::analysis
