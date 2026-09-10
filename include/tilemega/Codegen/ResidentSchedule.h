// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

namespace tilemega::codegen {
// No over-resident proof is attached to the current L-sched mapping.
// This contract is checked against actual occupancy before queue allocation.
inline bool ResidentScheduleLegal(bool resident_only, int grid, int resident_limit) {
  return resident_only && grid>0 && resident_limit>0 && grid<=resident_limit;
}
}  // namespace tilemega::codegen
