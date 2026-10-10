// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <tilemega/Analysis/Semantics.h>

namespace tilemega::analysis {
// g-dependent enumeration of issued reduction iterations. The semantic
// domain still names its original reduction axes and physical accesses.
struct TaskReductionIndex {
  IndexResult index;
  ClosedForm capacity = ClosedForm::Constant(0);
  ClosedForm issued_width = ClosedForm::Constant(0);
  // Nonzero selects floor(j*capacity/chunks) issued-iteration boundaries.
  // Zero retains the original uniform reduction_chunk contract.
  std::uint32_t chunks = 0;
};
} // namespace tilemega::analysis
