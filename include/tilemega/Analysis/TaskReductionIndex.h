// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/Semantics.h>

namespace tilemega::analysis {
// g-dependent enumeration of issued reduction iterations. The semantic
// domain still names its original reduction axes and physical accesses.
struct TaskReductionIndex {
  IndexResult index;
  ClosedForm capacity = ClosedForm::Constant(0);
  ClosedForm issued_width = ClosedForm::Constant(0);
};
} // namespace tilemega::analysis
