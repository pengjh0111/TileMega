// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2 Definition 4 closed-form metrics.
#pragma once
#include <tilemega/Analysis/ClosedForm.h>
namespace tilemega::analysis {
struct DerivedMetrics {
  ClosedForm wait = ClosedForm::Constant(0);
  ClosedForm fanout = ClosedForm::Constant(0);
  ClosedForm volume = ClosedForm::Constant(0);
  ClosedForm count = ClosedForm::Constant(0);
};
}  // namespace tilemega::analysis
