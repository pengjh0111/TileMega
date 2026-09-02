// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2 Definition 4 closed-form metrics.
#pragma once
#include <tilemega/Analysis/QuasiPolynomial.h>
namespace tilemega::analysis {
/// wait/fanout are functions of the consumer/producer task coordinate
/// respectively (Definition 4); volume/count are parameter-only (they do not
/// vary across the task space for any access pattern this codebase derives).
/// All four are isl_pw_qpolynomial-backed (see QuasiPolynomial.h) rather
/// than ClosedForm: they are *solved* quantities (barvinok counts over the
/// derived relation C), not built-up expressions.
struct DerivedMetrics {
  QuasiPolynomial wait;
  QuasiPolynomial fanout;
  QuasiPolynomial volume;
  QuasiPolynomial count;
};
}  // namespace tilemega::analysis
