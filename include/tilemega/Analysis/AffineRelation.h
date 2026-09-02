// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 1-3 and invariant I1.
//
// `AffineRelation` (the relation class) and its `ProducerMap`/`AffineRange`
// helpers are deleted: they were the hand-rolled, ClosedForm-AST-based
// representation of the coupling relation C, and the whole point of the
// isl/barvinok migration (see docs/DEPENDENCIES.md,
// docs/experiments/P3_ISL/) is that CouplingRelation (an isl_map, see
// CouplingRelation.h) is now the sole representation of C -- keeping both
// would recreate the "two semantic authorities for the same relation"
// problem the migration exists to remove.
//
// `AffineExpr` survives: it is not a relation, just a symbolic affine index
// expression (coefficient * task-coordinate + offset, with an optional
// overall divisor), used by AccessRelation to describe a read/write's base
// index before that description is turned into isl map constraints. It
// never computed a relation, a cardinality, or a containment check, so it
// was never the duplicated authority; it is the same kind of "build up an
// expression" value as ClosedForm; ToIslText renders it in isl syntax the
// same way ClosedForm::ToIslText does, for the same reason (a symbolic
// divisor is not representable in isl -- confirmed empirically, see
// docs/experiments/P3_ISL/result.md).
#pragma once

#include <string>
#include <vector>

#include <tilemega/Analysis/ClosedForm.h>

namespace tilemega::analysis {

struct TaskSpaceId {
  std::string name;
  friend bool operator==(TaskSpaceId const& a, TaskSpaceId const& b) {
    return a.name == b.name;
  }
};

/// sum_i(coeff_i(theta,g) * floordiv(coordinate_i, group_i)) + offset(theta,g),
/// then floordiv by an overall divisor if not 1.
struct AffineExpr {
  /// coefficient * floordiv(coordinate, group).  `group` defaults to 1, which
  /// is the plain affine term.  A group greater than one is what a GQA head
  /// map needs (kv_head = floor(query_head / (n_h / n_kv))); it is kept as a
  /// first-class term because folding it into a coefficient would be wrong.
  struct Term {
    std::string coordinate;
    ClosedForm coefficient = ClosedForm::Constant(1);
    ClosedForm group = ClosedForm::Constant(1);
  };

  std::vector<Term> terms;
  ClosedForm offset = ClosedForm::Constant(0);
  /// Floor divisor applied to the whole sum.  A value of 1 (the default) is a
  /// plain affine expression and is not printed.  Anything else denotes
  /// floordiv(sum, divisor), which §2.7 edge 5 (`(s,hh,j) -> (floor(s/Tm), hh)`)
  /// requires; it is exact only because the projected extent is one element.
  ClosedForm divisor = ClosedForm::Constant(1);

  static AffineExpr Constant(ClosedForm value);
  static AffineExpr Variable(std::string coordinate,
                             ClosedForm coefficient = ClosedForm::Constant(1),
                             ClosedForm group = ClosedForm::Constant(1));
  AffineExpr operator+(AffineExpr const& rhs) const;
  AffineExpr operator*(ClosedForm const& scale) const;
  /// Divide every coefficient and the offset exactly.  Returns false when any
  /// part does not cancel; the caller must then relax rather than guess.
  bool TryExactDivide(ClosedForm const& d, AffineExpr* quotient) const;
  /// Coordinates that occur in this expression.
  std::vector<std::string> Coordinates() const;
  /// Every distinct ClosedForm symbol occurring in coefficient/group/offset/
  /// divisor (never a task coordinate -- those are listed by Coordinates).
  std::vector<std::string> FreeSymbols() const;
  bool IsZero() const;
  std::string ToString() const;
  /// Render in isl syntax (see ClosedForm::ToIslText); `known` substitutes
  /// tile-size ("g") symbols to literals first, the same rule ClosedForm
  /// follows, since `group`/`coefficient`/`offset`/`divisor` are all
  /// ClosedForm values that may need the same substitution before an isl
  /// floor/ceildiv divisor is legal.
  std::string ToIslText(ParamBinding const& known) const;
};

}  // namespace tilemega::analysis
