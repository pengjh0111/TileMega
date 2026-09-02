// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definition 4, §3.1 solving-authority migration.
//
// The isl/barvinok-backed replacement for ClosedForm's former role as the
// type of a *solved* quantity (wait, fanout, volume, count). ClosedForm
// itself is kept (see ClosedForm.h) as the ordinary symbolic-arithmetic
// value type used to build up extents, tile shapes and access-map
// coefficients -- it never computed a relation or a cardinality, so it was
// never the "parallel solving authority" this migration removes. What
// changes is the type of the *result* DeriveCoupling/ComputeMetrics/
// Contains produce: previously ClosedForm's hand-rolled AST, now a genuine
// isl_pw_qpolynomial (piecewise quasi-polynomial), because wait(x)/fanout(y)
// are position-dependent functions in general (§2 Definition 4), which
// ClosedForm's grammar (constant/symbol/add/multiply/ceildiv/floordiv, no
// case split) cannot represent losslessly.
//
// This type stores only isl text (never a live isl object), matching
// ClosedForm's "no ISL types in the public interface" contract, so it can be
// an MLIR attribute payload without tying attribute lifetime to an isl_ctx.
#pragma once

#include <string>

#include <llvm/ADT/Hashing.h>

#include <tilemega/Analysis/ClosedForm.h>

namespace tilemega::analysis {

class CouplingRelation;  // fwd, for QuasiPolynomial::Card(CouplingRelation)

/// A piecewise quasi-polynomial, printed in isl syntax, e.g.
/// `[S] -> { S : S > 0 }` or, before all producer coordinates are eliminated,
/// a genuine function of a task coordinate, e.g. `[n] -> { [i] -> i : ... }`.
class QuasiPolynomial {
 public:
  QuasiPolynomial();  // the constant 0

  static QuasiPolynomial Constant(long value);
  /// `text` must be valid isl_pw_qpolynomial syntax; canonicalized by an
  /// isl parse/print round trip so equal quantities compare textually equal
  /// (needed for MLIR attribute uniquing).
  static QuasiPolynomial FromIslText(std::string const& text);
  /// card(C): image cardinality per domain point (Definition 4's wait(x) /
  /// fanout(y), depending on which side `relation` is oriented).
  static QuasiPolynomial Card(CouplingRelation const& relation);

  /// Fix every dimension (parameter or set/task-coordinate) named in
  /// `known` to its literal value; dimensions not named in `known` are left
  /// symbolic. Never throws: a partial substitution is always well-formed.
  QuasiPolynomial SubstituteParams(ParamBinding const& known) const;
  /// Fully evaluate: substitutes `known`, then requires the result to have
  /// no remaining parameters or task-coordinate dimensions. Throws
  /// std::out_of_range naming the first dimension still unbound, matching
  /// ClosedForm::Eval's contract.
  long Eval(ParamBinding const& known) const;
  /// True when, after substituting `known` on both sides, `*this` and
  /// `other` are the same function (of whatever task-coordinate dimensions
  /// remain) -- not merely equal at every point `known` happens to bind.
  /// Used by the CG verifier to check a stored metric against the metric
  /// the relation itself implies, without collapsing either to a scalar
  /// first (collapsing is wrong whenever the metric is genuinely
  /// position-dependent, e.g. a triangular access pattern).
  bool SemanticallyEqual(QuasiPolynomial const& other,
                        ParamBinding const& known) const;

  std::string const& ToString() const { return text_; }
  bool IsZero() const;

  friend bool operator==(QuasiPolynomial const& lhs,
                         QuasiPolynomial const& rhs) {
    return lhs.text_ == rhs.text_;
  }
  friend llvm::hash_code hash_value(QuasiPolynomial const& value);

 private:
  explicit QuasiPolynomial(std::string text) : text_(std::move(text)) {}
  std::string text_;
};

}  // namespace tilemega::analysis
