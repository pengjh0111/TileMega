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
#include <array>

#include <llvm/ADT/Hashing.h>

#include <tilemega/Analysis/ClosedForm.h>

namespace tilemega::analysis {

class CouplingRelation;  // fwd, for QuasiPolynomial::Card(CouplingRelation)

/// A piecewise quasi-polynomial, printed in isl syntax, e.g.
/// `[S] -> { S : S > 0 }` or, before all producer coordinates are eliminated,
/// a genuine function of a task coordinate, e.g. `[n] -> { [i] -> i : ... }`.
class QuasiPolynomial {
 public:
  struct PolynomialPiece {
    std::string domain;
    std::array<std::string,3> coefficients;  ///< exact rational, ascending degree
  };
  std::vector<PolynomialPiece> QuadraticPieces(std::string const& parameter) const;
  struct PolynomialInterval {
    long begin, end;
    std::array<std::string,3> coefficients;
  };
  // Split bounded floor domains exactly. Non-interval parameter sets and
  // degree > 2 remain explicit errors, never fitted polynomial samples.
  std::vector<PolynomialInterval> QuadraticIntervals(std::string const& parameter,
      long begin,long end) const;
  QuasiPolynomial();  // the constant 0

  static QuasiPolynomial Constant(long value);
  /// `text` must be valid isl_pw_qpolynomial syntax; canonicalized by an
  /// isl parse/print round trip so equal quantities compare textually equal
  /// (needed for MLIR attribute uniquing).
  static QuasiPolynomial FromIslText(std::string const& text);
  /// card(C): image cardinality per domain point (Definition 4's wait(x) /
  /// fanout(y), depending on which side `relation` is oriented).
  static QuasiPolynomial Card(CouplingRelation const& relation);

  /// Fix every parameter named in `known` to its literal value; parameters not named in `known` are left
  /// symbolic. Never throws: a partial substitution is always well-formed.
  QuasiPolynomial SubstituteParams(ParamBinding const& known) const;
  /// Restrict every task coordinate to one literal point. Missing coordinate
  /// bindings are errors; theta/g parameters are still bound separately.
  QuasiPolynomial BindCoordinates(ParamBinding const& point) const;
  /// Fully evaluate: substitutes `known`, then requires the result to have
  /// no remaining parameters or task-coordinate dimensions. Throws
  /// std::out_of_range naming the first dimension still unbound, matching
  /// ClosedForm::Eval's contract.
  long Eval(ParamBinding const& known) const;
  std::vector<long> EvalPoints(ParamBinding const& known,
      std::vector<ParamBinding> const& coordinates) const;
  /// Sum over task-coordinate dimensions, retaining symbolic parameters.
  QuasiPolynomial SumDomain() const;
  /// Add exact functions, treating points outside either domain as zero.
  QuasiPolynomial Add(QuasiPolynomial const& other) const;
  QuasiPolynomial Scale(long factor) const;
  QuasiPolynomial ScaleRational(std::string const& factor) const;
  QuasiPolynomial Multiply(QuasiPolynomial const& other) const;
  // One on the represented support and zero elsewhere. Used for per-task
  // fixed costs without charging nonexistent tasks in a symbolic wave.
  QuasiPolynomial SupportIndicator() const;
  /// Exact ISL domain splitting where a floor attains at most this many
  /// values. This changes representation, not parameter sampling or values.
  QuasiPolynomial SplitPeriods(int max_periods) const;
  static QuasiPolynomial Sum(std::vector<QuasiPolynomial> const& terms);
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
