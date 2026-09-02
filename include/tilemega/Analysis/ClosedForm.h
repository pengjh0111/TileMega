// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definition 4 and invariant I1.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <llvm/ADT/Hashing.h>

namespace tilemega::analysis {

/// Concrete values for one parameter namespace: theta (shape) or g (tile).
struct ParamBinding {
  std::unordered_map<std::string, long> values;

  ParamBinding& Bind(std::string name, long value);
  bool Contains(std::string const& name) const;
  long At(std::string const& name) const;
};

/// Minimal immutable closed-form expression used before barvinok integration.
/// The public interface has no ISL/barvinok types, so Phase 3 can replace its
/// implementation without changing Solver or Codegen clients.
class ClosedForm {
 public:
  ClosedForm();

  static ClosedForm Constant(long value);
  static ClosedForm Symbol(std::string const& name);
  /// Parse the stable textual form emitted by ToString.  The accepted grammar
  /// is integer/symbol, +, *, parentheses, ceildiv(a,b), and a//b.
  static ClosedForm Parse(std::string const& expression);

  ClosedForm operator+(ClosedForm const& rhs) const;
  ClosedForm operator*(ClosedForm const& rhs) const;
  ClosedForm CeilDiv(ClosedForm const& divisor) const;
  ClosedForm FloorDiv(ClosedForm const& divisor) const;

  long Eval(ParamBinding const& theta, ParamBinding const& g) const;
  std::string ToString() const;
  bool IsConstant() const;
  /// True when the expression is the literal constant `value`.  Used by the
  /// derivation to recognise the exact cases; it never guesses.
  bool IsLiteral(long value) const;
  /// Structural exact division.  Returns true and writes the quotient only
  /// when `divisor` cancels symbolically (matching multiplicative factors, or
  /// an integer constant that divides evenly, distributed over sums).  A false
  /// result means "not established", never "not divisible": callers must fall
  /// back to a conservative relaxation rather than inventing a quotient.
  bool TryExactDivide(ClosedForm const& divisor, ClosedForm* quotient) const;
  /// Multiplicative factors in canonical order, used by TryExactDivide and by
  /// diagnostics.  A non-product expression yields a single factor.
  std::vector<std::string> FactorStrings() const;

  /// Replace every symbol named in `known` with its literal value; symbols
  /// not in `known` are left symbolic. Used before building an isl object:
  /// isl's floor/ceildiv require a literal divisor (confirmed empirically,
  /// see docs/experiments/P3_ISL/result.md), so a tile size ("g") must be
  /// substituted before ToIslText, while a workload dimension ("theta") is
  /// deliberately left out of `known` so it survives as a genuine isl
  /// parameter (invariant I1).
  ClosedForm Substitute(ParamBinding const& known) const;
  /// Every distinct symbol name appearing in this expression, in no
  /// particular order. Used to collect the isl parameter list (every symbol
  /// not resolved by Substitute's `known` binding must be declared as an
  /// isl parameter before ToIslText's output can be parsed).
  std::vector<std::string> FreeSymbols() const;
  /// Render in isl syntax (ceild/floord instead of ceildiv/floordiv).
  /// Throws std::domain_error if a ceildiv/floordiv divisor is not constant
  /// after Substitute -- isl cannot represent a parametric divisor, so this
  /// is the point where an incomplete `known` binding is caught explicitly
  /// rather than producing invalid isl text.
  std::string ToIslText() const;
  friend bool operator==(ClosedForm const& lhs, ClosedForm const& rhs) {
    return lhs.ToString() == rhs.ToString();
  }
  friend llvm::hash_code hash_value(ClosedForm const& value) {
    return llvm::hash_value(value.ToString());
  }

 private:
  struct Node;
  explicit ClosedForm(std::shared_ptr<Node const> node);
  std::shared_ptr<Node const> node_;
};

}  // namespace tilemega::analysis
