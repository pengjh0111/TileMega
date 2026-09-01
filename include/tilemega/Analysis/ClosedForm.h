// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definition 4 and invariant I1.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

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

  ClosedForm operator+(ClosedForm const& rhs) const;
  ClosedForm operator*(ClosedForm const& rhs) const;
  ClosedForm CeilDiv(ClosedForm const& divisor) const;

  long Eval(ParamBinding const& theta, ParamBinding const& g) const;
  std::string ToString() const;
  bool IsConstant() const;

 private:
  struct Node;
  explicit ClosedForm(std::shared_ptr<Node const> node);
  std::shared_ptr<Node const> node_;
};

}  // namespace tilemega::analysis
