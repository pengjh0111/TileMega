// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definition 1 and §5.1 symbolic guards (Phase 1 stub).
#pragma once
#include <tilemega/Analysis/ClosedForm.h>

#include <string>
#include <unordered_map>
#include <vector>
namespace tilemega::frontend {
struct ParameterRange { long minimum = 0; long maximum = 0; };
struct ShapeConstraint {
  enum class Predicate { kEqual, kLessEqual, kDivisible };
  analysis::ClosedForm lhs;
  analysis::ClosedForm rhs;
  Predicate predicate = Predicate::kEqual;
  std::string source;
  bool redundant_after_normalization = false;
};
struct SymbolicShape {
  std::vector<std::string> dimensions;
  std::unordered_map<std::string, ParameterRange> ranges;
  std::vector<ShapeConstraint> constraints;
  std::unordered_map<std::string, std::string> canonical_symbol;
};
class SymbolicShapeBridge {
 public:
  SymbolicShape Parse(
      std::unordered_map<std::string, std::string> const& ranges,
      std::vector<std::string> const& guards,
      std::vector<std::vector<std::string>> const& user_input_shapes) const;
};
}  // namespace tilemega::frontend
