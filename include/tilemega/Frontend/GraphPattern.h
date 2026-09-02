// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.1 frontend grouping.
//
// A declarative matcher over the exported FX graph. A pattern is data: a list
// of slots, each carrying an operator role and constraints on its operands
// expressed in terms of earlier slots, plus shape constraints between slots.
// Matching yields the slot -> node binding, which the frontend uses to group
// operators into one fused task space.
//
// Two properties this file exists to provide:
//   * The frontend compares `target` strings in exactly one place, the role
//     table, so a graph normalized by `run_decompositions()` and the same
//     graph before it match the same patterns.
//   * Grouping is stated as use-def structure rather than as parameter names,
//     so a model whose modules are not called `layers.N.q_proj` still groups.
#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tilemega/Frontend/ModelPlan.h>

namespace tilemega::frontend {

/// One semantic role and the ATen targets that realize it. A role is a set
/// because the same operation appears under different targets depending on
/// which normalization ran: `aten.linear.default` before
/// `run_decompositions()`, `aten.mm`/`aten.addmm`/`aten.bmm` after.
struct OperatorRole {
  std::string role;
  std::vector<std::string> targets;
};

/// The whole ATen vocabulary the frontend knows. Anything absent from this
/// table is an unrecognized operator and must degrade, not fail.
std::vector<OperatorRole> const& CoreRoles();

/// A conjunction of clauses on one operand position.
struct OperandConstraint {
  enum class Kind {
    kValue,      ///< equals the node bound to `slot`, modulo layout-only ops
    kDepends,    ///< there is a use-def path from the node bound to `slot`
    kNotDepends,
    kParameter,  ///< a weight/buffer placeholder
    kInput,      ///< a model input placeholder (a state object, e.g. a KV cache)
    /// The nearest node of `role` behind the operand is the one bound to
    /// `slot`: reachable without crossing another node of that role. Plain
    /// dependence is transitive and therefore blind to layer boundaries --
    /// every projection of layer L is a predecessor of every attention of
    /// layer L+1 -- so a stacked model needs this stronger form.
    kNear,
  };
  struct Clause {
    Kind kind = Kind::kParameter;
    std::string slot;
    std::string role;
  };
  std::vector<Clause> clauses;  ///< empty is unconstrained
};

inline OperandConstraint Any() { return {}; }
inline OperandConstraint Value(std::string slot) {
  return {{{OperandConstraint::Kind::kValue, std::move(slot)}}};
}
inline OperandConstraint Dep(std::string slot) {
  return {{{OperandConstraint::Kind::kDepends, std::move(slot)}}};
}
inline OperandConstraint NotDep(std::string slot) {
  return {{{OperandConstraint::Kind::kNotDepends, std::move(slot)}}};
}
inline OperandConstraint Param() {
  return {{{OperandConstraint::Kind::kParameter, {}}}};
}
inline OperandConstraint Near(std::string role, std::string slot) {
  return {{{OperandConstraint::Kind::kNear, std::move(slot), std::move(role)}}};
}
inline OperandConstraint Input() {
  return {{{OperandConstraint::Kind::kInput, {}}}};
}
inline OperandConstraint All(OperandConstraint a, OperandConstraint b) {
  for (auto& clause : b.clauses) a.clauses.push_back(std::move(clause));
  return a;
}

struct PatternNode {
  std::string slot;
  std::string role;  ///< empty matches any call_function
  std::vector<OperandConstraint> operands;
  /// Match the constraints against the operands as a set rather than by
  /// position. Needed wherever the exporter's argument order is not fixed,
  /// e.g. a residual add.
  bool unordered = false;
  /// Node-level use-def constraints: this node must (not) be an ancestor of
  /// the node already bound to the named slot. Operand constraints alone
  /// cannot say "the projection that feeds this attention", because the path
  /// runs through an arbitrary number of reshapes.
  std::vector<std::string> ancestor_of;
  std::vector<std::string> not_ancestor_of;
};

/// `a.shape[axis_a] == b.shape[axis_b]`, checked on static extents only; a
/// symbolic extent makes the constraint vacuous rather than false, because a
/// dynamic axis is not evidence against the pattern. A negative axis counts
/// from the end.
struct AxisConstraint {
  std::string a;
  int axis_a = -1;
  std::string b;
  int axis_b = -1;
};

struct GraphPattern {
  std::string name;
  std::vector<PatternNode> nodes;
  std::vector<AxisConstraint> shape;
};

using PatternBinding = std::map<std::string, FxNodeRecord const*>;

class PatternMatcher {
 public:
  PatternMatcher(std::vector<FxNodeRecord> const& nodes,
                 std::vector<SignatureInput> const& inputs);

  /// The producer of `name` with every layout-only operator skipped. A view,
  /// permute or metadata assertion changes indexing, not the value, and
  /// normalization inserts many of them; a matcher that did not skip them
  /// would recognize the pre-decomposition graph only.
  std::string Value(std::string const& name) const;

  bool DependsOn(std::string const& value, std::string const& ancestor) const;
  bool IsParameter(std::string const& name) const;
  bool IsInput(std::string const& name) const;
  FxNodeRecord const* Find(std::string const& name) const;
  /// The role of `node`, or "" when no role covers its target.
  std::string const& RoleOf(FxNodeRecord const& node) const;

  /// Every binding of `pattern`, in the FX order of the first slot. `seed`
  /// pre-binds slots, which is how a pattern is anchored on an already
  /// matched group.
  std::vector<PatternBinding> FindAll(GraphPattern const& pattern,
                                      PatternBinding seed = {}) const;
  /// The nearest parameter placeholder reachable from `name` by use-def.
  std::string NearestParameter(std::string const& name) const;
  /// The first node of `role` behind `name`, not crossing a node of
  /// `blocker`. Empty when there is none.
  std::string FirstOfRole(std::string const& name, std::string const& role,
                          std::string const& blocker) const;
  /// The unique binding; throws when there is none or more than one.
  PatternBinding MatchOne(GraphPattern const& pattern) const;

 private:
  bool Admissible(GraphPattern const& pattern, std::size_t index,
                  FxNodeRecord const& node, PatternBinding const& bound) const;
  bool OperandsMatch(PatternNode const& slot, FxNodeRecord const& node,
                     PatternBinding const& bound) const;
  bool OneOperandMatches(OperandConstraint const& constraint,
                         std::string const& operand,
                         PatternBinding const& bound) const;

  std::vector<FxNodeRecord> const& nodes_;
  std::unordered_map<std::string, FxNodeRecord const*> by_name_;
  std::unordered_map<std::string, std::string> role_of_;
  std::unordered_set<std::string> parameters_;
  std::unordered_set<std::string> user_inputs_;
};

}  // namespace tilemega::frontend
