// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/GraphPattern.h>

#include <algorithm>
#include <cctype>
#include <numeric>
#include <stdexcept>

namespace tilemega::frontend {
namespace {

/// Operators that re-index a value without changing it. Skipping them is what
/// makes one pattern match both the composite export and its Core ATen form.
std::unordered_set<std::string> const& LayoutOnly() {
  static std::unordered_set<std::string> const set = {
      "aten.view.default",       "aten.permute.default",
      "aten.transpose.int",      "aten.contiguous.default",
      "aten.clone.default",      "aten.expand.default",
      "aten.to.dtype",           "aten._assert_tensor_metadata.default",
      "aten.unsqueeze.default",  "aten.squeeze.dim",
      "aten.alias.default",      "aten._to_copy.default",
  };
  return set;
}

bool StaticExtent(std::string const& value, long& out) {
  if (value.empty() ||
      !std::all_of(value.begin(), value.end(),
                   [](unsigned char c) { return std::isdigit(c) != 0; }))
    return false;
  out = std::stol(value);
  return true;
}

}  // namespace

std::vector<OperatorRole> const& CoreRoles() {
  // One entry per semantic role, listing every ATen target that realizes it.
  // The composite and the Core ATen spelling sit in the same entry, which is
  // why `run_decompositions()` does not change what the frontend recognizes.
  static std::vector<OperatorRole> const roles = {
      {"contraction",
       {"aten.linear.default", "aten.matmul.default", "aten.mm.default",
        "aten.addmm.default", "aten.bmm.default", "aten.baddbmm.default"}},
      {"concat", {"aten.cat.default"}},
      {"add", {"aten.add.Tensor"}},
      {"multiply", {"aten.mul.Tensor"}},
      {"activation", {"aten.silu.default", "aten.sigmoid.default",
                      "aten.gelu.default", "aten.relu.default"}},
      {"mean_reduce", {"aten.mean.dim", "aten.sum.dim_IntList"}},
      {"power", {"aten.pow.Tensor_Scalar"}},
      {"rsqrt", {"aten.rsqrt.default"}},
      {"softmax", {"aten.softmax.int", "aten._softmax.default"}},
      {"trig", {"aten.cos.default", "aten.sin.default"}},
      {"negate", {"aten.neg.default"}},
      {"split", {"aten.chunk.default", "aten.split_with_sizes.default",
                 "aten.split.Tensor"}},
      {"mask", {"aten.masked_fill.Scalar", "aten.where.self"}},
  };
  return roles;
}

PatternMatcher::PatternMatcher(std::vector<FxNodeRecord> const& nodes,
                               std::vector<SignatureInput> const& inputs)
    : nodes_(nodes) {
  for (auto const& node : nodes_) by_name_.emplace(node.name, &node);
  for (auto const& role : CoreRoles())
    for (auto const& target : role.targets) role_of_.emplace(target, role.role);
  for (auto const& input : inputs) {
    if (input.kind == "PARAMETER" || input.kind == "BUFFER")
      parameters_.insert(input.name);
    if (input.kind == "USER_INPUT") user_inputs_.insert(input.name);
  }
}

FxNodeRecord const* PatternMatcher::Find(std::string const& name) const {
  auto found = by_name_.find(name);
  return found == by_name_.end() ? nullptr : found->second;
}

bool PatternMatcher::IsParameter(std::string const& name) const {
  return parameters_.count(name) != 0;
}

bool PatternMatcher::IsInput(std::string const& name) const {
  return user_inputs_.count(name) != 0;
}

std::string const& PatternMatcher::RoleOf(FxNodeRecord const& node) const {
  static std::string const none;
  auto found = role_of_.find(node.target);
  return found == role_of_.end() ? none : found->second;
}

std::string PatternMatcher::Value(std::string const& name) const {
  std::string current = name;
  for (int guard = 0; guard < 64; ++guard) {
    FxNodeRecord const* node = Find(current);
    if (!node || node->op != "call_function") return current;
    if (node->target == "<built-in function getitem>") {
      if (node->inputs.empty()) return current;
      current = node->inputs.front();
      continue;
    }
    if (!LayoutOnly().count(node->target) || node->inputs.empty())
      return current;
    current = node->inputs.front();
  }
  return current;
}

bool PatternMatcher::DependsOn(std::string const& value,
                               std::string const& ancestor) const {
  if (value == ancestor) return true;
  std::vector<std::string> work{value};
  std::unordered_set<std::string> seen;
  while (!work.empty()) {
    std::string current = std::move(work.back());
    work.pop_back();
    if (!seen.insert(current).second) continue;
    if (current == ancestor) return true;
    FxNodeRecord const* node = Find(current);
    if (!node) continue;
    for (auto const& input : node->inputs) work.push_back(input);
  }
  return false;
}

bool PatternMatcher::OneOperandMatches(OperandConstraint const& constraint,
                                       std::string const& operand,
                                       PatternBinding const& bound) const {
  for (auto const& clause : constraint.clauses) {
    auto found = bound.find(clause.slot);
    switch (clause.kind) {
      case OperandConstraint::Kind::kParameter:
        if (!IsParameter(Value(operand))) return false;
        break;
      case OperandConstraint::Kind::kNear:
        if (found == bound.end() ||
            FirstOfRole(operand, clause.role, clause.role) !=
                found->second->name)
          return false;
        break;
      case OperandConstraint::Kind::kInput:
        if (!IsInput(Value(operand))) return false;
        break;
      case OperandConstraint::Kind::kValue:
        if (found == bound.end() || Value(operand) != found->second->name)
          return false;
        break;
      case OperandConstraint::Kind::kDepends:
        if (found == bound.end() || !DependsOn(operand, found->second->name))
          return false;
        break;
      case OperandConstraint::Kind::kNotDepends:
        if (found == bound.end() || DependsOn(operand, found->second->name))
          return false;
        break;
    }
  }
  return true;
}

bool PatternMatcher::OperandsMatch(PatternNode const& slot,
                                   FxNodeRecord const& node,
                                   PatternBinding const& bound) const {
  if (slot.operands.empty()) return true;
  if (!slot.unordered) {
    if (node.inputs.size() < slot.operands.size()) return false;
    for (std::size_t i = 0; i < slot.operands.size(); ++i)
      if (!OneOperandMatches(slot.operands[i], node.inputs[i], bound))
        return false;
    return true;
  }
  // Set matching: the exporter's argument order is not part of the semantics
  // for a commutative operator, so try every injective assignment. The arity
  // here is two or three, so the permutation cost is irrelevant.
  if (node.inputs.size() < slot.operands.size()) return false;
  std::vector<std::size_t> pick(node.inputs.size());
  std::iota(pick.begin(), pick.end(), std::size_t{0});
  std::sort(pick.begin(), pick.end());
  do {
    bool ok = true;
    for (std::size_t i = 0; i < slot.operands.size() && ok; ++i)
      ok = OneOperandMatches(slot.operands[i], node.inputs[pick[i]], bound);
    if (ok) return true;
  } while (std::next_permutation(pick.begin(), pick.end()));
  return false;
}

bool PatternMatcher::Admissible(GraphPattern const& pattern, std::size_t index,
                                FxNodeRecord const& node,
                                PatternBinding const& bound) const {
  PatternNode const& slot = pattern.nodes[index];
  if (node.op != "call_function") return false;
  if (!slot.role.empty() && RoleOf(node) != slot.role) return false;
  for (auto const& [name, other] : bound)
    if (other == &node) return false;  // slots bind distinct nodes
  for (auto const& target : slot.ancestor_of) {
    auto found = bound.find(target);
    if (found == bound.end() || !DependsOn(found->second->name, node.name))
      return false;
  }
  for (auto const& target : slot.not_ancestor_of) {
    auto found = bound.find(target);
    if (found == bound.end() || DependsOn(found->second->name, node.name))
      return false;
  }
  return OperandsMatch(slot, node, bound);
}

std::string PatternMatcher::FirstOfRole(std::string const& name,
                                        std::string const& role,
                                        std::string const& blocker) const {
  std::vector<std::string> work{name};
  std::unordered_set<std::string> seen;
  while (!work.empty()) {
    std::string current = std::move(work.front());
    work.erase(work.begin());
    if (!seen.insert(current).second) continue;
    FxNodeRecord const* node = Find(current);
    if (!node) continue;
    if (current != name) {
      if (RoleOf(*node) == role) return current;
      if (RoleOf(*node) == blocker) continue;  // a nearer node of the blocking
                                               // role shadows everything behind
    }
    for (auto const& input : node->inputs) work.push_back(input);
  }
  return {};
}

std::string PatternMatcher::NearestParameter(std::string const& name) const {
  std::vector<std::string> work{name};
  std::unordered_set<std::string> seen;
  while (!work.empty()) {
    std::string current = std::move(work.front());
    work.erase(work.begin());
    if (!seen.insert(current).second) continue;
    if (IsParameter(current)) return current;
    FxNodeRecord const* node = Find(current);
    if (!node) continue;
    for (auto const& input : node->inputs) work.push_back(input);
  }
  return {};
}

std::vector<PatternBinding> PatternMatcher::FindAll(GraphPattern const& pattern,
                                                    PatternBinding seed) const {
  std::vector<PatternBinding> found;
  if (pattern.nodes.empty()) return found;
  PatternBinding bound = std::move(seed);
  std::unordered_set<std::string> seeded;
  for (auto const& [name, node] : bound) seeded.insert(name);
  std::vector<std::size_t> cursor{0};
  std::vector<std::size_t> position{0};
  while (!cursor.empty()) {
    std::size_t slot = cursor.back();
    bool advanced = false;
    if (seeded.count(pattern.nodes[slot].slot)) {
      // A seeded slot is fixed: it yields exactly one "candidate".
      advanced = position.back() == 0;
      position.back() = 1;
    } else
    for (std::size_t& i = position.back(); i < nodes_.size(); ++i) {
      if (!Admissible(pattern, slot, nodes_[i], bound)) continue;
      bound[pattern.nodes[slot].slot] = &nodes_[i];
      ++i;
      advanced = true;
      break;
    }
    if (!advanced) {
      cursor.pop_back();
      position.pop_back();
      if (!cursor.empty() && !seeded.count(pattern.nodes[cursor.back()].slot))
        bound.erase(pattern.nodes[cursor.back()].slot);
      continue;
    }
    if (slot + 1 == pattern.nodes.size()) {
      bool shapes = true;
      for (auto const& constraint : pattern.shape) {
        FxNodeRecord const* a = bound.at(constraint.a);
        FxNodeRecord const* b = bound.at(constraint.b);
        long extent_a = 0, extent_b = 0;
        int index_a = constraint.axis_a < 0
                          ? static_cast<int>(a->shape.size()) + constraint.axis_a
                          : constraint.axis_a;
        int index_b = constraint.axis_b < 0
                          ? static_cast<int>(b->shape.size()) + constraint.axis_b
                          : constraint.axis_b;
        if (index_a < 0 || index_b < 0 ||
            index_a >= static_cast<int>(a->shape.size()) ||
            index_b >= static_cast<int>(b->shape.size()))
          continue;
        if (!StaticExtent(a->shape[index_a], extent_a) ||
            !StaticExtent(b->shape[index_b], extent_b))
          continue;  // a symbolic axis is not evidence against the pattern
        if (extent_a != extent_b) shapes = false;
      }
      if (shapes) found.push_back(bound);
      if (!seeded.count(pattern.nodes[slot].slot))
        bound.erase(pattern.nodes[slot].slot);
      continue;
    }
    cursor.push_back(slot + 1);
    position.push_back(0);
  }
  return found;
}

PatternBinding PatternMatcher::MatchOne(GraphPattern const& pattern) const {
  std::vector<PatternBinding> all = FindAll(pattern);
  if (all.empty())
    throw std::runtime_error("pattern " + pattern.name + " does not match");
  if (all.size() > 1)
    throw std::runtime_error("pattern " + pattern.name + " matches " +
                             std::to_string(all.size()) + " times");
  return all.front();
}

}  // namespace tilemega::frontend
