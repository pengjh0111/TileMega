// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/VirtualTaskBinding.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <stdexcept>
#include <algorithm>

namespace tilemega::analysis {
std::vector<VirtualDimensionBinding> VirtualBindings(SemanticOp const& op) {
  std::vector<VirtualDimensionBinding> bindings;
  for (auto const& dim : op.domain) {
    if (!dim.capacity) {
      if (!dim.binding_source.empty() || !dim.binding_requirement.empty())
        throw std::invalid_argument("runtime binding lacks a capacity");
      continue;
    }
    if (!op.exact_task_access || !dim.runtime || dim.binding_source.empty() ||
        (dim.binding_requirement != "prefix_sum" && dim.binding_requirement != "tensor_values") ||
        (dim.capacity->IsConstant() && dim.capacity->Eval({}, {}) <= 0))
      throw std::invalid_argument("invalid virtual dimension capacity or binding");
    auto symbols = dim.capacity->FreeSymbols();
    if (std::find(symbols.begin(), symbols.end(), dim.name) != symbols.end())
      throw std::invalid_argument("virtual capacity references its runtime coordinate");
    bindings.push_back({dim.name, dim.binding_source, dim.binding_requirement, *dim.capacity});
  }
  return bindings;
}
TensorSpace BindCapacityTaskSpace(SemanticOp const& op) {
  auto space = op.task_space;
  for (auto const& binding : VirtualBindings(op)) {
    bool found = false;
    for (unsigned axis = 0; axis < op.task_map.results.size(); ++axis) {
      auto const& map = op.task_map.results[axis];
      if (map.kind != IndexResult::Kind::kAffine || map.terms.size() != 1 ||
          map.terms[0].dim != binding.dimension) continue;
      auto const& term = map.terms[0];
      if (!term.coefficient.IsLiteral(1) || !term.group.IsLiteral(1) || !map.outer_divisor.IsLiteral(1) ||
          !term.shift.IsLiteral(0) || !map.offset.IsLiteral(0) || axis >= space.axes.size())
        throw std::invalid_argument("virtual ownership must expose its capacity coordinate directly");
      space.axes[axis].extent = binding.capacity;
      space.axes[axis].origin = op.Dim(binding.dimension)->origin;
      // The live extent belongs to the binding. Every capacity slot has a
      // scheduled task and completion event, including invalid/empty slots.
      space.axes[axis].runtime = false; found = true;
    }
    if (!found) throw std::invalid_argument("virtual dimension has no ownership coordinate");
  }
  return space;
}
RuntimeRequirement VirtualBindingRequirement(OperatorNode const& task) {
  auto requirement = RuntimeRequirement::kNone;
  if (!task.element_access) return requirement;
  for (auto const& binding : VirtualBindings(task.element_access->semantic)) {
    if (binding.requirement == "tensor_values") return RuntimeRequirement::kTensorValues;
    requirement = RuntimeRequirement::kPrefixSum;
  }
  return requirement;
}
}  // namespace tilemega::analysis
