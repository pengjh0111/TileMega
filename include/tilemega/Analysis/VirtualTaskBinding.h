// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/Semantics.h>
#include <tilemega/Analysis/CouplingDerivation.h>

namespace tilemega::analysis {
struct VirtualDimensionBinding {
  std::string dimension, source, requirement;
  ClosedForm capacity;
};
std::vector<VirtualDimensionBinding> VirtualBindings(SemanticOp const& op);
TensorSpace BindCapacityTaskSpace(SemanticOp const& op);
RuntimeRequirement VirtualBindingRequirement(OperatorNode const& task);
}  // namespace tilemega::analysis
