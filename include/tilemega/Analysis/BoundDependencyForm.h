// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/DependencyForm.h>
#include <tilemega/Analysis/DependencyTable.h>

namespace tilemega::analysis {
struct BoundDependencyForm {
  enum class Encoding { kWindow, kTable } encoding = Encoding::kTable;
  WaitWindow window;
  std::optional<DependencyTable> table;
  CouplingRelation encoded_relation;
};

BoundDependencyForm BindExactTaskDependencyLinear(CouplingRelation const& relation,
    std::uint32_t producers, std::uint32_t consumers);
BoundDependencyForm BindExactTaskDependency(CouplingEdge const& edge,
    OperatorNode const& producer, OperatorNode const& consumer,
    ParamBinding const& known);
}  // namespace tilemega::analysis
