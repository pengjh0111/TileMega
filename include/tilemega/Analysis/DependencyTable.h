// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingDerivation.h>
#include <cstdint>

namespace tilemega::analysis {
struct TaskInterval {
  std::uint32_t first = 0, count = 0;
};
struct DependencyTable {
  std::uint32_t consumers = 0, producers = 0, stride = 0;
  // Each row is padded with zero-count intervals to the edge's maximum.
  std::vector<TaskInterval> intervals;
  CouplingRelation linear_relation;
  CouplingRelation encoded_relation;
};

CouplingRelation LinearizeTaskCoupling(CouplingRelation const& relation,
    OperatorNode const& producer, OperatorNode const& consumer,
    ParamBinding const& known);
DependencyTable BuildDependencyTable(CouplingRelation const& relation,
    OperatorNode const& producer, OperatorNode const& consumer,
    ParamBinding const& known);
}  // namespace tilemega::analysis
