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

CouplingRelation LinearizeTaskCoordinates(OperatorNode const& node,
    std::vector<std::string> const& coordinates, ParamBinding const& known,
    char const* id);
// Prove serialized canonical interval rows directly against the source map.
// Does not recover endpoints or count relation pairs.
void ValidateDependencyTableLinear(DependencyTable const& table);
DependencyTable BuildDependencyTableLinear(CouplingRelation const& relation,
    std::uint32_t producers, std::uint32_t consumers);
CouplingRelation LinearizeTaskCoupling(CouplingRelation const& relation,
    OperatorNode const& producer, OperatorNode const& consumer,
    ParamBinding const& known);
DependencyTable BuildDependencyTable(CouplingRelation const& relation,
    OperatorNode const& producer, OperatorNode const& consumer,
    ParamBinding const& known);
}  // namespace tilemega::analysis
