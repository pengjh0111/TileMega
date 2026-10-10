// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingDerivation.h>

namespace tilemega::analysis {
enum class StorageHazardKind { kWAR, kWAW };
struct StorageTaskAccess {
  OperatorNode const* task = nullptr;
  // All accesses use the same physical storage coordinate space, after the
  // memory planner has applied each tensor's layout and arena offset.
  CouplingRelation elements;
  CouplingAttributes attributes;
};
struct StorageHazard {
  StorageHazardKind kind;
  CouplingEdge coupling;
};

std::vector<StorageHazard> DeriveStorageReuseHazards(
    StorageTaskAccess const& previous_write,
    std::vector<StorageTaskAccess> const& previous_reads,
    StorageTaskAccess const& next_write, ParamBinding const& known = {});
}  // namespace tilemega::analysis
