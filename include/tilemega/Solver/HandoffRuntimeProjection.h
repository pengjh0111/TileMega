// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingRelation.h>
#include <string>
#include <vector>
namespace tilemega::solver {
struct RuntimeHandoff {
  int producer=-1,consumer=-1;
  std::string kind;
  // Runtime task coordinates, after projecting both phase ownership maps.
  analysis::CouplingRelation consumer_to_producer;
  bool retain_producer=false;
};
struct HandoffRuntimeProjection {
  analysis::CouplingRelation tasks,dependencies;
  analysis::CouplingRelation phases; // new [stage,task] -> original [stage,task]
  std::vector<int> surviving_stages;
};
// Dependencies are composed through the phases retained by the rewritten CG.
// Internal handoff edges disappear; every external predecessor is preserved.
HandoffRuntimeProjection ProjectHandoffRuntime(
    analysis::CouplingRelation const& tasks,
    analysis::CouplingRelation const& dependencies,int stage_count,
    std::vector<RuntimeHandoff> const& handoffs);
}
