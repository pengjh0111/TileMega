// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2 Definition 3 derived coupling edges (Phase 3 stub).
#pragma once
#include <vector>
#include <tilemega/Analysis/AccessRelation.h>
namespace tilemega::analysis {
struct CouplingEdge { std::string source; std::string target; std::string relation; };
class CouplingDerivation {
 public:
  std::vector<CouplingEdge> Derive(std::vector<AccessRelation> const&) const;
};
}  // namespace tilemega::analysis
