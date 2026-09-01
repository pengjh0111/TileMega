// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 3-5 and §2.4 tiers.
#pragma once
#include <vector>
#include <tilemega/Analysis/AccessRelation.h>
#include <tilemega/Analysis/AffineRelation.h>
#include <tilemega/Analysis/DerivedMetrics.h>
namespace tilemega::analysis {
enum class Tier {
  kAffine = 0,
  kSharedInjectiveLayout = 1,
  kStructuredRagged = 2,
  kDataDependent = 3,
};
enum class SyncKind { kGlobal, kCluster, kLocal };
struct CouplingEdge {
  TaskSpaceId src;
  TaskSpaceId dst;
  AffineRelation C;
  DerivedMetrics metrics;
  Tier tier = Tier::kAffine;
  SyncKind sync = SyncKind::kGlobal;
};
class CouplingDerivation {
 public:
  /// Phase-3 entry point. It must return structured relations; the current
  /// stub never converts a diagnostic string into a false semantic relation.
  std::vector<CouplingEdge> Derive(std::vector<AccessRelation> const&) const;
};
}  // namespace tilemega::analysis
