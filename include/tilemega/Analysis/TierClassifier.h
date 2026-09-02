// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2.4 coupling analyzability tier (Part 4).
#pragma once
#include <tilemega/Analysis/CouplingDerivation.h>
namespace tilemega::analysis {
/// Structural Tier reclassification from the relation alone (no access-map
/// context): an isl_map query, not a re-derivation. CouplingDerivation's own
/// Tier assignment (which also consults layout ids and runtime task-space
/// flags CouplingRelation cannot see) remains the primary classifier; this
/// exists for Part 4's isl-side cross-check (Tier 0 iff C is a single-valued
/// affine map with no relaxation-shaped disjunct).
class TierClassifier {
 public:
  Tier Classify(CouplingRelation const& relation) const;
};
}  // namespace tilemega::analysis
