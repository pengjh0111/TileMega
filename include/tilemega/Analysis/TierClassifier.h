// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2.4 coupling analyzability tier (Phase 3 stub).
#pragma once
#include <tilemega/Analysis/CouplingDerivation.h>
namespace tilemega::analysis {
class TierClassifier {
 public:
  Tier Classify(AffineRelation const&) const;
};
}  // namespace tilemega::analysis
