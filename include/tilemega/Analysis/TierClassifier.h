// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §3.2 communication tier selection (Phase 3 stub).
#pragma once
#include <tilemega/Analysis/DerivedMetrics.h>
namespace tilemega::analysis {
enum class CommunicationTier { kRegister, kShared, kCluster, kGlobal };
class TierClassifier {
 public:
  CommunicationTier Classify(DerivedMetrics const&) const;
};
}  // namespace tilemega::analysis
