// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 invariant I2 and §5.3 event synthesis.
#pragma once
#include <vector>
#include <tilemega/Analysis/CouplingDerivation.h>
namespace tilemega::analysis {
struct EventRequirement {
  TaskSpaceId producer;
  TaskSpaceId consumer;
  std::vector<ClosedForm> shape;  ///< image(C_kappa)
  ClosedForm wait;
  ClosedForm fanout;
  ClosedForm count;
  Tier tier = Tier::kAffine;
  SyncKind sync = SyncKind::kGlobal;
  bool exact = true;
  std::string guard;
};
class EventSynthesis {
 public:
  /// Synthesize one event tensor requirement per derived C edge.  No printed
  /// relation or scalar surrogate is accepted, preserving I1 through L3b.
  std::vector<EventRequirement> Synthesize(
      std::vector<CouplingEdge> const& edges) const;
};
}  // namespace tilemega::analysis
