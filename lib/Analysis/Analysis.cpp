// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/CuteLayoutBridge.h>
#include <tilemega/Analysis/EventSynthesis.h>
#include <tilemega/Analysis/TierClassifier.h>
namespace tilemega::analysis {
LayoutProjection CuteLayoutBridge::Project(std::string const& relation) const {
  // TODO(P3.2): lower the Presburger relation into the selected layout algebra.
  return {relation, false};
}
std::vector<CouplingEdge> CouplingDerivation::Derive(std::vector<AccessRelation> const& accesses) const {
  std::vector<CouplingEdge> result;
  for (auto const& a : accesses) result.push_back({a.producer, a.consumer, a.presburger_map});
  return result;
}
CommunicationTier TierClassifier::Classify(DerivedMetrics const& metrics) const {
  // TODO(P3.3): use target budgets and schedule legality, not this placeholder.
  return metrics.reuse > 0 ? CommunicationTier::kShared : CommunicationTier::kGlobal;
}
std::vector<EventRequirement> EventSynthesis::Synthesize(std::vector<std::string> const& edges) const {
  std::vector<EventRequirement> result;
  for (auto const& edge : edges) result.push_back({edge, edge, "1"});
  return result;
}
}  // namespace tilemega::analysis
