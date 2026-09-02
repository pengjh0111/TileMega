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
Tier TierClassifier::Classify(AffineRelation const& relation) const {
  return relation.empty() ? Tier::kDataDependent : Tier::kAffine;
}
std::vector<EventRequirement> EventSynthesis::Synthesize(std::vector<std::string> const& edges) const {
  std::vector<EventRequirement> result;
  for (auto const& edge : edges) result.push_back({edge, edge, "1"});
  return result;
}
}  // namespace tilemega::analysis
