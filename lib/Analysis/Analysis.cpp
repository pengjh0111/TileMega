// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/CuteLayoutBridge.h>
#include <tilemega/Analysis/EventSynthesis.h>
#include <tilemega/Analysis/TierClassifier.h>
#include <stdexcept>
namespace tilemega::analysis {
LayoutProjection CuteLayoutBridge::Project(
    LayoutDescriptor const& write, LayoutDescriptor const* read) const {
  if (read && !write.layout_id.empty() &&
      write.layout_id == read->layout_id && write.injective && read->injective)
    return {InverseStrategy::kCancelSharedLayout,
            Tier::kSharedInjectiveLayout, true,
            "shared injective layout cancels before inversion"};
  if (write.swizzled || (read && read->swizzled))
    return {InverseStrategy::kRaiseTier, Tier::kDataDependent, false,
            "CuTe swizzle is not a Presburger affine stride map"};
  if (write.dynamic_stride || (read && read->dynamic_stride))
    return {InverseStrategy::kRaiseTier, Tier::kStructuredRagged, false,
            "parameter*coordinate dynamic stride is not Presburger affine"};
  if (!write.injective)
    return {InverseStrategy::kRaiseTier, Tier::kStructuredRagged, false,
            "write layout is not injective"};
  if (write.static_tile_shape)
    return {InverseStrategy::kCuteStaticRightInverse, Tier::kAffine, true,
            "g-specialized static tile admits cute.right_inverse"};
  return {InverseStrategy::kPresburgerRelation, Tier::kAffine, true,
          "symbolic domain extent with constant strides is Presburger affine"};
}
Tier TierClassifier::Classify(CouplingRelation const& relation) const {
  if (relation.empty()) return Tier::kDataDependent;
  return relation.IsSingleValued() ? Tier::kAffine : Tier::kStructuredRagged;
}
std::vector<EventRequirement> EventSynthesis::Synthesize(
    std::vector<CouplingEdge> const& edges) const {
  std::vector<EventRequirement> result;
  result.reserve(edges.size());
  for (auto const& edge : edges) {
    auto shape = edge.event_shape;
    if (shape.empty() && edge.tier == Tier::kDataDependent)
      shape.push_back(ClosedForm::Constant(1));  // operator-level barrier
    if (shape.empty())
      throw std::invalid_argument("non-Tier-3 coupling has empty image(C_kappa): " +
                                  edge.src.name + " -> " + edge.dst.name);
    result.push_back({edge.src, edge.dst, std::move(shape),
                      edge.metrics.wait, edge.metrics.fanout,
                      edge.metrics.count, edge.tier, edge.sync,
                      edge.exact, edge.guard});
  }
  return result;
}
}  // namespace tilemega::analysis
