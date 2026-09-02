// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.2 and V-F's three-level inverse policy.
#pragma once
#include <string>
#include <vector>
#include <tilemega/Analysis/CouplingDerivation.h>
namespace tilemega::analysis {
struct LayoutDescriptor {
  std::vector<ClosedForm> extents;
  std::vector<ClosedForm> strides;
  bool static_tile_shape = false;
  bool dynamic_stride = false;
  bool swizzled = false;
  bool injective = true;
  std::string layout_id;
};
enum class InverseStrategy {
  kCuteStaticRightInverse,
  kPresburgerRelation,
  kCancelSharedLayout,
  kRaiseTier,
};
struct LayoutProjection {
  InverseStrategy strategy = InverseStrategy::kRaiseTier;
  Tier minimum_tier = Tier::kDataDependent;
  bool presburger_affine = false;
  std::string reason;
};
class CuteLayoutBridge {
 public:
  LayoutProjection Project(LayoutDescriptor const& write,
                           LayoutDescriptor const* read = nullptr) const;

  /// A flat CuTe layout `(s_0,...,s_k):(d_0,...,d_k)` as the isl_map
  ///
  ///     [i_0,...,i_k] -> [offset + sum_j i_j * d_j] : 0 <= i_j < s_j
  ///
  /// This is §3.5's conversion rule, and it presumes the layout has already
  /// been flattened/coalesced on the CuTe side -- a nested or composed
  /// layout must be flattened first, which is exactly why `Project` rejects
  /// a swizzled (non-flattenable) layout instead of approximating it.
  ///
  /// `known` fixes the symbols that must be literal for the result to be an
  /// isl affine map. Extents may stay symbolic -- they are only bounds, and
  /// leaving e.g. the sequence length as an isl parameter is what keeps
  /// invariant I1. Strides may not: `stride * coordinate` with a symbolic
  /// stride is parameter times variable, which is not Presburger affine.
  /// That is not an isl quirk to work around, it is the same boundary V-F
  /// found in CuTe (`RightInverse` rejects a dynamic-shape layout), arrived
  /// at from the other side.
  ///
  /// Throws std::domain_error, naming the offending stride, when the layout
  /// is not representable. Callers that want the Tier consequence without an
  /// exception should ask `Project` first.
  CouplingRelation ToIslMap(LayoutDescriptor const& layout,
                            ParamBinding const& known = {}) const;

  /// The write-back direction: read shape and stride back off a
  /// single-valued affine isl_map produced by `ToIslMap` (or by a solver
  /// that rewrote one).
  ///
  /// Extents are required to be literal here, which is a real restriction
  /// and a deliberate one: what comes back from a solver is a *chosen*
  /// layout, and a choice is concrete. A symbolic extent belongs to the
  /// model, not to a solver result, so it is rejected rather than guessed.
  /// Throws std::domain_error if the map is not a single-valued affine map
  /// over a bounded rectangular domain.
  LayoutDescriptor FromIslMap(CouplingRelation const& map) const;
};
}  // namespace tilemega::analysis
