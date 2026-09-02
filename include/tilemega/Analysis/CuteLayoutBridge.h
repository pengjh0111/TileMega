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
};
}  // namespace tilemega::analysis
