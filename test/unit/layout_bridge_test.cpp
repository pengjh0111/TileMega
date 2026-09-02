// SPDX-License-Identifier: BSD-3-Clause
// P3.1: the three-level inverse policy (V-F), and the CuTe <-> isl_map
// conversion itself -- §3.5's rule that a flattened layout
// `(s_0,...,s_k):(d_0,...,d_k)` is the map
// `[i_0..i_k] -> [offset + sum i_j*d_j] : 0 <= i_j < s_j`.
#include <tilemega/Analysis/CuteLayoutBridge.h>

#include <cassert>
#include <string>

using namespace tilemega::analysis;

namespace {

LayoutDescriptor Layout(std::vector<long> extents, std::vector<long> strides) {
  LayoutDescriptor layout;
  for (long extent : extents)
    layout.extents.push_back(ClosedForm::Constant(extent));
  for (long stride : strides)
    layout.strides.push_back(ClosedForm::Constant(stride));
  layout.static_tile_shape = true;
  return layout;
}

/// A layout survives CuTe -> isl_map -> CuTe unchanged.
void RoundTrips(CuteLayoutBridge const& bridge, LayoutDescriptor const& layout) {
  LayoutDescriptor back = bridge.FromIslMap(bridge.ToIslMap(layout));
  assert(back.extents.size() == layout.extents.size());
  assert(back.strides.size() == layout.strides.size());
  for (std::size_t i = 0; i < layout.extents.size(); ++i) {
    assert(back.extents[i].ToString() == layout.extents[i].ToString());
    assert(back.strides[i].ToString() == layout.strides[i].ToString());
  }
}

}  // namespace

int main() {
  CuteLayoutBridge bridge;

  // --- The three-level inverse policy (V-F) ---------------------------------
  LayoutDescriptor fixed{{ClosedForm::Constant(128)},
                         {ClosedForm::Constant(1)}, true};
  assert(bridge.Project(fixed).strategy ==
         InverseStrategy::kCuteStaticRightInverse);
  LayoutDescriptor symbolic{{ClosedForm::Symbol("S")},
                            {ClosedForm::Constant(1)}, false};
  assert(bridge.Project(symbolic).strategy ==
         InverseStrategy::kPresburgerRelation);
  LayoutDescriptor shared = symbolic;
  shared.layout_id = "kv_cache";
  assert(bridge.Project(shared, &shared).strategy ==
         InverseStrategy::kCancelSharedLayout);
  symbolic.dynamic_stride = true;
  assert(bridge.Project(symbolic).minimum_tier == Tier::kStructuredRagged);
  symbolic.dynamic_stride = false;
  symbolic.swizzled = true;
  assert(bridge.Project(symbolic).minimum_tier == Tier::kDataDependent);

  // --- CuTe layout -> isl_map ----------------------------------------------
  // Row-major (4,8):(8,1): offset = 8*i0 + i1 over the 4x8 box.
  CouplingRelation row_major = bridge.ToIslMap(Layout({4, 8}, {8, 1}));
  assert(row_major.ToString().find("8i0") != std::string::npos ||
         row_major.ToString().find("8*i0") != std::string::npos);
  // It is a function: one offset per coordinate.
  assert(row_major.IsSingleValued());
  // A layout is injective exactly when distinct coordinates get distinct
  // offsets, which for a map is its inverse being single-valued.
  assert(row_major.Reverse().IsSingleValued());

  // Column-major (4,8):(1,4) covers the same 32 offsets by a different route.
  CouplingRelation col_major = bridge.ToIslMap(Layout({4, 8}, {1, 4}));
  assert(col_major.IsSingleValued());
  assert(col_major.Reverse().IsSingleValued());
  assert(row_major != col_major);

  // A broadcast mode (stride 0) is a layout but not injective, and isl says
  // so without being told: several coordinates share one offset.
  CouplingRelation broadcast = bridge.ToIslMap(Layout({4, 8}, {1, 0}));
  assert(broadcast.IsSingleValued());
  assert(!broadcast.Reverse().IsSingleValued());

  // --- Round trip ----------------------------------------------------------
  RoundTrips(bridge, Layout({4, 8}, {8, 1}));        // row major
  RoundTrips(bridge, Layout({4, 8}, {1, 4}));        // column major
  RoundTrips(bridge, Layout({128}, {1}));            // 1-D contiguous
  RoundTrips(bridge, Layout({2, 3, 5}, {15, 5, 1})); // 3-D row major
  RoundTrips(bridge, Layout({8, 4}, {1, 16}));       // strided/padded

  // --- A symbolic extent stays an isl parameter (invariant I1) -------------
  LayoutDescriptor dynamic_rows;
  dynamic_rows.extents = {ClosedForm::Symbol("S"), ClosedForm::Constant(128)};
  dynamic_rows.strides = {ClosedForm::Constant(128), ClosedForm::Constant(1)};
  CouplingRelation ragged = bridge.ToIslMap(dynamic_rows);
  assert(ragged.ToString().find("S") != std::string::npos);
  // ... but writing one back is refused rather than guessed: a solver's
  // chosen layout is concrete, a symbolic extent belongs to the model.
  bool refused_symbolic = false;
  try {
    (void)bridge.FromIslMap(ragged);
  } catch (std::domain_error const&) {
    refused_symbolic = true;
  }
  assert(refused_symbolic);

  // --- The two rejections are explicit, not silent approximations ----------
  LayoutDescriptor swizzled = Layout({4, 8}, {8, 1});
  swizzled.swizzled = true;
  bool refused_swizzle = false;
  try {
    (void)bridge.ToIslMap(swizzled);
  } catch (std::domain_error const&) {
    refused_swizzle = true;
  }
  assert(refused_swizzle);

  LayoutDescriptor dynamic_stride;
  dynamic_stride.extents = {ClosedForm::Constant(4)};
  dynamic_stride.strides = {ClosedForm::Symbol("ld")};  // parameter * coordinate
  bool refused_stride = false;
  try {
    (void)bridge.ToIslMap(dynamic_stride);
  } catch (std::domain_error const&) {
    refused_stride = true;
  }
  assert(refused_stride);
  // Binding the stride to a literal makes the same layout representable --
  // the boundary is "is it a literal by construction time", not "was it ever
  // written as a symbol".
  ParamBinding known;
  known.Bind("ld", 16);
  CouplingRelation bound = bridge.ToIslMap(dynamic_stride, known);
  assert(bound.IsSingleValued());
  LayoutDescriptor back = bridge.FromIslMap(bound);
  assert(back.strides.front().ToString() == "16");

  return 0;
}
