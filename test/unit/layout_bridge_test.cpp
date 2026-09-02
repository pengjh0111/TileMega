// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CuteLayoutBridge.h>
#include <cassert>

int main() {
  using namespace tilemega::analysis;
  CuteLayoutBridge bridge;
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
  return 0;
}
