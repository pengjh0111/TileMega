// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Solver/EftPlacement.h>

namespace tilemega::solver {
struct SkeletonRequest {
  PlanSkeleton const* skeleton=nullptr;
  HopCurve hop;
  int sms=0,ctas_per_sm=1;
  std::vector<int> worker_sm;
  std::vector<ResourceVector> task_lanes;
  bool pure_template=false;
};
struct SkeletonPlacementStats {
  std::uint64_t placed=0,affinity=0,home=0,spread_other=0,candidate_sum=0;
  std::uint64_t lazy_requeues=0,transitions=0,adjacent_slots=0;
  // Affinity and home overlap; the exclusive destination bins above cannot
  // recover displacement from the structural template.
  std::uint64_t moved_from_home=0;
  double interleaving=0;
};
bool ScheduleBySkeleton(SkeletonRequest const& request,EftSchedule* out,
    SkeletonPlacementStats* stats,std::string* error);
}
