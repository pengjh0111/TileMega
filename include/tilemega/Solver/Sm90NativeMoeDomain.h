// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Support/Json.h>

namespace tilemega::solver {
inline constexpr char kSm90NativeMoeDomainApproximation[]=
    "approximate_expected_active_expert_tile_tasks_round_nearest";

double Sm90NativeExpectedActiveExperts(int experts,int top_k,int batch);
int Sm90NativeExpectedExpertTasks(int experts,int top_k,int batch,int tiles);

struct Sm90NativeMoeTaskDomain {
  int tasks=0,tiles=0;
  double expected_active=0;
  // The user-approved finite prediction domain can end within an expert.
  // This projection describes that approximation, never an observed route.
  analysis::CouplingRelation ownership; // task -> [active slot, column tile]
  json::Value ToJson() const;
};
Sm90NativeMoeTaskDomain Sm90NativeMoeDomain(int experts,int top_k,int batch,int tiles);
analysis::CouplingRelation Sm90NativeMoeExpertCoupling(
    Sm90NativeMoeTaskDomain const& producer,Sm90NativeMoeTaskDomain const& consumer);
analysis::CouplingRelation Sm90NativeMoeCombineCoupling(Sm90NativeMoeTaskDomain const& down);
} // namespace tilemega::solver
