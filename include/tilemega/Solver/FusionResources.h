// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/FusionAccess.h>
#include <tilemega/Solver/BackendCostQuery.h>
#include <tilemega/Solver/CostModel.h>
#include <map>

namespace tilemega::solver {
struct ModelFusionCandidate;
#ifndef TILEMEGA_FUSION_TASK_COST
#define TILEMEGA_FUSION_TASK_COST 1
#endif
struct FusionTaskPrice {
  long producer_tasks=0,consumer_tasks=0,recomputed_tasks=0;
  long producer_waves=0,consumer_waves=0;
  double separate_ns=0,fused_ns=0,recompute_ns=0;
  double global_bytes_before=0,global_bytes_after=0,local_bytes=0;
};
// Task-only price at an externally pinned whole-kernel residency. Events
// belong to the reprojected L-sched, not to a guessed per-logical-edge fee.
FusionTaskPrice PriceFusionTasks(ModelFusionCandidate const& candidate,
    CostModel const& cost,BackendTraits const& producer,BackendTraits const& consumer,
    Residency residency,ModelDescription const& model);
struct FusionResources {
  long intermediate_bytes = 0;
  int shared_bytes = 0, registers = 0, threads = 0;
};
// Allocation granularity is supplied by the backend resource contract.
// Returns zero for an impossible CTA; never promotes it to residency one.
int FusionCtasPerSm(FusionResources const& resources, TargetSpec const& target,
                    int register_allocation_per_warp, int static_shared_bytes);
double FusionRecomputeNs(analysis::FusionAccesses const& accesses,
    CostModel const& cost, DerivedTaskInput const& producer, BackendTraits const& traits,
    Residency residency, ModelDescription const& model, int chunks,
    double active_ctas_per_sm);
// Registers are tier-3 input, not inferred from the access relation.
FusionResources DeriveFusionResources(analysis::FusionAccesses const& accesses,
    analysis::ParamBinding const& theta, std::map<std::string,int> const& element_bytes,
    BackendTraits const& producer, int producer_registers,
    BackendTraits const& consumer, int consumer_registers,
    long allocated_intermediate_bytes=0);
}  // namespace tilemega::solver
