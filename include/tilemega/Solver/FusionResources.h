// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/FusionAccess.h>
#include <tilemega/Solver/BackendCostQuery.h>
#include <map>

namespace tilemega::solver {
struct FusionResources {
  long intermediate_bytes = 0;
  int shared_bytes = 0, registers = 0, threads = 0;
};
// Allocation granularity is supplied by the backend resource contract.
// Returns zero for an impossible CTA; never promotes it to residency one.
int FusionCtasPerSm(FusionResources const& resources, TargetSpec const& target,
                    int register_allocation_per_warp, int static_shared_bytes);
// Registers are tier-3 input, not inferred from the access relation.
FusionResources DeriveFusionResources(analysis::FusionAccesses const& accesses,
    analysis::ParamBinding const& theta, std::map<std::string,int> const& element_bytes,
    BackendTraits const& producer, int producer_registers,
    BackendTraits const& consumer, int consumer_registers);
}  // namespace tilemega::solver
