// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/StorageHazards.h>
#include <tilemega/Frontend/ModelPlan.h>

namespace tilemega::solver {
struct MemoryAlias {
  unsigned buffer;
  std::uint64_t offset, bytes;
};
struct MemoryPlan {
  std::uint64_t arena_bytes = 0, live_peak_bytes = 0;
  bool fits_l2_budget = false;
  std::vector<MemoryAlias> aliases;
  std::vector<analysis::StorageHazard> hazards;
};
MemoryPlan PlanBufferReuse(frontend::ModelPlan const&,
    analysis::OperatorGraph const&, analysis::ParamBinding const&,
    std::string const& policy, std::uint64_t l2_budget_bytes = 0);
} // namespace tilemega::solver
