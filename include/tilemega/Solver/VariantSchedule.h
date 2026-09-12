// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §5.7.4 (the solver decides a schedule), §8.11 (codegen and
//                the host only consume one).
//
// The legacy stage permutation: one entry per stage in the order the L1 path
// executes them, with that stage's slice of the consumer-sorted dependency
// table.  It is the `legacy_grid_stride` input of the Plan contract (§5.7.1)
// and stays here so that `lib/Codegen` holds no scheduling decision at all.
#pragma once

#include <tilemega/Codegen/RuntimePlan.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tilemega::solver {

struct StageScheduleEntry {
  std::uint32_t stage;
  std::uint32_t dependency_begin;
  std::uint32_t dependency_count;
};

struct VariantStageSchedule {
  std::vector<StageScheduleEntry> schedule;
  /// Maximum producer-to-consumer distance in the emitted order; a
  /// non-positive distance is a cycle and is rejected rather than returned.
  std::uint32_t max_dependency_span = 0;
};

/// `dependencies` must already be sorted by consumer, which is what makes each
/// stage's slice a contiguous range.  Throws std::invalid_argument on a stage
/// index outside the model or on a cycle.
VariantStageSchedule BuildVariantStageSchedule(
    std::vector<codegen::DependencyRecord> const& dependencies,
    std::size_t stage_count);

}  // namespace tilemega::solver
