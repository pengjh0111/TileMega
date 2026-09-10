// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
#include <cstdint>
#include <array>

#ifndef TILEMEGA_CHUNKED_ATTENTION
#define TILEMEGA_CHUNKED_ATTENTION 0
#endif

namespace tilemega::codegen {
enum class AttentionPhase : std::uint32_t {
  kDirect = 0,
  kScores = 1,
  kNormalize = 2,
  kPartialValue = 3,
  kCombine = 4,
};
struct AttentionRuntimeRecord {
  std::uint32_t chunks = 1;
  // Capacity of the compiled chunk scratch, not the workload's total length.
  std::uint32_t chunk_extent = 0;
};
struct AttentionPlanSelection {
  int stage = -1;
  AttentionRuntimeRecord runtime;
};
inline constexpr std::array<AttentionPhase, 4> kAttentionExpandedPhases{
    AttentionPhase::kScores, AttentionPhase::kNormalize,
    AttentionPhase::kPartialValue, AttentionPhase::kCombine};
struct AttentionInternalDependency {
  int producer, consumer, div, scale, count;
};
constexpr std::array<AttentionInternalDependency, 3> AttentionInternalDependencies(int chunks) {
  return {{{0,1,1,chunks,chunks}, {1,2,chunks,1,1}, {2,3,1,chunks,chunks}}};
}

TILEMEGA_TASK_HD constexpr int AttentionPhaseTasks(AttentionPhase phase,
    int queries, int chunks) {
  return (phase == AttentionPhase::kScores || phase == AttentionPhase::kPartialValue)
      ? queries * chunks : queries;
}
TILEMEGA_TASK_HD constexpr int AttentionChunkBegin(int total, int chunk, int chunks) {
  return static_cast<int>(static_cast<std::int64_t>(total) * chunk / chunks);
}
}  // namespace tilemega::codegen
