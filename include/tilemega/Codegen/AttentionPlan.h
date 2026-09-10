// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
#include <cstdint>

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

TILEMEGA_TASK_HD constexpr int AttentionPhaseTasks(AttentionPhase phase,
    int queries, int chunks) {
  return (phase == AttentionPhase::kScores || phase == AttentionPhase::kPartialValue)
      ? queries * chunks : queries;
}
TILEMEGA_TASK_HD constexpr int AttentionChunkBegin(int total, int chunk, int chunks) {
  return static_cast<int>(static_cast<std::int64_t>(total) * chunk / chunks);
}
}  // namespace tilemega::codegen
