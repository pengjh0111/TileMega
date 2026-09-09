// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>

#ifndef TILEMEGA_ATTENTION_MAX_TOTAL
#define TILEMEGA_ATTENTION_MAX_TOTAL 4096
#endif

namespace tilemega::codegen {

// Shared storage belongs to the implementation, including the one-float
// empty-storage ABI. The GEMM collective provides its own storage type.
constexpr int SimtSharedElements(TaskKind kind, int threads, int attention_extent) {
  switch (kind) {
    case TaskKind::kRMSNorm: return threads;
    case TaskKind::kAttention: return attention_extent;
    case TaskKind::kRoPE:
    case TaskKind::kKVAppend:
    case TaskKind::kElementwise:
    case TaskKind::kGemmCombine: return 1;
    case TaskKind::kGemm: return 0;
  }
  return 0;
}

template <TaskKind Kind, int Threads, int AttentionExtent = TILEMEGA_ATTENTION_MAX_TOTAL>
struct SimtTaskResources {
  static_assert(Kind != TaskKind::kGemm, "read GEMM resources from its collective");
  static_assert(Threads>0 && AttentionExtent>0, "invalid TaskBody resource dimensions");
  using SharedStorage = float[SimtSharedElements(Kind,Threads,AttentionExtent)];
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr int kStages = 0;  // No asynchronous multistage mainloop.
};

}  // namespace tilemega::codegen
