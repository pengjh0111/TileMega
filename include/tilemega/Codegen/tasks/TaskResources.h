// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/TaskBase.h>
#include <tilemega/Codegen/tasks/ScalarDataflow.h>

#ifndef TILEMEGA_ATTENTION_MAX_TOTAL
#define TILEMEGA_ATTENTION_MAX_TOTAL 4096
#endif
#ifndef TILEMEGA_ATTENTION_SCRATCH_EXTENT
#define TILEMEGA_ATTENTION_SCRATCH_EXTENT TILEMEGA_ATTENTION_MAX_TOTAL
#endif

namespace tilemega::codegen {

// Shared storage belongs to the implementation, including the one-float
// empty-storage ABI. The GEMM collective provides its own storage type.
constexpr int SimtSharedElements(TaskKind kind, int threads, int attention_extent) {
  switch (kind) {
    case TaskKind::kRMSNorm:
    case TaskKind::kQKNorm: return threads;
    case TaskKind::kEmbedding: return 1;
    case TaskKind::kAttention: return attention_extent;
    case TaskKind::kRoPE:
    case TaskKind::kKVAppend:
    case TaskKind::kElementwise:
    case TaskKind::kAdd:
    case TaskKind::kGemmCombine: return 1;
    case TaskKind::kAttentionMerge:
    case TaskKind::kArgmaxReduce: return 4;
    case TaskKind::kGemm: return 0;
  }
  return 0;
}

// The K tile is reused as the raw V staging area after QK; the V tile is
// reused as the raw K staging area before PV.
constexpr int ServingAttentionSharedBytes(int head_dim) {
  return 416 * head_dim + 6272;
}

template <TaskKind Kind, int Threads, int AttentionExtent = TILEMEGA_ATTENTION_SCRATCH_EXTENT>
struct SimtTaskResources {
  static_assert(Kind != TaskKind::kGemm, "read GEMM resources from its collective");
  static_assert(Threads>0 && AttentionExtent>0, "invalid TaskBody resource dimensions");
  using SharedStorage = float[SimtSharedElements(Kind,Threads,AttentionExtent)];
  using Traits = TaskTraits<Threads, sizeof(SharedStorage)>;
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr int kStages = 0;  // No asynchronous multistage mainloop.
  static ScalarDataflow Dataflow() { return ScalarTaskDataflow(Kind); }
};

struct TaskResourceInfo {
  int threads;
  int shared_bytes;
};

template <TaskKind Kind, int Threads>
constexpr TaskResourceInfo ReadSimtTaskResources() {
  using Traits = typename SimtTaskResources<Kind, Threads>::Traits;
  return {Traits::kThreads, int(Traits::kSharedStorageBytes)};
}

template <int Threads>
inline TaskResourceInfo ReadSimtTaskResources(TaskKind kind) {
  switch (kind) {
    case TaskKind::kRMSNorm: return ReadSimtTaskResources<TaskKind::kRMSNorm, Threads>();
    case TaskKind::kEmbedding: return ReadSimtTaskResources<TaskKind::kEmbedding, Threads>();
    case TaskKind::kQKNorm: return ReadSimtTaskResources<TaskKind::kQKNorm, Threads>();
    case TaskKind::kRoPE: return ReadSimtTaskResources<TaskKind::kRoPE, Threads>();
    case TaskKind::kKVAppend: return ReadSimtTaskResources<TaskKind::kKVAppend, Threads>();
    case TaskKind::kElementwise: return ReadSimtTaskResources<TaskKind::kElementwise, Threads>();
    case TaskKind::kAttention: return ReadSimtTaskResources<TaskKind::kAttention, Threads>();
    case TaskKind::kAdd: return ReadSimtTaskResources<TaskKind::kAdd, Threads>();
    case TaskKind::kGemmCombine: return ReadSimtTaskResources<TaskKind::kGemmCombine, Threads>();
    case TaskKind::kAttentionMerge: return ReadSimtTaskResources<TaskKind::kAttentionMerge, Threads>();
    case TaskKind::kArgmaxReduce: return ReadSimtTaskResources<TaskKind::kArgmaxReduce, Threads>();
    default: throw std::invalid_argument("TaskBody has no scalar resource declaration");
  }
}

inline TaskResourceInfo ReadSimtTaskResources(TaskKind kind, int launch_threads) {
  if (launch_threads == 128) return ReadSimtTaskResources<128>(kind);
  if (launch_threads == 256) return ReadSimtTaskResources<256>(kind);
  throw std::invalid_argument("unsupported collective launch width for scalar TaskBody");
}

}  // namespace tilemega::codegen
