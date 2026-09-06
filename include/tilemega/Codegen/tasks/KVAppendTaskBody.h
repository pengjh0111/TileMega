// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>

namespace tilemega::codegen {

/// operand = {appended, past, full}; `extent` is the KV head count.  The body
/// writes rows [past, past + seq) and copies the retained prefix: exactly the
/// sub-window the derived coupling guards on.
template <class Arch, class SmemUnion, int Threads>
struct KVAppendTaskBody {
  using SharedStorage = float[1];
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr bool kLegal = true;

  /// Two grid-stride loops (append, retain) over the same buffer; the CTA
  /// count is the wider of the two.
  __device__ static TaskOwnership Ownership(Params const& p,
                                            StageDesc const& stage) {
    if (p.ownership_flags & kKVTileOwnership)
      return {TaskOwnershipKind::kTilePerBlock,
              p.dims.seq * static_cast<int>(stage.extent)};
    int appended = p.dims.seq * static_cast<int>(stage.extent) *
                   static_cast<int>(stage.width);
    int retained = p.dims.past * static_cast<int>(stage.extent) *
                   static_cast<int>(stage.width);
    int widest = appended > retained ? appended : retained;
    return {OwnershipOf(TaskKind::kKVAppend),
            (widest + Threads - 1) / Threads};
  }

  __device__ static void RunTask(Params const& p, StageDesc const& stage,
                                 SmemUnion&, int task) {
    ModelElement const* source = p.buffers[stage.operand[0]];
    ModelElement const* prefix = p.buffers[stage.operand[1]];
    ModelElement* full = p.buffers[stage.operand[2]];
    int const dim = static_cast<int>(stage.width);
    int const seq = p.dims.seq, past = p.dims.past, total = p.dims.total;
    int const kv_heads = static_cast<int>(stage.extent);
    if (p.ownership_flags & kKVTileOwnership) {
      int token = task / kv_heads;
      int kv = task % kv_heads;
      for (int d = threadIdx.x; d < dim; d += blockDim.x)
        full[(kv * total + past + token) * dim + d] =
            source[(token * kv_heads + kv) * dim + d];
      return;
    }
    int const index = task * Threads + threadIdx.x;
    int appended = seq * kv_heads * dim;
    if (index < appended) {
      int d = index % dim;
      int temp = index / dim;
      int kv = temp % kv_heads;
      int token = temp / kv_heads;
      full[(kv * total + past + token) * dim + d] = source[index];
    }
    int retained = kv_heads * past * dim;
    if (index < retained) {
      int d = index % dim;
      int temp = index / dim;
      int pos = temp % past;
      int kv = temp / past;
      full[(kv * total + pos) * dim + d] = prefix[index];
    }
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    int const dim = static_cast<int>(stage.width);
    int const kv_heads = static_cast<int>(stage.extent);

    if (p.ownership_flags & kKVTileOwnership) {
      for (int task = PlacedBlock(); task < p.dims.seq * kv_heads;
           task += gridDim.x)
        RunTask(p, stage, smem, task);
      return;
    }

    int appended = p.dims.seq * kv_heads * dim;
    int retained = kv_heads * p.dims.past * dim;
    int widest = appended > retained ? appended : retained;
    int chunks = (widest + Threads - 1) / Threads;
    for (int task = PlacedBlock(); task < chunks; task += gridDim.x)
      RunTask(p, stage, smem, task);
  }
};

}  // namespace tilemega::codegen
