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
    int appended = p.dims.seq * static_cast<int>(stage.extent) *
                   static_cast<int>(stage.width);
    int retained = p.dims.past * static_cast<int>(stage.extent) *
                   static_cast<int>(stage.width);
    int widest = appended > retained ? appended : retained;
    return {TaskOwnershipKind::kElementChunk,
            (widest + Threads - 1) / Threads};
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion&) const {
    float const* source = p.buffers[stage.operand[0]];
    float const* prefix = p.buffers[stage.operand[1]];
    float* full = p.buffers[stage.operand[2]];
    int const dim = static_cast<int>(stage.width);
    int const seq = p.dims.seq, past = p.dims.past, total = p.dims.total;
    int const kv_heads = static_cast<int>(stage.extent);

    int appended = seq * kv_heads * dim;
    for (int index = PlacedBlock() * blockDim.x + threadIdx.x; index < appended;
         index += gridDim.x * blockDim.x) {
      int d = index % dim;
      int temp = index / dim;
      int kv = temp % kv_heads;
      int token = temp / kv_heads;
      full[(kv * total + past + token) * dim + d] = source[index];
    }
    int retained = kv_heads * past * dim;
    for (int index = PlacedBlock() * blockDim.x + threadIdx.x; index < retained;
         index += gridDim.x * blockDim.x) {
      int d = index % dim;
      int temp = index / dim;
      int pos = temp % past;
      int kv = temp / past;
      full[(kv * total + pos) * dim + d] = prefix[index];
    }
  }
};

}  // namespace tilemega::codegen
