// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>

namespace tilemega::codegen {

/// operand = {input, output, inv_freq}; `extent` is the head count of this
/// tensor (a per-token count, so the token axis stays symbolic).
template <class Arch, class SmemUnion, int Threads>
struct RoPETaskBody {
  using SharedStorage = float[1];
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr bool kLegal = true;

  /// Grid-stride over (token, head, half-dim) pairs -- an element chunk, not
  /// a task tile.
  __device__ static TaskOwnership Ownership(Params const& p,
                                            StageDesc const& stage) {
    int pairs = p.dims.seq * static_cast<int>(stage.extent) *
                (static_cast<int>(stage.width) / 2);
    return {TaskOwnershipKind::kElementChunk,
            (pairs + Threads - 1) / Threads};
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion&) const {
    float const* input = p.buffers[stage.operand[0]];
    float* output = p.buffers[stage.operand[1]];
    float const* inv_freq = p.buffers[stage.operand[2]];
    int const dim = static_cast<int>(stage.width), half_dim = dim / 2;
    int const heads = static_cast<int>(stage.extent);
    int pairs = p.dims.seq * heads * half_dim;
    for (int index = PlacedBlock() * blockDim.x + threadIdx.x; index < pairs;
         index += gridDim.x * blockDim.x) {
      int half = index % half_dim;
      int head_token = index / half_dim;
      int token = head_token / heads;
      int base = head_token * dim;
      float angle = (p.dims.past + token) * inv_freq[half];
      float c = cosf(angle), s = sinf(angle);
      float a = input[base + half], b = input[base + half + half_dim];
      output[base + half] = a * c - b * s;
      output[base + half + half_dim] = b * c + a * s;
    }
  }
};

}  // namespace tilemega::codegen
