// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>

namespace tilemega::codegen {

/// operand = {gate, up, out}; `extent` is the per-token width.
template <class Arch, class SmemUnion, int Threads>
struct ElementwiseTaskBody {
  using SharedStorage = float[1];
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr bool kLegal = true;

  /// Grid-stride over a flat element range: CTA `b` owns elements
  /// `b*Threads + k*gridDim.x*Threads`, which depends on the launch grid,
  /// not on the task space.
  __device__ static TaskOwnership Ownership(Params const& p,
                                            StageDesc const& stage) {
    int count = p.dims.seq * static_cast<int>(stage.extent);
    return {TaskOwnershipKind::kElementChunk,
            (count + Threads - 1) / Threads};
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion&) const {
    float const* gate = p.buffers[stage.operand[0]];
    float const* up = p.buffers[stage.operand[1]];
    float* out = p.buffers[stage.operand[2]];
    int count = p.dims.seq * static_cast<int>(stage.extent);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
      float x = gate[i];
      out[i] = (x / (1.0f + expf(-x))) * up[i];
    }
  }
};

}  // namespace tilemega::codegen
