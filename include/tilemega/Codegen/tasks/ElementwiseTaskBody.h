// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>

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
    if (p.ownership_flags & kActivationTileOwnership)
      return {TaskOwnershipKind::kTilePerBlock, p.dims.seq};
    int count = p.dims.seq * static_cast<int>(stage.extent);
    return {OwnershipOf(TaskKind::kElementwise),
            (count + Threads - 1) / Threads};
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion&) const {
    ModelElement const* gate = p.buffers[stage.operand[0]];
    ModelElement const* up = p.buffers[stage.operand[1]];
    ModelElement* out = p.buffers[stage.operand[2]];
    int count = p.dims.seq * static_cast<int>(stage.extent);
    if (p.ownership_flags & kActivationTileOwnership) {
      int const width = static_cast<int>(stage.extent);
      for (int token = PlacedBlock(); token < p.dims.seq;
           token += gridDim.x)
        for (int d = threadIdx.x; d < width; d += blockDim.x) {
          int i = token * width + d;
          float x = static_cast<float>(gate[i]);
          float silu = static_cast<float>(
              ModelElement(x / (1.0f + expf(-x))));
          out[i] = ModelElement(silu * static_cast<float>(up[i]));
        }
      return;
    }
    for (int i = PlacedBlock() * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
      float x = static_cast<float>(gate[i]);
      float silu = static_cast<float>(
          ModelElement(x / (1.0f + expf(-x))));
      out[i] = ModelElement(silu * static_cast<float>(up[i]));
    }
  }
};

}  // namespace tilemega::codegen
