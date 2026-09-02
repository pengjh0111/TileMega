// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2.4 split reduction -- the combiner of a split-K GEMM.
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>

namespace tilemega::codegen {

/// operand = {partials, out}; `width` is N, `group` is the chunk count.
/// The partials are laid out chunk-major, so chunk `c` of element `i` is at
/// `c * seq * width + i`. Summation runs in chunk order, which keeps the
/// result bitwise reproducible across runs (it is not bitwise equal to the
/// unsplit GEMM: a different association of the same sum).
template <class Arch, class SmemUnion, int Threads>
struct GemmCombineTaskBody {
  using SharedStorage = float[1];
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr bool kLegal = true;

  __device__ static TaskOwnership Ownership(Params const& p,
                                            StageDesc const& stage) {
    int count = p.dims.seq * static_cast<int>(stage.width);
    return {TaskOwnershipKind::kElementChunk,
            (count + Threads - 1) / Threads};
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion&) const {
    float const* partials = p.buffers[stage.operand[0]];
    float* out = p.buffers[stage.operand[1]];
    int count = p.dims.seq * static_cast<int>(stage.width);
    int chunks = static_cast<int>(stage.group);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
      float sum = 0.0f;
      for (int c = 0; c < chunks; ++c) sum += partials[c * count + i];
      out[i] = sum;
    }
  }
};

}  // namespace tilemega::codegen
