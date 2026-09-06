// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2.4 split reduction -- the combiner of a split-K GEMM.
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
#include <tilemega/Codegen/tasks/Placement.cuh>

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
    if (p.ownership_flags & kCombinerTileOwnership) {
      auto const& invocation =
          static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
      return {TaskOwnershipKind::kTilePerBlock,
              invocation.tiles_m * invocation.tiles_n};
    }
    int count = p.dims.seq * static_cast<int>(stage.width);
    return {OwnershipOf(TaskKind::kGemmCombine),
            (count + Threads - 1) / Threads};
  }

  __device__ static void RunTask(Params const& p, StageDesc const& stage,
                                 SmemUnion&, int task) {
    ModelElement const* partials = p.buffers[stage.operand[0]];
    ModelElement* out = p.buffers[stage.operand[1]];
    int count = p.dims.seq * static_cast<int>(stage.width);
    int chunks = static_cast<int>(stage.group);
    if (p.ownership_flags & kCombinerTileOwnership) {
      auto const& invocation =
          static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
      int const tile_elements = invocation.tile_m * invocation.tile_n;
      int tile_m = task / invocation.tiles_n;
      int tile_n = task % invocation.tiles_n;
      for (int local = threadIdx.x; local < tile_elements;
           local += blockDim.x) {
        int row = tile_m * invocation.tile_m + local / invocation.tile_n;
        int col = tile_n * invocation.tile_n + local % invocation.tile_n;
        if (row >= p.dims.seq || col >= static_cast<int>(stage.width))
          continue;
        int i = row * static_cast<int>(stage.width) + col;
        float sum = 0.0f;
        for (int c = 0; c < chunks; ++c)
          sum += static_cast<float>(partials[c * count + i]);
        out[i] = ModelElement(sum);
      }
      return;
    }
    int const i = task * Threads + threadIdx.x;
    if (i >= count) return;
    float sum = 0.0f;
    for (int c = 0; c < chunks; ++c)
      sum += static_cast<float>(partials[c * count + i]);
    out[i] = ModelElement(sum);
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    int count = p.dims.seq * static_cast<int>(stage.width);
    if (p.ownership_flags & kCombinerTileOwnership) {
      auto const& invocation =
          static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
      int const tiles = invocation.tiles_m * invocation.tiles_n;
      for (int task = PlacedBlock(); task < tiles; task += gridDim.x)
        RunTask(p, stage, smem, task);
      return;
    }
    int tasks = (count + Threads - 1) / Threads;
    for (int task = PlacedBlock(); task < tasks; task += gridDim.x)
      RunTask(p, stage, smem, task);
  }
};

}  // namespace tilemega::codegen
