// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2.4 split reduction -- the combiner of a split-K GEMM.
#pragma once
#include <tilemega/Codegen/tasks/PhaseTrace.cuh>
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
#include <tilemega/Codegen/tasks/ServingGemmCombineTaskBody.h>
#include <tilemega/Codegen/tasks/Placement.cuh>
#include <tilemega/Codegen/tasks/TaskResources.h>

namespace tilemega::codegen {

/// operand = {partials, out}; `width` is N, `group` is the chunk count.
/// The partials are laid out chunk-major, so chunk `c` of element `i` is at
/// `c * seq * width + i`. Summation runs in chunk order, which keeps the
/// result bitwise reproducible across runs (it is not bitwise equal to the
/// unsplit GEMM: a different association of the same sum).
template <class Arch, class SmemUnion, int Threads>
struct GemmCombineTaskBody {
  using ResourceTraits = SimtTaskResources<TaskKind::kGemmCombine,Threads>;
  using SharedStorage = typename ResourceTraits::SharedStorage;
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr int kStages = ResourceTraits::kStages;
  static constexpr bool kLegal = true;

#if TILEMEGA_SERVING_RUNTIME
  template <int Variant, backend::ServingEpilogueOp Op>
  __device__ static void RunServingOp(Params const& p,
                                      StageDesc const& stage,
                                      SmemUnion& smem, int task) {
    using V = GemmVariant<Variant>;
    auto const& invocation =
        static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
    int rows = stage.batch_rows ? p.dims.batch : p.dims.tokens();
    int columns = static_cast<int>(stage.width);
    ServingGemmCombineTaskBody<V::kTileM, V::kTileN, Op>::Run(
        reinterpret_cast<float const*>(p.buffers[stage.operand[0]]),
        invocation.chunks, task / invocation.tiles_n,
        task % invocation.tiles_n, rows, columns, columns,
        invocation.serving_output_stride,
        reinterpret_cast<cutlass::bfloat16_t*>(p.buffers[stage.operand[1]]),
        reinterpret_cast<cutlass::bfloat16_t const*>(invocation.residual),
        reinterpret_cast<float*>(p.buffers[stage.operand[1]]),
        invocation.serving_argmax_index,
        reinterpret_cast<float*>(&smem.gemm));
  }

  template <int Variant>
  __device__ static void RunServingVariant(Params const& p,
                                           StageDesc const& stage,
                                           SmemUnion& smem, int task) {
    auto const& invocation =
        static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
    switch (invocation.serving_op) {
      case backend::ServingEpilogueOp::kStore:
        RunServingOp<Variant, backend::ServingEpilogueOp::kStore>(p, stage, smem, task); break;
      case backend::ServingEpilogueOp::kResidual:
        RunServingOp<Variant, backend::ServingEpilogueOp::kResidual>(p, stage, smem, task); break;
      case backend::ServingEpilogueOp::kSwiGLU:
        RunServingOp<Variant, backend::ServingEpilogueOp::kSwiGLU>(p, stage, smem, task); break;
      case backend::ServingEpilogueOp::kArgmaxPartial:
        RunServingOp<Variant, backend::ServingEpilogueOp::kArgmaxPartial>(p, stage, smem, task); break;
      default: asm volatile("trap;"); break;
    }
  }

  template <int Variant = 0>
  __device__ static void DispatchServing(Params const& p,
                                          StageDesc const& stage,
                                          SmemUnion& smem, int task) {
    auto const& invocation =
        static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
    if (invocation.variant == Variant) {
      RunServingVariant<Variant>(p, stage, smem, task);
    } else if constexpr (Variant + 1 < TILEMEGA_GEMM_VARIANT_COUNT) {
      DispatchServing<Variant + 1>(p, stage, smem, task);
    } else {
      asm volatile("trap;");
    }
  }
#endif

  /// The combined sum, with an element near a BF16 rounding boundary settled
  /// against the whole dot product rather than against the split association.
  __device__ static float Refine(Params const& p, StageDesc const& stage,
                                 int row, int col, float sum) {
#if TILEMEGA_MIDPOINT_REFINE
    auto const& invocation =
        static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
    return RefinedGemmElement(invocation, row, col, sum);
#else
    (void)p; (void)stage; (void)row; (void)col;
    return sum;
#endif
  }

  __device__ static ModelElement Finish(float sum, Params const& p,
                                        StageDesc const& stage, int index) {
#if TILEMEGA_FP32_PARTIALS && TILEMEGA_MODEL_BF16
    // Preserve the exported Linear -> BF16 -> residual-add boundary, but
    // round only the complete GEMM sum, never each partial or partial+residual.
    auto const& invocation = static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
    ModelElement rounded(sum);
    return invocation.residual_beta == 0 ? rounded : ModelElement(
        static_cast<float>(rounded) + invocation.residual_beta *
        static_cast<float>(invocation.residual[index]));
#else
    return ModelElement(sum);
#endif
  }

  __device__ static TaskOwnership Ownership(Params const& p,
                                            StageDesc const& stage) {
    if (p.ownership_flags & kCombinerTileOwnership) {
      auto const& invocation =
          static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
      return {TaskOwnershipKind::kTilePerBlock,
              invocation.tiles_m * invocation.tiles_n};
    }
    int count = p.dims.tokens() * static_cast<int>(stage.width);
    return {OwnershipOf(TaskKind::kGemmCombine),
            (count + Threads - 1) / Threads};
  }

  __device__ static void RunTask(Params const& p, StageDesc const& stage,
                                 SmemUnion& smem, int task TILEMEGA_PHASE_ARG) {
#if TILEMEGA_SERVING_RUNTIME
    DispatchServing(p, stage, smem, task);
    return;
#endif
    auto const* partials = reinterpret_cast<ModelPartialElement const*>(p.buffers[stage.operand[0]]);
    ModelElement* out = p.buffers[stage.operand[1]];
    int count = p.dims.tokens() * static_cast<int>(stage.width);
    int chunks = static_cast<int>(stage.group);
    TILEMEGA_PHASE_SIMT_SETUP();
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
        if (row >= p.dims.tokens() || col >= static_cast<int>(stage.width))
          continue;
        int i = row * static_cast<int>(stage.width) + col;
        float sum = 0.0f;
        for (int c = 0; c < chunks; ++c)
          sum += static_cast<float>(partials[c * count + i]);
        out[i] = Finish(Refine(p, stage, row, col, sum), p, stage, i);
      }
      return;
    }
    int const i = task * Threads + threadIdx.x;
    if (i >= count) return;
    float sum = 0.0f;
    for (int c = 0; c < chunks; ++c)
      sum += static_cast<float>(partials[c * count + i]);
    out[i] = Finish(Refine(p, stage, i / static_cast<int>(stage.width),
                           i % static_cast<int>(stage.width), sum),
                    p, stage, i);
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    int count = p.dims.tokens() * static_cast<int>(stage.width);
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
