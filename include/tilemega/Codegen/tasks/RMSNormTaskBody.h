// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/PhaseTrace.cuh>
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <tilemega/Codegen/tasks/WarpReduce.cuh>

namespace tilemega::codegen {

/// operand = {input, weight, output}.  One CTA per token row.
template <class Arch, class SmemUnion, int Threads>
struct RMSNormTaskBody {
  using ResourceTraits = SimtTaskResources<TaskKind::kRMSNorm,Threads>;
  using SharedStorage = typename ResourceTraits::SharedStorage;
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr int kStages = ResourceTraits::kStages;
  static constexpr bool kLegal = true;

  /// One token per CTA: `blockIdx.x` is the task index (§2.7's `m`).
  __device__ static TaskOwnership Ownership(Params const& p,
                                            StageDesc const&) {
    return {OwnershipOf(TaskKind::kRMSNorm), p.dims.seq};
  }

#if TILEMEGA_PREFETCH_RUNTIME
  /// §5.3.1 Prefetch: the scale row. It is the body's claim about what it will
  /// read, not a claim that the row has no in-edge -- the executor checks the
  /// derived frontier before honouring it.
  __host__ __device__ static PrefetchOperand Prefetch(StageDesc const& stage) {
    return {stage.operand[1], stage.width};
  }
#endif

  __device__ static void RunTask(Params const& p, StageDesc const& stage,
                                 SmemUnion& smem, int token TILEMEGA_PHASE_ARG
                                 TILEMEGA_PREFETCH_ARG) {
    ModelElement const* input = p.buffers[stage.operand[0]];
    ModelElement const* weight = p.buffers[stage.operand[1]];
#if TILEMEGA_PREFETCH_RUNTIME
    if (prefetched != nullptr) weight = prefetched;
#endif
    ModelElement* output = p.buffers[stage.operand[2]];
    int hidden = static_cast<int>(stage.width);
    TILEMEGA_PHASE_SIMT_SETUP();
    RunRow(input + token * hidden, weight, output + token * hidden,
           hidden, smem.rms TILEMEGA_PHASE_PASS);
  }

  __device__ static void RunRow(ModelElement const* input,
                                ModelElement const* weight,
                                ModelElement* output, int hidden, float* rms TILEMEGA_PHASE_ARG) {
    float local = 0.0f;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
      float value = static_cast<float>(input[d]);
      local += value * value;
    }
    // R8 BE-4: the shared-memory tree this replaces cost log2(Threads)
    // barriers per row -- seven at 128 threads, with half the CTA idle in each
    // step. The shuffle reduction keeps the sum in FP32 and needs two.
    float const total = BlockReduce<SumOp, Threads>(local, rms);
    TILEMEGA_PHASE_STAMP(3);
    float scale = rsqrtf(total / hidden + TILEMEGA_NORM_EPSILON);
    for (int d = threadIdx.x; d < hidden; d += blockDim.x)
      output[d] = ModelElement(
          static_cast<float>(ModelElement(
              static_cast<float>(input[d]) * scale)) *
          static_cast<float>(weight[d]));
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    for (int token = PlacedBlock(); token < p.dims.seq;
         token += gridDim.x) {
      RunTask(p, stage, smem, token);
      __syncthreads();
    }
  }
};

}  // namespace tilemega::codegen
