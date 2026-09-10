// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>
#include <tilemega/Codegen/tasks/TaskResources.h>

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

  __device__ static void RunTask(Params const& p, StageDesc const& stage,
                                 SmemUnion& smem, int token) {
    ModelElement const* input = p.buffers[stage.operand[0]];
    ModelElement const* weight = p.buffers[stage.operand[1]];
    ModelElement* output = p.buffers[stage.operand[2]];
    int hidden = static_cast<int>(stage.width);
    RunRow(input + token * hidden, weight, output + token * hidden,
           hidden, smem.rms);
  }

  __device__ static void RunRow(ModelElement const* input,
                                ModelElement const* weight,
                                ModelElement* output, int hidden, float* rms) {
    float local = 0.0f;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
      float value = static_cast<float>(input[d]);
      local += value * value;
    }
    rms[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset /= 2) {
      if (threadIdx.x < offset)
        rms[threadIdx.x] += rms[threadIdx.x + offset];
      __syncthreads();
    }
    float scale = rsqrtf(rms[0] / hidden + 1.0e-6f);
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
