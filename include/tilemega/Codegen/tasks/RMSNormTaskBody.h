// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>

namespace tilemega::codegen {

/// operand = {input, weight, output}.  One CTA per token row.
template <class Arch, class SmemUnion, int Threads>
struct RMSNormTaskBody {
  using SharedStorage = float[Threads];
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
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
    float local = 0.0f;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
      float value = static_cast<float>(input[token * hidden + d]);
      local += value * value;
    }
    smem.rms[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset /= 2) {
      if (threadIdx.x < offset)
        smem.rms[threadIdx.x] += smem.rms[threadIdx.x + offset];
      __syncthreads();
    }
    float scale = rsqrtf(smem.rms[0] / hidden + 1.0e-6f);
    for (int d = threadIdx.x; d < hidden; d += blockDim.x)
      output[token * hidden + d] = ModelElement(
          static_cast<float>(ModelElement(
              static_cast<float>(input[token * hidden + d]) * scale)) *
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
