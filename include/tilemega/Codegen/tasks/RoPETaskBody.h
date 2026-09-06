// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>

namespace tilemega::codegen {

/// The source graph materializes an integer arange then converts it to model
/// storage. Keeping that conversion boundary here makes non-zero-past RoPE
/// independent of backend-specific low-precision arange midpoint behavior.
__device__ inline float RoPEPosition(int past, int token) {
  return static_cast<float>(ModelElement(static_cast<float>(past + token)));
}

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
    if (p.ownership_flags & kRoPETileOwnership)
      return {TaskOwnershipKind::kTilePerBlock,
              p.dims.seq * static_cast<int>(stage.extent)};
    int pairs = p.dims.seq * static_cast<int>(stage.extent) *
                (static_cast<int>(stage.width) / 2);
    return {OwnershipOf(TaskKind::kRoPE),
            (pairs + Threads - 1) / Threads};
  }

  __device__ static void RunTask(Params const& p, StageDesc const& stage,
                                 SmemUnion&, int task) {
    ModelElement const* input = p.buffers[stage.operand[0]];
    ModelElement* output = p.buffers[stage.operand[1]];
    ModelElement const* inv_freq = p.buffers[stage.operand[2]];
    int const dim = static_cast<int>(stage.width), half_dim = dim / 2;
    int const heads = static_cast<int>(stage.extent);
    if (p.ownership_flags & kRoPETileOwnership) {
      int const token = task / heads;
      int const base = task * dim;
      for (int half = threadIdx.x; half < half_dim; half += blockDim.x) {
        float position = RoPEPosition(p.dims.past, token);
        float angle = static_cast<float>(ModelElement(
            position * static_cast<float>(inv_freq[half])));
        float c = static_cast<float>(ModelElement(cosf(angle)));
        float s = static_cast<float>(ModelElement(sinf(angle)));
        float a = static_cast<float>(input[base + half]);
        float b = static_cast<float>(input[base + half + half_dim]);
        float ac = static_cast<float>(ModelElement(a * c));
        float bs = static_cast<float>(ModelElement(b * s));
        float bc = static_cast<float>(ModelElement(b * c));
        float as = static_cast<float>(ModelElement(a * s));
        output[base + half] = ModelElement(ac - bs);
        output[base + half + half_dim] = ModelElement(bc + as);
      }
      return;
    }
    int const index = task * Threads + threadIdx.x;
    int const pairs = p.dims.seq * heads * half_dim;
    if (index >= pairs) return;
    int half = index % half_dim;
    int head_token = index / half_dim;
    int token = head_token / heads;
    int base = head_token * dim;
    float position = RoPEPosition(p.dims.past, token);
    float angle = static_cast<float>(ModelElement(
        position * static_cast<float>(inv_freq[half])));
    float c = static_cast<float>(ModelElement(cosf(angle)));
    float s = static_cast<float>(ModelElement(sinf(angle)));
    float a = static_cast<float>(input[base + half]);
    float b = static_cast<float>(input[base + half + half_dim]);
    float ac = static_cast<float>(ModelElement(a * c));
    float bs = static_cast<float>(ModelElement(b * s));
    float bc = static_cast<float>(ModelElement(b * c));
    float as = static_cast<float>(ModelElement(a * s));
    output[base + half] = ModelElement(ac - bs);
    output[base + half + half_dim] = ModelElement(bc + as);
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    int const half_dim = static_cast<int>(stage.width) / 2;
    int const heads = static_cast<int>(stage.extent);
    if (p.ownership_flags & kRoPETileOwnership) {
      for (int task = PlacedBlock(); task < p.dims.seq * heads;
           task += gridDim.x)
        RunTask(p, stage, smem, task);
      return;
    }
    int pairs = p.dims.seq * heads * half_dim;
    int chunks = (pairs + Threads - 1) / Threads;
    for (int task = PlacedBlock(); task < chunks; task += gridDim.x)
      RunTask(p, stage, smem, task);
  }
};

}  // namespace tilemega::codegen
