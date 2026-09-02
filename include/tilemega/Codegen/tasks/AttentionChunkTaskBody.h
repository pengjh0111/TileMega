// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>

namespace tilemega::codegen {

/// operand = {q_rot, full_k, full_v, context}.  One CTA per (token, head);
/// the chunk loop and the combine are fused in this body.
template <class Arch, class SmemUnion, int Threads>
struct AttentionTaskBody {
  using SharedStorage = float[Threads];
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr bool kLegal = true;

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    float const* q_rot = p.buffers[stage.operand[0]];
    float const* full_k = p.buffers[stage.operand[1]];
    float const* full_v = p.buffers[stage.operand[2]];
    float* context = p.buffers[stage.operand[3]];
    int const dim = static_cast<int>(stage.width);
    int const total = p.dims.total, past = p.dims.past;
    int const heads = static_cast<int>(stage.extent);
    int const group = static_cast<int>(stage.group);
    int const kv_heads = heads / group;

    int query = static_cast<int>(blockIdx.x);
    bool active = query < p.dims.seq * heads;
    int token = query / heads;
    int head = query % heads;
    int kv = head / (heads / kv_heads);
    if (active && threadIdx.x < total) {
      int key_pos = threadIdx.x;
      float score = -INFINITY;
      if (key_pos <= past + token) {
        score = 0.0f;
        int qbase = (token * heads + head) * dim;
        int kbase = (kv * total + key_pos) * dim;
        for (int d = 0; d < dim; ++d)
          score = fmaf(q_rot[qbase + d], full_k[kbase + d], score);
        score /= sqrtf(static_cast<float>(dim));
      }
      smem.attention[threadIdx.x] = score;
    }
    __syncthreads();
    if (active && threadIdx.x == 0) {
      float maximum = -INFINITY;
      for (int j = 0; j < total; ++j)
        maximum = fmaxf(maximum, smem.attention[j]);
      float sum = 0.0f;
      for (int j = 0; j < total; ++j) {
        float value = expf(smem.attention[j] - maximum);
        smem.attention[j] = value;
        sum += value;
      }
      for (int j = 0; j < total; ++j) smem.attention[j] /= sum;
    }
    __syncthreads();
    if (active)
      for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float value = 0.0f;
        for (int j = 0; j < total; ++j)
          value = fmaf(smem.attention[j], full_v[(kv * total + j) * dim + d],
                       value);
        context[(token * heads + head) * dim + d] = value;
      }
    __syncthreads();
  }
};

}  // namespace tilemega::codegen
