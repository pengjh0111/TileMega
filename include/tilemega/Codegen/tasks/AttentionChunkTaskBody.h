// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>

namespace tilemega::codegen {

#ifndef TILEMEGA_ATTENTION_MAX_TOTAL
#define TILEMEGA_ATTENTION_MAX_TOTAL 4096
#endif

/// operand = {q_rot, full_k, full_v, context}.  One CTA per (token, head);
/// the chunk loop and the combine are fused in this body.
template <class Arch, class SmemUnion, int Threads>
struct AttentionTaskBody {
  using SharedStorage = float[TILEMEGA_ATTENTION_MAX_TOTAL];
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr bool kLegal = true;

  /// One (token, head) query per CTA: `blockIdx.x` is the task index.
  __device__ static TaskOwnership Ownership(Params const& p,
                                            StageDesc const& stage) {
    return {OwnershipOf(TaskKind::kAttention),
            p.dims.seq * static_cast<int>(stage.extent)};
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    ModelElement const* q_rot = p.buffers[stage.operand[0]];
    ModelElement const* full_k = p.buffers[stage.operand[1]];
    ModelElement const* full_v = p.buffers[stage.operand[2]];
    ModelElement* context = p.buffers[stage.operand[3]];
    int const dim = static_cast<int>(stage.width);
    int const total = p.dims.total, past = p.dims.past;
    int const heads = static_cast<int>(stage.extent);
    int const group = static_cast<int>(stage.group);
    int const kv_heads = heads / group;

    // One resident CTA owns query b, b+grid, ... . The previous single-query
    // body silently left context rows unwritten as soon as seq*heads exceeded
    // the persistent grid; SEQSCAN is intentionally large enough to catch it.
    for (int query = PlacedBlock(); query < p.dims.seq * heads;
         query += gridDim.x) {
      int token = query / heads;
      int head = query % heads;
      int kv = head / (heads / kv_heads);
      for (int key_pos = threadIdx.x; key_pos < total;
           key_pos += blockDim.x) {
        float score = -INFINITY;
        if (key_pos <= past + token) {
          score = 0.0f;
          int qbase = (token * heads + head) * dim;
          int kbase = (kv * total + key_pos) * dim;
          for (int d = 0; d < dim; ++d)
            score = fmaf(static_cast<float>(q_rot[qbase + d]),
                         static_cast<float>(full_k[kbase + d]), score);
          // torch.matmul materializes a BF16 score before the explicit
          // score.float() softmax in the source graph.
          score = static_cast<float>(
              ModelElement(score / sqrtf(static_cast<float>(dim))));
        }
        smem.attention[key_pos] = score;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        float maximum = -INFINITY;
        for (int j = 0; j < total; ++j)
          maximum = fmaxf(maximum, smem.attention[j]);
        float sum = 0.0f;
        for (int j = 0; j < total; ++j) {
          float value = expf(smem.attention[j] - maximum);
          smem.attention[j] = value;
          sum += value;
        }
        for (int j = 0; j < total; ++j)
          smem.attention[j] = static_cast<float>(
              ModelElement(smem.attention[j] / sum));
      }
      __syncthreads();
      for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float value = 0.0f;
        for (int j = 0; j < total; ++j)
          value = fmaf(smem.attention[j],
                       static_cast<float>(full_v[(kv * total + j) * dim + d]),
                       value);
        context[(token * heads + head) * dim + d] = ModelElement(value);
      }
      __syncthreads();
    }
  }
};

}  // namespace tilemega::codegen
