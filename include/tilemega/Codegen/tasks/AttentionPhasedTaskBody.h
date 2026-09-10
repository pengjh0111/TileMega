// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/AttentionPlan.h>
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/AttentionCombineTaskBody.h>

namespace tilemega::codegen {
// operand[4/5] are FP32 scores/partials, [6] the selected chunk count and
// [7] AttentionPhase. Keeping normalization separate preserves both BF16
// rounding boundaries (QK scores and normalized probabilities) of direct
// attention; an online (max,sum,PV) merge would erase the second boundary.
template <class Arch, class SmemUnion, int Threads>
struct AttentionPhasedTaskBody {
  __device__ static void RunTask(Params const& p, StageDesc const& stage,
      SmemUnion& smem, int task) {
    auto phase = static_cast<AttentionPhase>(stage.operand[7]);
    int chunks = static_cast<int>(stage.operand[6]);
    bool partitioned = phase == AttentionPhase::kScores || phase == AttentionPhase::kPartialValue;
    int query = partitioned ? task / chunks : task;
    int chunk = partitioned ? task % chunks : 0;
    int width = static_cast<int>(stage.width), heads = static_cast<int>(stage.extent);
    int token = query / heads, head = query % heads;
    int kv = head / static_cast<int>(stage.group), total = p.dims.total;
    int begin = AttentionChunkBegin(total, chunk, chunks);
    int end = AttentionChunkBegin(total, chunk + 1, chunks);
    auto* scores = reinterpret_cast<float*>(p.buffers[stage.operand[4]]) +
                   static_cast<std::size_t>(query) * total;
    auto* partials = reinterpret_cast<float*>(p.buffers[stage.operand[5]]);
    if (phase == AttentionPhase::kScores) {
      auto* q = p.buffers[stage.operand[0]];
      auto* k = p.buffers[stage.operand[1]];
      for (int pos = begin + threadIdx.x; pos < end; pos += blockDim.x) {
        float score = -INFINITY;
        if (pos <= p.dims.past + token) {
          score = 0.0f;
          for (int d = 0; d < width; ++d)
            score = fmaf(static_cast<float>(q[query * width + d]),
                static_cast<float>(k[(kv * total + pos) * width + d]), score);
          score = static_cast<float>(ModelElement(score / sqrtf(static_cast<float>(width))));
        }
        smem.attention[pos - begin] = score;
      }
      __syncthreads();
      for (int pos = begin + threadIdx.x; pos < end; pos += blockDim.x)
        scores[pos] = smem.attention[pos - begin];
    } else if (phase == AttentionPhase::kNormalize) {
      if (threadIdx.x != 0) return;
      float maximum = -INFINITY;
      for (int pos = 0; pos < total; ++pos) maximum = fmaxf(maximum, scores[pos]);
      float sum = 0.0f;
      for (int pos = 0; pos < total; ++pos) {
        float value = expf(scores[pos] - maximum);
        scores[pos] = value;
        sum += value;
      }
      for (int pos = 0; pos < total; ++pos)
        scores[pos] = static_cast<float>(ModelElement(scores[pos] / sum));
    } else if (phase == AttentionPhase::kPartialValue) {
      auto* v = p.buffers[stage.operand[2]];
      for (int d = threadIdx.x; d < width; d += blockDim.x) {
        float value = 0.0f;
        for (int pos = begin; pos < end; ++pos)
          value = fmaf(scores[pos], static_cast<float>(v[(kv * total + pos) * width + d]), value);
        partials[(static_cast<std::size_t>(query) * chunks + chunk) * width + d] = value;
      }
    } else if (phase == AttentionPhase::kCombine) {
      AttentionCombineTaskBody<Arch, void, Threads>::Reduce(partials,
          p.buffers[stage.operand[3]], query, chunks, width);
    } else {
      asm volatile("trap;");
    }
  }
};
}  // namespace tilemega::codegen
