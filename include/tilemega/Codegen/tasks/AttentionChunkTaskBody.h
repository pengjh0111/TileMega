// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/PhaseTrace.cuh>
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <tilemega/Codegen/tasks/AttentionPhasedTaskBody.h>
#include <tilemega/Codegen/tasks/WarpReduce.cuh>

namespace tilemega::codegen {

/// operand = {q_rot, full_k, full_v, context}.  One CTA per (token, head);
/// the chunk loop and the combine are fused in this body.
template <class Arch, class SmemUnion, int Threads>
struct AttentionTaskBody {
  using ResourceTraits = SimtTaskResources<TaskKind::kAttention,Threads>;
  using SharedStorage = typename ResourceTraits::SharedStorage;
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr int kStages = ResourceTraits::kStages;
  static constexpr bool kLegal = true;

  /// One (token, head) query per CTA: `blockIdx.x` is the task index.
  __device__ static TaskOwnership Ownership(Params const& p,
                                            StageDesc const& stage) {
    return {OwnershipOf(TaskKind::kAttention),
            stage.operand[7] == kNoOperand || stage.operand[7] == 0
                ? p.dims.seq * static_cast<int>(stage.extent)
                : AttentionPhaseTasks(static_cast<AttentionPhase>(stage.operand[7]),
                    p.dims.seq * static_cast<int>(stage.extent),stage.operand[6])};
  }

  __device__ static void RunTask(Params const& p, StageDesc const& stage,
                                 SmemUnion& smem, int query TILEMEGA_PHASE_ARG) {
#if TILEMEGA_CHUNKED_ATTENTION
    if (stage.operand[7] != kNoOperand && stage.operand[7] != 0) {
      AttentionPhasedTaskBody<Arch,SmemUnion,Threads>::RunTask(p,stage,smem,query);
      return;
    }
#endif
    ModelElement const* q_rot = p.buffers[stage.operand[0]];
    ModelElement const* full_k = p.buffers[stage.operand[1]];
    ModelElement const* full_v = p.buffers[stage.operand[2]];
    ModelElement* context = p.buffers[stage.operand[3]];
    int const dim = static_cast<int>(stage.width);
    int const total = p.dims.total, past = p.dims.past;
    int const heads = static_cast<int>(stage.extent);
    int const group = static_cast<int>(stage.group);
    int const kv_heads = heads / group;
    int token = query / heads;
    int head = query % heads;
    int kv = head / (heads / kv_heads);
    TILEMEGA_PHASE_SIMT_SETUP();
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
        score = static_cast<float>(
            ModelElement(score / sqrtf(static_cast<float>(dim))));
      }
      smem.attention[key_pos] = score;
    }
    TILEMEGA_PHASE_SIMT_BARRIER();
    // R8 BE-3: FlashAttention-style running statistics, computed by the whole
    // CTA. Every thread folds the keys it owns into a running max and, once
    // the max is known, into a running exponential sum; the two folds are
    // shuffle reductions, so no lane ever walks the key sequence. What is
    // *not* changed is where the values are rounded: the exported golden
    // computes softmax in FP32 and casts the probabilities to the model dtype
    // (`torch.softmax(...).to(dtype)`), so the probability is rounded here at
    // the same point it was before. Keeping p in FP32 would be textbook Flash
    // and would disagree with the reference by construction.
    float* const reduce = &smem.attention[TILEMEGA_ATTENTION_SCRATCH_EXTENT];
    float running_max = -INFINITY;
    for (int j = threadIdx.x; j < total; j += blockDim.x)
      running_max = fmaxf(running_max, smem.attention[j]);
    float const maximum =
        BlockReduce<MaxOp, Threads>(running_max, reduce);
    float running_sum = 0.0f;
    for (int j = threadIdx.x; j < total; j += blockDim.x) {
      float const value = expf(smem.attention[j] - maximum);
      smem.attention[j] = value;
      running_sum += value;
    }
    float const sum = BlockReduce<SumOp, Threads>(running_sum, reduce);
    for (int j = threadIdx.x; j < total; j += blockDim.x)
      smem.attention[j] =
          static_cast<float>(ModelElement(smem.attention[j] / sum));
    TILEMEGA_PHASE_SIMT_BARRIER();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      float value = 0.0f;
      for (int j = 0; j < total; ++j)
        value = fmaf(smem.attention[j],
                     static_cast<float>(full_v[(kv * total + j) * dim + d]),
                     value);
      context[(token * heads + head) * dim + d] = ModelElement(value);
    }
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    for (int query = PlacedBlock(); query < Ownership(p,stage).count;
         query += gridDim.x) {
      RunTask(p, stage, smem, query);
      __syncthreads();
    }
  }
};

}  // namespace tilemega::codegen
