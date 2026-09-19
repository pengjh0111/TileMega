// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/PhaseTrace.cuh>
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>
#include <tilemega/Codegen/tasks/RMSNormTaskBody.h>
#include <tilemega/Codegen/tasks/TaskResources.h>

namespace tilemega::codegen {

/// operand = {input, weight, output}; `extent` is the head count and `width`
/// the head dimension.
///
/// The per-head query and key normalization owns a different unit from the row
/// normalization beside it: one (token, head) rather than one token. The
/// arithmetic is the same -- it calls the row body -- but the task space is
/// `seq * heads`, so the two cannot share an ownership declaration even though
/// they share an epsilon and a reduction.
template <class Arch, class SmemUnion, int Threads>
struct QKNormTaskBody {
  using ResourceTraits = SimtTaskResources<TaskKind::kQKNorm,Threads>;
  using SharedStorage = typename ResourceTraits::SharedStorage;
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr int kStages = ResourceTraits::kStages;
  static constexpr bool kLegal = true;

  __device__ static TaskOwnership Ownership(Params const& p,
                                            StageDesc const& stage) {
    return {OwnershipOf(TaskKind::kQKNorm),
            p.dims.seq * static_cast<int>(stage.extent)};
  }

#if TILEMEGA_PREFETCH_RUNTIME
  /// §5.3.1 Prefetch: the per-head scale row, read once per head.
  __host__ __device__ static PrefetchOperand Prefetch(StageDesc const& stage) {
    return {stage.operand[1], stage.width};
  }
#endif

  __device__ static void RunTask(Params const& p, StageDesc const& stage,
                                 SmemUnion& smem, int task TILEMEGA_PHASE_ARG
                                 TILEMEGA_PREFETCH_ARG) {
    ModelElement const* input = p.buffers[stage.operand[0]];
    ModelElement const* weight = p.buffers[stage.operand[1]];
#if TILEMEGA_PREFETCH_RUNTIME
    if (prefetched != nullptr) weight = prefetched;
#endif
    ModelElement* output = p.buffers[stage.operand[2]];
    int const head_dim = static_cast<int>(stage.width);
    TILEMEGA_PHASE_SIMT_SETUP();
    // `task` is the flattened (token, head): the tensor is stored token-major
    // with the heads contiguous inside a row, so one head is one contiguous
    // slice and the row body applies unchanged.
    RMSNormTaskBody<Arch, SmemUnion, Threads>::RunRow(
        input + task * head_dim, weight, output + task * head_dim, head_dim,
        smem.rms TILEMEGA_PHASE_PASS);
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    int const tasks = p.dims.seq * static_cast<int>(stage.extent);
    for (int task = PlacedBlock(); task < tasks; task += gridDim.x) {
      RunTask(p, stage, smem, task);
      __syncthreads();
    }
  }
};

}  // namespace tilemega::codegen
