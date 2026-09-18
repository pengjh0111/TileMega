// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3.  Handwritten TaskBody; every shape arrives at run time.
#pragma once
#include <tilemega/Codegen/tasks/PhaseTrace.cuh>
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/Placement.cuh>
#include <tilemega/Codegen/tasks/TaskResources.h>

namespace tilemega::codegen {

/// operand = {ids, table, output}; `width` is the hidden size.  One token row
/// per CTA, which is the same ownership the row normalizations declare, so a
/// coupling between them can still be the identity.
///
/// The identifiers keep the exported index tensor's own width
/// (`TILEMEGA_TOKEN_ID_BITS`) and occupy that many model elements: the buffer
/// table is uniformly `ModelElement**`, and widening a token id to the storage
/// type would silently alias vocabulary entries above its integer range (2^8
/// for BF16).
template <class Arch, class SmemUnion, int Threads>
struct EmbeddingTaskBody {
  using ResourceTraits = SimtTaskResources<TaskKind::kEmbedding,Threads>;
  using SharedStorage = typename ResourceTraits::SharedStorage;
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr int kNumThreads = Threads;
  static constexpr int kStages = ResourceTraits::kStages;
  static constexpr bool kLegal = true;

  using TokenIdType =
      std::conditional_t<TILEMEGA_TOKEN_ID_BITS == 64, std::int64_t, std::int32_t>;

  __device__ static long TokenId(ModelElement const* ids, int token) {
    return static_cast<long>(reinterpret_cast<TokenIdType const*>(ids)[token]);
  }

  __device__ static TaskOwnership Ownership(Params const& p,
                                            StageDesc const&) {
    return {OwnershipOf(TaskKind::kEmbedding), p.dims.seq};
  }

  __device__ static void RunTask(Params const& p, StageDesc const& stage,
                                 SmemUnion&, int token TILEMEGA_PHASE_ARG) {
    ModelElement const* ids = p.buffers[stage.operand[0]];
    ModelElement const* table = p.buffers[stage.operand[1]];
    ModelElement* output = p.buffers[stage.operand[2]];
    int const hidden = static_cast<int>(stage.width);
    TILEMEGA_PHASE_SIMT_SETUP();
    // A gather, not an arithmetic operator: the row is copied at storage
    // precision, so there is no accumulation whose precision could differ.
    ModelElement const* row = table + TokenId(ids, token) * hidden;
    TILEMEGA_PHASE_STAMP(3);
    for (int d = threadIdx.x; d < hidden; d += blockDim.x)
      output[token * hidden + d] = row[d];
  }

  __device__ void operator()(Params const& p, StageDesc const& stage,
                             SmemUnion& smem) const {
    for (int token = PlacedBlock(); token < p.dims.seq; token += gridDim.x)
      RunTask(p, stage, smem, token);
  }
};

}  // namespace tilemega::codegen
