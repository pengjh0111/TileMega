// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
#include <tilemega/Codegen/tasks/RMSNormTaskBody.h>

#ifndef TILEMEGA_FUSED_GEMM_TASK_BODY
#define TILEMEGA_FUSED_GEMM_TASK_BODY 1
#endif

namespace tilemega::codegen {

template <class Arch, int Variant, int Threads>
struct FusedGemmTaskBody {
  using V = GemmVariant<Variant>;
  union Scratch {
    typename V::Mainloop::SharedStorage gemm;
    float rms[Threads];
  };
  struct SharedStorage {
    Scratch scratch;
    ModelElement intermediate[V::kTileM * V::kTileN];
  };
  using Gemm = GemmStageTaskBody<Arch, Scratch, Threads>;
  using Norm = RMSNormTaskBody<Arch, Scratch, Threads>;
  static constexpr int kSmemBytes = sizeof(SharedStorage);
  static constexpr bool kLegal = V::Impl::kShapeLegal && Threads == V::Impl::kThreads;

  __device__ static void Produce(GemmInvocation const& invocation, int tile,
                                 SharedStorage& smem) {
    static_assert(TILEMEGA_FUSED_GEMM_TASK_BODY || Variant < 0,
                  "fused GEMM body is disabled");
    Gemm::template RunTask<Variant, true>(invocation, tile,
        reinterpret_cast<char*>(&smem.scratch.gemm), smem.intermediate);
    __syncthreads();
  }

  __device__ static void Add(GemmInvocation const& invocation, int tile,
                             ModelElement const* residual, ModelElement* output,
                             SharedStorage& smem) {
    Produce(invocation, tile, smem);
    int m0 = (tile / invocation.tiles_n) * V::kTileM;
    int n0 = (tile % invocation.tiles_n) * V::kTileN;
    int m_extent = cute::get<0>(invocation.problem);
    int n_extent = cute::get<1>(invocation.problem);
    for (int i = threadIdx.x; i < V::kTileM * V::kTileN; i += Threads) {
      int m = m0 + i / V::kTileN, n = n0 + i % V::kTileN;
      if (m < m_extent && n < n_extent)
        output[m * n_extent + n] = ModelElement(
            float(smem.intermediate[i]) + float(residual[m * n_extent + n]));
    }
  }

  // The consumer indexes rows. Each row recomputes its entire producer tile;
  // this fanout cost must remain visible to the solver, not hidden in a loop.
  __device__ static void RMSNorm(GemmInvocation const& invocation, int row,
                                 ModelElement const* weight, ModelElement* output,
                                 SharedStorage& smem) {
    int width = cute::get<1>(invocation.problem);
    if (invocation.tiles_n != 1 || width > V::kTileN) {
      asm volatile("trap;"); return;
    }
    Produce(invocation, row / V::kTileM, smem);
    Norm::RunRow(smem.intermediate + (row % V::kTileM) * V::kTileN,
                 weight, output + row * width, width, smem.scratch.rms);
  }
};

template <class Arch, int Threads, int Variant = 0>
constexpr std::size_t FusedGemmStorageBytes() {
  std::size_t bytes = sizeof(typename FusedGemmTaskBody<Arch,Variant,Threads>::SharedStorage);
  if constexpr (Variant + 1 < kGemmVariantCount) {
    auto next = FusedGemmStorageBytes<Arch,Threads,Variant+1>();
    if (next > bytes) bytes = next;
  }
  return bytes;
}

/// operand = {consumer source/weight, final output}; gemm names the producer.
template <class Arch, class SmemUnion, int Threads>
struct FusedGemmStageTaskBody {
  static constexpr bool kLegal = Threads == kGemmThreads;
  __device__ static TaskOwnership Ownership(Params const& p, StageDesc const& stage) {
    auto const& invocation = static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
    return {TaskOwnershipKind::kTilePerBlock, stage.kind == TaskKind::kGemmRMSNorm
        ? p.dims.seq : invocation.tiles_m * invocation.tiles_n};
  }
  template <int Variant = 0>
  __device__ static void Dispatch(GemmInvocation const& invocation,
      StageDesc const& stage, Params const& p, SmemUnion& smem, int task) {
    if (invocation.variant == Variant) {
      using Body = FusedGemmTaskBody<Arch,Variant,Threads>;
      auto& storage = *reinterpret_cast<typename Body::SharedStorage*>(&smem.fused_gemm);
      if (stage.kind == TaskKind::kGemmAdd)
        Body::Add(invocation,task,p.buffers[stage.operand[0]],p.buffers[stage.operand[1]],storage);
      else
        Body::RMSNorm(invocation,task,p.buffers[stage.operand[0]],p.buffers[stage.operand[1]],storage);
    } else if constexpr (Variant+1 < kGemmVariantCount) {
      Dispatch<Variant+1>(invocation,stage,p,smem,task);
    } else { asm volatile("trap;"); }
  }
  __device__ static void RunTask(Params const& p, StageDesc const& stage,
                                 SmemUnion& smem, int task) {
    Dispatch(static_cast<GemmInvocation const*>(p.gemms)[stage.gemm],stage,p,smem,task);
  }
  __device__ void operator()(Params const& p, StageDesc const& stage, SmemUnion& smem) const {
    for (int task=PlacedBlock();task<Ownership(p,stage).count;task+=gridDim.x) {
      RunTask(p,stage,smem,task);
      __syncthreads();
    }
  }
};

}  // namespace tilemega::codegen
