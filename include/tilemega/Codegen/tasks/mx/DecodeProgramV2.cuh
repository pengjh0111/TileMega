// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Mx/DecodeAttentionV2.h>
#include <tilemega/Backend/Mx/DecodeHeadV2.h>
#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <tilemega/Codegen/tasks/mx/StageProgram.cuh>

#include <cstddef>
#include <cstdint>

namespace tilemega::codegen::mx {

struct MxLayerBindingV2 {
  Bf16 const *w_qkv, *w_o, *w_gu, *w_down, *q_norm, *k_norm;
  Bf16 *k_cache, *vt_cache;
};
struct MxScratchV2 {
  Bf16 *x, *qkv, *ctx, *h;
  float *gemm_partial, *attn_partial;
  backend::mx::MxHeadPartialV2* head_partial;
  unsigned *t_qkv, *t_o, *t_gu, *t_down, *t_attn, *t_head;
  // One 64-byte line per program stage, monotonic across launches/steps.
  unsigned long long* barrier;
  WatchdogRecord* watchdog;
  unsigned long long watchdog_ns;
  unsigned diag_skip;
};
template <int Layers>
struct MxBindingV2 {
  MxLayerBindingV2 layer[Layers];
  Bf16 const *embed, *w_head, *rope_cos, *rope_sin;
  int* tokens;
  MxScratchV2 s;
};
enum MxSlotV2 : int {
  kSlotTokensV2, kSlotRopeCosV2, kSlotRopeSinV2, kSlotEmbedV2, kSlotHeadV2,
  kSlotQkvV2, kSlotOV2, kSlotGateUpV2, kSlotDownV2, kSlotQNormV2, kSlotKNormV2, kSlotKCacheV2, kSlotVtCacheV2
};
struct MxBufferRowV2 {
  char const* name;
  std::uint32_t dtype;
  std::uint64_t constant, per_batch;
  char const* pack_json;
  int slot, layer;
};
struct MxTraceV2 {
  unsigned long long* data;
  unsigned slots;
};

// These accessors construct the body ABI at the call site. The layer-zero
// activation/residual address uses the same token broadcast by the barrier.
template <class P>
struct DenseOperandsV2 {
  using Binding = typename P::Binding;
  using Tokens = backend::mx::MxSharedTokensV2;
  using GemmOps = backend::mx::MxGemmOperandsV2;

  __device__ __forceinline__ static MxScratchV2 const& Scratch(Binding const& b) {
    // Keep scratch-field reloads at their stage; hoisting them spilled the 576-thread kernel.
    return (&b.s)[backend::mx::UniformOpaque(0)];
  }
  __device__ __forceinline__ static GemmOps Gemm(Binding const& b, Bf16 const* a, Bf16 const* w,
                                                Bf16 const* residual, Bf16* out, unsigned* ticket,
                                                int a_stride, int residual_stride,
                                                Tokens* a_tokens = nullptr, Tokens* residual_tokens = nullptr) {
    auto const& scratch = Scratch(b);
    return {a, w, residual, out, scratch.gemm_partial, ticket, a_tokens, residual_tokens,
            a_stride, residual_stride, P::kEpsilon};
  }
  __device__ __forceinline__ static GemmOps QkvWeights(Binding const& b, int l) {
    l = backend::mx::UniformOpaque(l);
    return Gemm(b, nullptr, b.layer[l].w_qkv, nullptr, nullptr, nullptr, P::kHidden, 0);
  }
  __device__ __forceinline__ static GemmOps Qkv(Binding const& b, int l, int past, Tokens* tokens) {
    auto const& scratch = Scratch(b);
    l = backend::mx::UniformOpaque(l);
    (void)past;
    return Gemm(b, l == 0 ? b.embed : scratch.x, b.layer[l].w_qkv, nullptr, scratch.qkv, scratch.t_qkv,
                P::kHidden, 0, l == 0 ? tokens : nullptr);
  }
  __device__ __forceinline__ static GemmOps O(Binding const& b, int l, Tokens* tokens) {
    auto const& scratch = Scratch(b);
    l = backend::mx::UniformOpaque(l);
    return Gemm(b, scratch.ctx, b.layer[l].w_o, l == 0 ? b.embed : scratch.x, scratch.x, scratch.t_o,
                P::OShape::k, P::kHidden, nullptr, l == 0 ? tokens : nullptr);
  }
  __device__ __forceinline__ static GemmOps GateUp(Binding const& b, int l) {
    auto const& scratch = Scratch(b);
    l = backend::mx::UniformOpaque(l);
    return Gemm(b, scratch.x, b.layer[l].w_gu, nullptr, scratch.h, scratch.t_gu, P::kHidden, 0);
  }
  __device__ __forceinline__ static GemmOps Down(Binding const& b, int l) {
    auto const& scratch = Scratch(b);
    l = backend::mx::UniformOpaque(l);
    return Gemm(b, scratch.h, b.layer[l].w_down, scratch.x, scratch.x, scratch.t_down, P::DownShape::k, P::kHidden);
  }
  __device__ __forceinline__ static backend::mx::MxAttentionOperandsV2 Attn(Binding const& b, int l, int past) {
    auto const& scratch = Scratch(b);
    l = backend::mx::UniformOpaque(l), past = backend::mx::UniformOpaque(past);
    auto const& layer = b.layer[l];
    return {scratch.qkv, layer.q_norm, layer.k_norm, b.rope_cos, b.rope_sin,
            layer.k_cache, layer.vt_cache, scratch.ctx, scratch.attn_partial, scratch.t_attn,
            P::kCapacity, past, P::kEpsilon};
  }
  __device__ __forceinline__ static backend::mx::MxHeadOperandsV2 Head(Binding const& b, int past) {
    auto const& scratch = Scratch(b);
    past = backend::mx::UniformOpaque(past);
    return {Gemm(b, scratch.x, b.w_head, nullptr, nullptr, nullptr, P::kHidden, 0),
            scratch.head_partial, scratch.t_head, b.tokens, P::kCapacity, past};
  }
};

template <class P>
struct DecodeControlV2 {
  using Sync = StageSyncV2<P::kThreads>;
  static constexpr int kStages = P::kStageCount;
  static constexpr int kTokenOffset = P::kBodySmemBytes;
  static constexpr int kGenerationOffset = StageAlign16V2(kTokenOffset + P::kBatch * 4);
  static constexpr int kSharedBytes = kGenerationOffset + 16;
  static_assert(P::kBodySmemBytes % 16 == 0);
  static_assert(kStages == 5 * P::kLayers + 1);
  static_assert(P::kGrid == 104);

  __device__ __forceinline__ static void Broadcast(typename P::Binding const& b, int past, int* tokens) {
    using GlobalInt = __attribute__((address_space(1))) int const;
#pragma unroll
    for (int row = 0; row < P::kBatch; ++row)
      tokens[row] = *((GlobalInt*)b.tokens + std::size_t(row) * P::kCapacity + past);
  }
  __device__ __forceinline__ static int Past(int const* pasts, unsigned index) {
    using GlobalInt = __attribute__((address_space(1))) int const;
    return __builtin_mxc_readfirstlane(*((GlobalInt*)pasts + index));
  }
  __device__ __forceinline__ static void Mark(MxTraceV2 trace, unsigned long long iteration,
                                             unsigned stage, int which) {
#if TILEMEGA_TRACE_STAGE
    if (Sync::IsRole() && trace.data && trace.slots)
      trace.data[((std::size_t(iteration % trace.slots) * kStages + stage) * gridDim.x + blockIdx.x) * 3 + which] =
          maca::GlobalTimerNs();
#else
    (void)trace, (void)iteration, (void)stage, (void)which;
#endif
  }
};

template <class P>
__global__ void __launch_bounds__(P::kThreads)
mx_decode_kernel_v2(typename P::Binding b, int const* pasts, unsigned first, unsigned steps,
                    unsigned long long iteration, unsigned long long* step_ns, MxTraceV2 trace) {
  extern __shared__ __align__(16) char smem[];
  using G = DecodeControlV2<P>;
  using Sync = typename G::Sync;
  if (!steps) return;
  auto* const tokens = reinterpret_cast<int*>(smem + G::kTokenOffset);
  auto* const shared_tokens = (backend::mx::MxSharedTokensV2*)tokens;
  auto* const generation = reinterpret_cast<unsigned*>(smem + G::kGenerationOffset);
  int const initial_past = G::Past(pasts, first);
  if (Sync::IsRole()) {
    *generation = 0;
    G::Broadcast(b, initial_past, tokens);
  }
  // The only entrance synchronization precedes every data prefetch.
  Sync::CtaSync();
  auto pf_q = P::template EntryPrefetch<P>(b, initial_past, shared_tokens);
#if TILEMEGA_MX_DIAG
  if (step_ns && blockIdx.x == 0 && Sync::IsRole()) step_ns[0] = maca::GlobalTimerNs();
#else
  (void)step_ns;
#endif
#pragma unroll 1
  for (unsigned s = 0; s < steps; ++s) {
    int const past = G::Past(pasts, first + s);
    unsigned long long const it = iteration + s;
    unsigned stage = 0;
    G::Mark(trace, it, 0, 0);
    auto barrier = [&] {
      G::Mark(trace, it, stage, 1);
      StageBarrierV2<Sync>(b.s.barrier + 8 * std::size_t(stage),
                           (unsigned long long)P::kGrid * (it + 1), generation,
                           s * G::kStages + stage + 1,
                           [&] {
                             if (stage + 1 == G::kStages && s + 1 < steps)
                               G::Broadcast(b, G::Past(pasts, first + s + 1), tokens);
                           }, StageWatchV2{b.s.watchdog, b.s.watchdog_ns, stage, it});
      G::Mark(trace, it, stage, 2);
      ++stage;
      if (stage < G::kStages) G::Mark(trace, it, stage, 0);
    };
    auto run = [&](int stage_class) {
#if TILEMEGA_MX_DIAG
      return !((b.s.diag_skip >> stage_class) & 1u);
#else
      (void)stage_class;
      return true;
#endif
    };
    P::template RunStep<P>(b, past, smem, shared_tokens, pf_q, barrier, run);
#if TILEMEGA_MX_DIAG
    if (step_ns && blockIdx.x == 0 && Sync::IsRole()) step_ns[s + 1] = maca::GlobalTimerNs();
#endif
  }
}

}  // namespace tilemega::codegen::mx
