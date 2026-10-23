// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Codegen/tasks/mx/DecodeProgram.cuh
//
// The MX-4 decode program (MX4 prompt §3.2): one generated kernel per plan
// (model x B), G CTAs x 512 threads, one CTA per AP, dynamic shared memory
// only. Every step runs
//   Embed | (Qkv | Attn | O | GateUp | Down) x L | Head
// with one barrier after each stage, 5L+2 per step; each barrier carries the
// next stage's prefetch (DecodeBarrier.cuh). The last layer's D->Q prefetch
// targets layer 0 and is discarded, so Qkv is instantiated once and its
// prefetch registers ride the layer loop's back edge.
//
// The generator (lib/Codegen/MxDecodeProgram.cpp) supplies the plan type P:
// constexpr shapes and stage bodies of every class, the model constants and
// the binding structure, passed by value (§6 D2). There is no stage table,
// no descriptor and no runtime kind or variant dispatch.
#pragma once

#include <tilemega/Backend/Mx/DecodeAttention.h>
#include <tilemega/Backend/Mx/DecodeGemm.h>
#include <tilemega/Codegen/executor/Watchdog.cuh>
#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <tilemega/Codegen/tasks/mx/DecodeBarrier.cuh>
#include <tilemega/Target/Maca/Bf16.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>

#include <cstdint>

#ifndef TILEMEGA_TRACE_STAGE
#define TILEMEGA_TRACE_STAGE 0
#endif
#ifndef TILEMEGA_MX_DIAG
#define TILEMEGA_MX_DIAG 0
#endif

namespace tilemega::codegen::mx {

using backend::mx::MxAttentionOperands;
using backend::mx::MxGemmOperands;

/// Per-layer weights and caches, from the plan's external buffers.
struct MxLayerBinding {
  Bf16 const *w_qkv, *w_o, *w_gu, *w_down, *q_norm, *k_norm;
  Bf16 *k_cache, *vt_cache;
};

/// Program scratch, allocated and cleared by tm_plan_create.
struct MxScratch {
  Bf16 *x, *qkv, *ctx, *h;
  float* gemm_partial;   // [T][P][B][TN+1], shared by the GEMM classes (barrier-separated)
  float* attn_partial;   // [B*hkv][ceil(cap/32)][Q][D+2]
  float* head_value;     // [B][G]
  int* head_index;       // [B][G]
  unsigned *t_qkv, *t_o, *t_gu, *t_down, *t_attn, *t_head;
  unsigned long long* barrier;  // 5L+2 counters, 64 B apart, never reset after create
  WatchdogRecord* watchdog;     // TILEMEGA_WATCHDOG builds only
  unsigned long long watchdog_ns;
  /// TILEMEGA_MX_DIAG builds only (never measured): bit c skips the work of
  /// stage class c (0 embed, 1 qkv, 2 attn, 3 o, 4 gate_up, 5 down, 6 head)
  /// at run time, keeping the generated code; from TILEMEGA_MX_DIAG_SKIP.
  unsigned diag_skip;
};

template <int L>
struct MxBinding {
  MxLayerBinding layer[L];
  Bf16 const *embed, *w_head, *rope_cos, *rope_sin;
  int* tokens;  // [B][capacity]
  MxScratch s;
};

/// Where an external buffer binds into MxBinding (the generated buffer
/// table of DecodeRuntime.cuh).
enum MxSlot : int {
  kSlotTokens, kSlotRopeCos, kSlotRopeSin, kSlotEmbed, kSlotHead,
  kSlotQkv, kSlotO, kSlotGateUp, kSlotDown, kSlotQNorm, kSlotKNorm, kSlotKCache, kSlotVtCache
};

struct MxBufferRow {
  char const* name;
  std::uint32_t dtype;
  std::uint64_t constant, per_batch;
  char const* pack_json;
  int slot, layer;
};

/// TILEMEGA_TRACE_STAGE: per (slot, stage, CTA) three timestamps, ns:
/// start (previous barrier's wait returned), work end (before this
/// barrier's arrival), release (this barrier's wait returned). The slot is
/// the launch iteration modulo `slots` (TILEMEGA_TRACE_LAUNCHES).
struct MxTrace {
  unsigned long long* data;
  unsigned slots;
};

template <class P>
struct Program {
  using Binding = typename P::Binding;
  static constexpr int kLayers = P::kLayers, kStages = 5 * P::kLayers + 2;

  __device__ __forceinline__ static MxGemmOperands Gemm(Binding const& b, Bf16 const* a, Bf16 const* w, Bf16* out,
                                                         unsigned* tickets) {
    return MxGemmOperands{a,       w,       out,     b.s.gemm_partial, tickets, nullptr,      nullptr, nullptr,
                          nullptr, nullptr, nullptr, nullptr,          nullptr, nullptr,      nullptr, P::kCapacity,
                          0,       P::kEpsilon};
  }
  __device__ __forceinline__ static MxGemmOperands Qkv(Binding const& b, int l, int past) {
    auto const& w = b.layer[l];
    return MxGemmOperands{b.s.x,     w.w_qkv,    b.s.qkv,      b.s.gemm_partial, b.s.t_qkv, w.q_norm,
                          w.k_norm,  b.rope_cos, b.rope_sin,   w.k_cache,        w.vt_cache, nullptr,
                          nullptr,   nullptr,    nullptr,      P::kCapacity,     past,       P::kEpsilon};
  }
  __device__ __forceinline__ static MxGemmOperands O(Binding const& b, int l) {
    return Gemm(b, b.s.ctx, b.layer[l].w_o, b.s.x, b.s.t_o);
  }
  __device__ __forceinline__ static MxGemmOperands GateUp(Binding const& b, int l) {
    return Gemm(b, b.s.x, b.layer[l].w_gu, b.s.h, b.s.t_gu);
  }
  __device__ __forceinline__ static MxGemmOperands Down(Binding const& b, int l) {
    return Gemm(b, b.s.h, b.layer[l].w_down, b.s.x, b.s.t_down);
  }
  __device__ __forceinline__ static MxGemmOperands Head(Binding const& b, int past) {
    return MxGemmOperands{b.s.x,        b.w_head, nullptr,        b.s.gemm_partial, nullptr,
                          nullptr,      nullptr,  nullptr,        nullptr,          nullptr,
                          nullptr,      b.s.head_value, b.s.head_index, b.s.t_head, b.tokens,
                          P::kCapacity, past,     P::kEpsilon};
  }
  __device__ __forceinline__ static MxAttentionOperands Attn(Binding const& b, int l, int past) {
    return MxAttentionOperands{b.s.qkv,          b.layer[l].k_cache, b.layer[l].vt_cache, b.s.ctx,
                               b.s.attn_partial, b.s.t_attn,         P::kCapacity,        past};
  }

  __device__ __forceinline__ static void Mark(MxTrace const& t, unsigned long long it, unsigned stage, int which) {
#if TILEMEGA_TRACE_STAGE
    if (threadIdx.x == 0 && t.data && t.slots)
      t.data[((std::size_t(it % t.slots) * kStages + stage) * gridDim.x + blockIdx.x) * 3 + which] =
          maca::GlobalTimerNs();
#else
    (void)t, (void)it, (void)stage, (void)which;
#endif
  }
};

/// The per-step program of §3.2.
template <class P>
__global__ void __launch_bounds__(512)
    mx_decode_kernel(typename P::Binding b, int const* pasts, unsigned first, unsigned steps,
                     unsigned long long iteration, unsigned long long* step_ns, MxTrace trace) {
  extern __shared__ __align__(16) char smem[];
  using G = Program<P>;
  typename P::QkvShape const qkv_shape{};
  typename P::OShape const o_shape{};
  typename P::GateUpShape const gu_shape{};
  typename P::DownShape const down_shape{};
  typename P::HeadShape const head_shape{};
  typename P::AttnShape const attn_shape{};
  if (step_ns && blockIdx.x == 0 && threadIdx.x == 0) step_ns[0] = maca::GlobalTimerNs();
#pragma unroll 1
  for (unsigned s = 0; s < steps; ++s) {
    // This step writes KV at `past`; L = past + 1. Uniform: as a plain load it
    // counted as divergent, and every branch on the attention plan was
    // structurized into chains that kept the bodies' inputs live (MIR).
    int const past = __builtin_mxc_readfirstlane(pasts[first + s]);
    unsigned long long const it = iteration + s;
    unsigned stage = 0;
    auto barrier = [&](auto prefetch) {
      G::Mark(trace, it, stage, 1);
      auto pf = Barrier(b.s.barrier + 8 * std::size_t(stage), (unsigned long long)gridDim.x * (it + 1), prefetch,
                        MxWatch{b.s.watchdog, b.s.watchdog_ns, stage, it});
      G::Mark(trace, it, stage, 2);
      ++stage;
      G::Mark(trace, it, stage, 0);
      return pf;
    };
    // Diagnostic builds only: skip a class's work (the barriers remain).
    auto run = [&](int c) { return !TILEMEGA_MX_DIAG || !((b.s.diag_skip >> c) & 1u); };
    G::Mark(trace, it, 0, 0);
    if (run(0)) backend::mx::MxDecodeEmbed(b.embed, b.tokens, b.s.x, backend::mx::MxShape{0, 0, P::kBatch}, P::kHidden,
                               P::kCapacity, past);
    auto pf_q = barrier([&] { return P::Qkv::Prefetch(G::Qkv(b, 0, past), qkv_shape); });
#pragma unroll 1
    for (int l = 0; l < P::kLayers; ++l) {
      if (run(1)) P::Qkv::Run(pf_q, G::Qkv(b, l, past), qkv_shape, smem);
      auto pf_a = barrier([&] { return P::Attn::Prefetch(G::Attn(b, l, past), attn_shape); });
      if (run(2)) P::Attn::Run(pf_a, G::Attn(b, l, past), attn_shape, smem);
      auto pf_o = barrier([&] { return P::O::Prefetch(G::O(b, l), o_shape); });
      if (run(3)) P::O::Run(pf_o, G::O(b, l), o_shape, smem);
      auto pf_g = barrier([&] { return P::GateUp::Prefetch(G::GateUp(b, l), gu_shape); });
      if (run(4)) P::GateUp::Run(pf_g, G::GateUp(b, l), gu_shape, smem);
      auto pf_d = barrier([&] { return P::Down::Prefetch(G::Down(b, l), down_shape); });
      if (run(5)) P::Down::Run(pf_d, G::Down(b, l), down_shape, smem);
      pf_q = barrier([&] { return P::Qkv::Prefetch(G::Qkv(b, l + 1 < P::kLayers ? l + 1 : 0, past), qkv_shape); });
    }
    // The last arriver writes tokens[b][past+1].
    auto const pf_h = P::Head::Prefetch(G::Head(b, past), head_shape);
    if (run(6)) P::Head::Run(pf_h, G::Head(b, past), head_shape, smem);
    barrier([] { return NoPrefetch{}; });
    if (step_ns && blockIdx.x == 0 && threadIdx.x == 0) step_ns[s + 1] = maca::GlobalTimerNs();
  }
}

}  // namespace tilemega::codegen::mx
