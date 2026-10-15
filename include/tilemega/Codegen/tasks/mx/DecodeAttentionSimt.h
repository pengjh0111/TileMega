// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Codegen/tasks/mx/DecodeAttentionSimt.h
//
// xcore1000 decode attention (`attention_impl=simt`, one token per request):
// one CTA owns (batch, KV group, KV block). Each 32-lane logical warp scores
// tokens with D/8 lanes per token and eight dimensions per lane, straight
// from the global KV cache: kUnroll tokens per lane group are loaded before
// their arithmetic so >= 8 16-byte loads are in flight (P-10). Online softmax
// runs in the exp2 domain with the same scale as the NVIDIA bodies; the four
// warp partials are merged in a fixed order (C-2).
//
// The current token's K (q/k-norm + RoPE) and V are produced here, written to
// the cache at `past`, and read back from shared memory by this CTA.
#pragma once

#include <tilemega/Codegen/tasks/mx/AttentionCommon.h>

namespace tilemega::codegen::mx {

template <int D, int Q, bool QkNorm>
struct DecodeAttentionSimt {
  static_assert((D == 64 || D == 128) && Q >= 1 && Q <= 8, "simt decode attention: D in {64,128}, Q <= 8");
  using Element = tilemega::Bf16;
  static constexpr int kWarps = kComputeThreads / 32;
  static constexpr int kLanes = D / 8, kGroups = 32 / kLanes, kUnroll = 4;

  struct SharedStorage {
    alignas(16) Element query[Q * D];
    alignas(16) Element new_key[D];
    alignas(16) Element new_value[D];
    alignas(16) float partial[kWarps * Q * D];
    float lse[kWarps][Q];
  };

  __device__ static int Begin(ServingAttentionOperands const& p, int c) { return c * p.block_extent; }
  __device__ static int End(ServingAttentionOperands const& p, int c) {
    return min((c + 1) * p.block_extent, p.past + 1);
  }

  __device__ static void Run(ServingAttentionOperands const& p, SharedStorage& s, int b, int g, int c) {
    int const begin = Begin(p, c), end = End(p, c);
    if (begin >= end) return;
    int const warp = ComputeThread() / 32, lane = ComputeThread() % 32;
    constexpr int width = (Q + 2) * D;
    Element const* x = p.qkv + (std::size_t(b) * p.heads_kv + g) * width;
    std::size_t const head = (std::size_t(b) * p.heads_kv + g) * p.capacity;
    bool const owns_new = p.past >= begin && p.past < end;

    // Query rows by warp; the new K row by the last warp.
    for (int row = warp; row < Q; row += kWarps)
      RotateRow<D, QkNorm>(x + row * D, p.q_norm, p.cosine + std::size_t(p.past) * D,
                           p.sine + std::size_t(p.past) * D, p.epsilon,
                           [&](int d, uint4 v) { *reinterpret_cast<uint4*>(s.query + row * D + d) = v; });
    if (owns_new && warp == kWarps - 1) {
      RotateRow<D, QkNorm>(x + Q * D, p.k_norm, p.cosine + std::size_t(p.past) * D,
                           p.sine + std::size_t(p.past) * D, p.epsilon, [&](int d, uint4 v) {
                             *reinterpret_cast<uint4*>(s.new_key + d) = v;
                             backend::StoreGlobal16(p.key_cache + (head + p.past) * D + d, v);
                           });
      int const d = lane * 8;
      if (d < D) {
        uint4 v = backend::LoadGlobal16(x + (Q + 1) * D + d);
        *reinterpret_cast<uint4*>(s.new_value + d) = v;
        backend::StoreGlobal16(p.value_cache + (head + p.past) * D + d, v);
      }
    }
    ComputeSync();

    int const group = lane / kLanes, column = (lane % kLanes) * 8;
    float query[Q][8], acc[Q][8], maximum[Q], sum[Q];
    float const scale = kLog2e / sqrtf(float(D));
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      uint4 raw = *reinterpret_cast<uint4 const*>(s.query + q * D + column);
      auto const* values = reinterpret_cast<Element const*>(&raw);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        query[q][i] = float(values[i]) * scale;
        acc[q][i] = 0;
      }
      maximum[q] = -INFINITY;
      sum[q] = 0;
    }

    // Token t of this block goes to warp (t / kGroups) % kWarps, lane group
    // t % kGroups; a step covers kWarps * kGroups * kUnroll tokens.
    constexpr int kStep = kWarps * kGroups;
    for (int first = begin; first < end; first += kStep * kUnroll) {
      uint4 key[kUnroll], value[kUnroll];
      bool active[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        int const position = first + u * kStep + warp * kGroups + group;
        active[u] = position < end;
        key[u] = value[u] = uint4{0, 0, 0, 0};
        if (active[u]) {
          if (position == p.past) {
            key[u] = *reinterpret_cast<uint4 const*>(s.new_key + column);
            value[u] = *reinterpret_cast<uint4 const*>(s.new_value + column);
          } else {
            key[u] = backend::LoadGlobal16(p.key_cache + (head + position) * D + column);
            value[u] = backend::LoadGlobal16(p.value_cache + (head + position) * D + column);
          }
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        auto const* k = reinterpret_cast<Element const*>(&key[u]);
        auto const* v = reinterpret_cast<Element const*>(&value[u]);
#pragma unroll
        for (int q = 0; q < Q; ++q) {
          float score = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) score += query[q][i] * float(k[i]);
#pragma unroll
          for (int delta = kLanes / 2; delta > 0; delta /= 2)
            score += __shfl_xor_sync(maca::kFullWaveMask, score, delta, 32);
          score = active[u] ? score : -INFINITY;
          float const next = fmaxf(maximum[q], score);
          float const alpha = isfinite(maximum[q]) ? exp2f(maximum[q] - next) : 0;
          float const weight = active[u] ? exp2f(score - next) : 0;
          sum[q] = sum[q] * alpha + weight;
          maximum[q] = next;
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[q][i] = acc[q][i] * alpha + weight * float(v[i]);
        }
      }
    }

    // Merge the lane groups of this warp, then the warps.
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      float m = maximum[q];
#pragma unroll
      for (int delta = kLanes; delta < 32; delta *= 2)
        m = fmaxf(m, __shfl_xor_sync(maca::kFullWaveMask, m, delta, 32));
      float const weight = isfinite(maximum[q]) ? exp2f(maximum[q] - m) : 0;
      float normalizer = weight * sum[q];
#pragma unroll
      for (int delta = kLanes; delta < 32; delta *= 2)
        normalizer += __shfl_xor_sync(maca::kFullWaveMask, normalizer, delta, 32);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float value = weight * acc[q][i];
#pragma unroll
        for (int delta = kLanes; delta < 32; delta *= 2)
          value += __shfl_xor_sync(maca::kFullWaveMask, value, delta, 32);
        if (lane < kLanes) s.partial[(warp * Q + q) * D + column + i] = normalizer > 0 ? value / normalizer : 0;
      }
      if (lane == 0) s.lse[warp][q] = normalizer > 0 ? m + log2f(normalizer) : -INFINITY;
    }
    ComputeSync();

    int const cmax = (p.capacity + p.block_extent - 1) / p.block_extent;
    for (int vector = ComputeThread(); vector < Q * D / 8; vector += kComputeThreads) {
      int const row = vector / (D / 8), d = (vector % (D / 8)) * 8;
      float m = -INFINITY, normalizer = 0, values[8] = {};
      for (int w = 0; w < kWarps; ++w) m = fmaxf(m, s.lse[w][row]);
      for (int w = 0; w < kWarps; ++w) {
        float const weight = isfinite(s.lse[w][row]) ? exp2f(s.lse[w][row] - m) : 0;
        normalizer += weight;
        float const* src = s.partial + (w * Q + row) * D + d;
        for (int i = 0; i < 8; ++i) values[i] += weight * src[i];
      }
      for (int i = 0; i < 8; ++i) values[i] = normalizer > 0 ? values[i] / normalizer : 0;
      if (cmax == 1) {
        alignas(16) Element result[8];
        for (int i = 0; i < 8; ++i) result[i] = Element(values[i]);
        backend::StoreGlobal16(p.context + ((std::size_t(b) * p.heads_kv + g) * Q + row) * D + d,
                               *reinterpret_cast<uint4 const*>(result));
      } else {
        std::size_t const base = ((std::size_t(b) * p.heads_kv + g) * cmax + c) * Q + row;
        for (int i = 0; i < 8; ++i) p.partial[base * D + d + i] = values[i];
        if (d == 0) p.lse[base] = m + log2f(normalizer);
      }
    }
    ComputeSync();
  }
};

}  // namespace tilemega::codegen::mx
