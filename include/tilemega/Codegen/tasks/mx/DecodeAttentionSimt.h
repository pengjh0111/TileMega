// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Codegen/tasks/mx/DecodeAttentionSimt.h
//
// xcore1000 decode attention (`attention_impl=simt`, one token per request):
// one CTA owns (batch, KV group, KV block). Lane groups of D/8 lanes read K
// and V rows coalesced, eight dimensions per lane, straight from the global
// KV cache. Tokens are processed in chunks: every lane writes its 8-dim
// partial dot products to shared memory, one thread per (token, query) sums
// them, and each lane group then runs an online softmax over its own tokens
// in the exp2 domain (same scale as the NVIDIA bodies). The per-token
// cross-lane shuffle reduction this replaces cost ~46 ns per shuffle on
// xcore1000 and dominated the loop (measured). The next chunk's K/V loads are
// issued before the chunk's barriers. Lane groups merge by xor shuffles
// inside a warp; warps merge in a fixed order (C-2) into one shared buffer,
// which overlays the chunk buffers so Q = 8 fits a 16 KiB CTA (residency 4).
//
// The current token's K (q/k-norm + RoPE) and V are produced here, written to
// the cache at `past`, and read back from shared memory by this CTA.
#pragma once

#include <tilemega/Codegen/tasks/mx/AttentionCommon.h>
#include <tilemega/Codegen/tasks/TaskResources.h>

namespace tilemega::codegen::mx {

template <int D, int Q, bool QkNorm>
struct DecodeAttentionSimt {
  static_assert((D == 64 || D == 128) && Q >= 1 && Q <= 8, "simt decode attention: D in {64,128}, Q <= 8");
  using Element = tilemega::Bf16;
  static constexpr int kWarps = kComputeThreads / 32;
  static constexpr int kLanes = D / 8, kGroups = 32 / kLanes, kAllGroups = kWarps * kGroups;
  /// Tokens per lane group per chunk; the partial buffer stays at 4 KiB.
  static constexpr int kUnroll = Q >= 8 ? 1 : 8 / Q;
  static constexpr int kChunk = kAllGroups * kUnroll;
  static_assert(kChunk * Q <= kComputeThreads, "one reducing thread per (token, query)");

  struct Chunk {
    float part[kChunk * Q][kLanes];
    float score[kChunk * Q];
  };
  struct Merge {
    alignas(16) float out[Q * D];
    float max[kWarps][Q];
    float sum[kWarps][Q];
  };
  struct SharedStorage {
    alignas(16) Element query[Q * D];
    alignas(16) Element new_key[D];
    alignas(16) Element new_value[D];
    union {
      Chunk chunk;
      Merge merge;
    };
  };

  static_assert(sizeof(SharedStorage) == ServingDecodeAttentionSimtSharedBytes(D, Q));

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

    int const group = warp * kGroups + lane / kLanes, sub = lane % kLanes, column = sub * 8;
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

    // Chunk token j = u * kAllGroups + group. Positions past `end` read the
    // last valid row (unconditional loads); their scores are -inf.
    uint4 key[kUnroll], value[kUnroll];
    auto load = [&](int first) {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        int const position = min(first + u * kAllGroups + group, end - 1);
        key[u] = backend::LoadGlobal16(p.key_cache + (head + position) * D + column);
        value[u] = backend::LoadGlobal16(p.value_cache + (head + position) * D + column);
        if (position == p.past) {
          key[u] = *reinterpret_cast<uint4 const*>(s.new_key + column);
          value[u] = *reinterpret_cast<uint4 const*>(s.new_value + column);
        }
      }
    };
    load(begin);
    for (int first = begin; first < end; first += kChunk) {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        auto const* k = reinterpret_cast<Element const*>(&key[u]);
#pragma unroll
        for (int q = 0; q < Q; ++q) {
          float dot = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) dot += query[q][i] * float(k[i]);
          s.chunk.part[(u * kAllGroups + group) * Q + q][sub] = dot;
        }
      }
      uint4 current[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) current[u] = value[u];
      if (first + kChunk < end) load(first + kChunk);
      ComputeSync();
      if (ComputeThread() < kChunk * Q) {
        int const j = ComputeThread() / Q;
        float total = 0;
#pragma unroll
        for (int l = 0; l < kLanes; ++l) total += s.chunk.part[ComputeThread()][l];
        s.chunk.score[ComputeThread()] = first + j < end ? total : -INFINITY;
      }
      ComputeSync();
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        auto const* v = reinterpret_cast<Element const*>(&current[u]);
#pragma unroll
        for (int q = 0; q < Q; ++q) {
          float const score = s.chunk.score[(u * kAllGroups + group) * Q + q];
          float const next = fmaxf(maximum[q], score);
          float const alpha = isfinite(maximum[q]) ? maca::Exp2(maximum[q] - next) : 0;
          float const weight = isfinite(score) ? maca::Exp2(score - next) : 0;
          sum[q] = sum[q] * alpha + weight;
          maximum[q] = next;
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[q][i] = acc[q][i] * alpha + weight * float(v[i]);
        }
      }
      // The next chunk's partial writes reuse `part`; scores were read above.
      ComputeSync();
    }

    // Merge the lane groups of each warp (xor offsets kLanes..16 stay in the
    // 32-lane warp), relative to the warp maximum.
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      float m = maximum[q];
#pragma unroll
      for (int delta = kLanes; delta < 32; delta *= 2)
        m = fmaxf(m, __shfl_xor_sync(maca::kFullWaveMask, m, delta, 32));
      float const weight = isfinite(maximum[q]) ? maca::Exp2(maximum[q] - m) : 0;
      float total = weight * sum[q];
#pragma unroll
      for (int delta = kLanes; delta < 32; delta *= 2)
        total += __shfl_xor_sync(maca::kFullWaveMask, total, delta, 32);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float value = weight * acc[q][i];
#pragma unroll
        for (int delta = kLanes; delta < 32; delta *= 2)
          value += __shfl_xor_sync(maca::kFullWaveMask, value, delta, 32);
        acc[q][i] = value;
      }
      maximum[q] = m;
      sum[q] = total;
      if (lane == 0) {
        s.merge.max[warp][q] = m;
        s.merge.sum[warp][q] = total;
      }
    }
    ComputeSync();
    // Warps accumulate in a fixed order (C-2): out = sum_w acc_w * 2^(m_w - M).
    for (int w = 0; w < kWarps; ++w) {
      if (warp == w && lane < kLanes) {
#pragma unroll
        for (int q = 0; q < Q; ++q) {
          float m = -INFINITY;
          for (int v = 0; v < kWarps; ++v) m = fmaxf(m, s.merge.max[v][q]);
          float const weight = isfinite(maximum[q]) ? maca::Exp2(maximum[q] - m) : 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float& slot = s.merge.out[q * D + column + i];
            slot = (w == 0 ? 0.f : slot) + weight * acc[q][i];
          }
        }
      }
      ComputeSync();
    }

    int const cmax = (p.capacity + p.block_extent - 1) / p.block_extent;
    for (int vector = ComputeThread(); vector < Q * D / 8; vector += kComputeThreads) {
      int const row = vector / (D / 8), d = (vector % (D / 8)) * 8;
      float m = -INFINITY, normalizer = 0;
      for (int w = 0; w < kWarps; ++w) m = fmaxf(m, s.merge.max[w][row]);
      for (int w = 0; w < kWarps; ++w)
        normalizer += isfinite(s.merge.max[w][row]) ? maca::Exp2(s.merge.max[w][row] - m) * s.merge.sum[w][row] : 0;
      float values[8];
      for (int i = 0; i < 8; ++i) values[i] = normalizer > 0 ? s.merge.out[row * D + d + i] / normalizer : 0;
      if (cmax == 1) {
        alignas(16) Element result[8];
        for (int i = 0; i < 8; ++i) result[i] = Element(values[i]);
        backend::StoreGlobal16(p.context + ((std::size_t(b) * p.heads_kv + g) * Q + row) * D + d,
                               *reinterpret_cast<uint4 const*>(result));
      } else {
        std::size_t const base = ((std::size_t(b) * p.heads_kv + g) * cmax + c) * Q + row;
        for (int i = 0; i < 8; ++i) p.partial[base * D + d + i] = values[i];
        if (d == 0) p.lse[base] = normalizer > 0 ? m + log2f(normalizer) : -INFINITY;
      }
    }
    ComputeSync();
  }
};

}  // namespace tilemega::codegen::mx
