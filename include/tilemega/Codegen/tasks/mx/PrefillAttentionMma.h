// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Codegen/tasks/mx/PrefillAttentionMma.h
//
// xcore1000 prefill attention (SEQ > 1): causal attention for 16 query rows
// at a time against KV tiles of kKvTile keys, with the bf16 16x16x16 MMA for
// both S = Q K^T and O += P V (fragment layout: Backend/Mx/ServingGemmMma.h).
// Wave w computes key atoms [w*kKeyAtoms/2, ...) of S and dimension atoms
// [w*kDimAtoms/2, ...) of O. Row maxima and sums cross the two waves through
// shared memory; online softmax runs in the exp2 domain with the NVIDIA scale.
// P is rounded to bf16 for the PV product; the row sums use the FP32 values.
#pragma once

#include <tilemega/Backend/Mx/ServingGemmMma.h>
#include <tilemega/Codegen/executor/Prefetch.cuh>
#include <tilemega/Codegen/tasks/mx/AttentionCommon.h>
#include <tilemega/Codegen/tasks/TaskResources.h>

namespace tilemega::codegen::mx {

/// `Vt`: the value cache is V^T [B][hkv][D][cap] (the `vt` kv_layout of the
/// MX-4 dense decode program, MX4 prompt §3.4); otherwise rows [B][hkv][cap][D].
template <int D, int QPerKV, int Tokens, int QRows, int KvTile, bool QkNorm, bool Vt = false>
struct PrefillAttentionMma {
  static_assert(D == 64 || D == 128);
  static_assert(QPerKV >= 1 && QPerKV <= 8);
  static_assert((QRows > 0 && QRows % 16 == 0) || Tokens == 1);
  static_assert(KvTile == 32 || KvTile == 64);
  using Element = tilemega::Bf16;
  static constexpr int kWaves = kComputeThreads / maca::kWaveSize;
  static constexpr int kQueryExtent = Tokens * QPerKV;
  static constexpr int kKeyAtoms = KvTile / 16, kDimAtoms = D / 16;
  static_assert(kKeyAtoms % kWaves == 0 && kDimAtoms % kWaves == 0);
  static constexpr int kKeyAtomsPerWave = kKeyAtoms / kWaves, kDimAtomsPerWave = kDimAtoms / kWaves;

  struct SharedStorage {
    alignas(16) Element query[16 * D];
    alignas(16) Element key[KvTile * D];
    alignas(16) Element value_t[D * KvTile];
    alignas(16) Element prob[16 * KvTile];
    float maxima[2][16], sums[2][16];
    float row_max[16], row_sum[16], alpha[16], pad[16];
  };
  static_assert(sizeof(SharedStorage) == ServingAttentionSharedBytes(D, KvTile));

  __device__ static void Query(ServingAttentionOperands const& p, SharedStorage& s, int b, int g, int begin) {
    int const warp = ComputeThread() >> 5, lane = ComputeThread() & 31;
    constexpr int width = (QPerKV + 2) * D;
    for (int row = warp; row < 16; row += kComputeThreads / 32) {
      int const r = begin + row, token = r / QPerKV, head = r % QPerKV;
      if (r < kQueryExtent) {
        auto x = p.qkv + (std::size_t(b) * Tokens + token) * p.heads_kv * width + g * width + head * D;
        RotateRow<D, QkNorm>(x, p.q_norm, p.cosine + std::size_t(p.past + token) * D,
                             p.sine + std::size_t(p.past + token) * D, p.epsilon,
                             [&](int d, uint4 v) { *reinterpret_cast<uint4*>(s.query + row * D + d) = v; });
      } else if (lane * 8 < D) {
        *reinterpret_cast<uint4*>(s.query + row * D + lane * 8) = uint4{0, 0, 0, 0};
      }
    }
  }

  __device__ static void StoreValue(SharedStorage& s, int row, int d, uint4 v) {
    auto const* e = reinterpret_cast<Element const*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) s.value_t[(d + i) * KvTile + row] = e[i];
  }

  __device__ static void LoadKV(ServingAttentionOperands const& p, SharedStorage& s, int b, int g, int begin,
                                int limit, int query_begin, int q_begin) {
    constexpr int width = (QPerKV + 2) * D;
    std::size_t const head = (std::size_t(b) * p.heads_kv + g) * p.capacity;
    for (int i = ComputeThread() * 8; i < KvTile * D; i += kComputeThreads * 8) {
      int const row = i / D, d = i % D, position = begin + row;
      if (position < limit && position < p.past) {
        *reinterpret_cast<uint4*>(s.key + row * D + d) = backend::LoadGlobal16(p.key_cache + (head + position) * D + d);
        if constexpr (!Vt) StoreValue(s, row, d, backend::LoadGlobal16(p.value_cache + (head + position) * D + d));
      } else if (position >= limit) {
        *reinterpret_cast<uint4*>(s.key + row * D + d) = uint4{0, 0, 0, 0};
        if constexpr (!Vt) StoreValue(s, row, d, uint4{0, 0, 0, 0});
      }
    }
    if constexpr (Vt) {
      // V^T: one 16 B load is 8 tokens of one dimension, straight into
      // value_t. Positions >= limit are zero; [past, limit) are this call's
      // new tokens, written below by another thread, so a group that
      // reaches them goes element by element.
      int const cached = min(limit, p.past);
      std::size_t const vt = (std::size_t(b) * p.heads_kv + g) * D * p.capacity;
      for (int i = ComputeThread(); i < D * (KvTile / 8); i += kComputeThreads) {
        int const d = i / (KvTile / 8), row = i % (KvTile / 8) * 8, position = begin + row;
        Element* const target = s.value_t + d * KvTile + row;
        Element const* const source = p.value_cache + vt + std::size_t(d) * p.capacity + position;
        if (position + 8 <= cached) {
          *reinterpret_cast<uint4*>(target) = backend::LoadGlobal16(source);
        } else if (position >= limit) {
          *reinterpret_cast<uint4*>(target) = uint4{0, 0, 0, 0};
        } else {
          for (int e = 0; e < 8; ++e) {
            if (position + e < cached) target[e] = source[e];
            else if (position + e >= limit) target[e] = Element::bitcast(0);
          }
        }
      }
    }
    int const warp = ComputeThread() >> 5, lane = ComputeThread() & 31;
    for (int row = warp; row < KvTile; row += kComputeThreads / 32) {
      int const position = begin + row;
      if (position < p.past || position >= limit) continue;
      int const token = position - p.past;
      auto x = p.qkv + (std::size_t(b) * Tokens + token) * p.heads_kv * width + g * width + QPerKV * D;
      bool const writer =
          query_begin == q_begin && token * QPerKV >= q_begin && token * QPerKV < q_begin + QRows;
      RotateRow<D, QkNorm>(x, p.k_norm, p.cosine + std::size_t(position) * D, p.sine + std::size_t(position) * D,
                           p.epsilon, [&](int d, uint4 v) {
                             *reinterpret_cast<uint4*>(s.key + row * D + d) = v;
                             if (writer) backend::StoreGlobal16(p.key_cache + (head + position) * D + d, v);
                           });
      if (lane * 8 < D) {
        int const d = lane * 8;
        uint4 v = backend::LoadGlobal16(x + D + d);
        StoreValue(s, row, d, v);
        if (writer) {
          if constexpr (Vt) {
            // Column `position` of V^T: one element per dimension.
            auto const* e = reinterpret_cast<Element const*>(&v);
            Element* const column = p.value_cache + (std::size_t(b) * p.heads_kv + g) * D * p.capacity + position;
#pragma unroll
            for (int i = 0; i < 8; ++i) column[std::size_t(d + i) * p.capacity] = e[i];
          } else {
            backend::StoreGlobal16(p.value_cache + (head + position) * D + d, v);
          }
        }
      }
    }
  }

  __device__ static void Run(ServingAttentionOperands const& p, SharedStorage& s, int b, int g, int qb, int c) {
    int const block_begin = c * p.block_extent;
    int limit = min((c + 1) * p.block_extent, p.past + Tokens);
    int const q_begin = qb * QRows;
    if constexpr (Tokens > 1) limit = min(limit, p.past + (min(q_begin + QRows, kQueryExtent) - 1) / QPerKV + 1);
    if (block_begin >= limit) return;
    int const lane = ComputeThread() % maca::kWaveSize, wave = ComputeThread() / maca::kWaveSize;
    int const r = lane & 15, gq = lane >> 4;
    int const cmax = (p.capacity + p.block_extent - 1) / p.block_extent;
    float const scale = kLog2e / sqrtf(float(D));

    for (int query_begin = q_begin; query_begin < min(q_begin + QRows, kQueryExtent); query_begin += 16) {
      Query(p, s, b, g, query_begin);
      if (ComputeThread() < 16) {
        s.row_max[ComputeThread()] = -INFINITY;
        s.row_sum[ComputeThread()] = 0;
      }
      backend::mx::MmaAccum output[kDimAtomsPerWave];
#pragma unroll
      for (int j = 0; j < kDimAtomsPerWave; ++j) output[j] = backend::mx::MmaAccum{0.f, 0.f, 0.f, 0.f};

      for (int begin = block_begin; begin < limit; begin += KvTile) {
        LoadKV(p, s, b, g, begin, limit, query_begin, q_begin);
        ComputeSync();
        // S for this wave's key atoms: lane holds key r, query rows 4*gq+i.
        backend::mx::MmaAccum score[kKeyAtomsPerWave];
#pragma unroll
        for (int j = 0; j < kKeyAtomsPerWave; ++j) {
          int const key = (wave * kKeyAtomsPerWave + j) * 16 + r;
          backend::mx::MmaAccum acc{0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int d0 = 0; d0 < D; d0 += 32) {
            uint4 a = *reinterpret_cast<uint4 const*>(s.query + r * D + d0 + 8 * gq);
            uint4 k = *reinterpret_cast<uint4 const*>(s.key + key * D + d0 + 8 * gq);
            acc = backend::mx::Mma(backend::mx::Lo(a), backend::mx::Lo(k), acc);
            acc = backend::mx::Mma(backend::mx::Hi(a), backend::mx::Hi(k), acc);
          }
          score[j] = acc;
        }
        float maximum[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < kKeyAtomsPerWave; ++j) {
          int const position = begin + (wave * kKeyAtomsPerWave + j) * 16 + r;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            int const row = query_begin + 4 * gq + i, token = row / QPerKV;
            bool const valid = row < kQueryExtent && position < limit && position <= p.past + token;
            score[j][i] = valid ? score[j][i] * scale : -INFINITY;
            maximum[i] = fmaxf(maximum[i], score[j][i]);
          }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
          for (int delta = 1; delta < 16; delta *= 2)
            maximum[i] = fmaxf(maximum[i], __shfl_xor_sync(maca::kFullWaveMask, maximum[i], delta, 16));
          if (r == 0) s.maxima[wave][4 * gq + i] = maximum[i];
        }
        ComputeSync();
        if (ComputeThread() < 16) {
          int const row = ComputeThread();
          float m = s.row_max[row];
          for (int w = 0; w < kWaves; ++w) m = fmaxf(m, s.maxima[w][row]);
          s.alpha[row] = isfinite(s.row_max[row]) ? maca::Exp2(s.row_max[row] - m) : 0;
          s.row_max[row] = m;
        }
        ComputeSync();
        float sum[4] = {0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < kKeyAtomsPerWave; ++j) {
          int const key = (wave * kKeyAtomsPerWave + j) * 16 + r;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            int const row = 4 * gq + i;
            float const prob = isfinite(score[j][i]) ? maca::Exp2(score[j][i] - s.row_max[row]) : 0;
            sum[i] += prob;
            s.prob[row * KvTile + key] = Element(prob);
          }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
          for (int delta = 1; delta < 16; delta *= 2)
            sum[i] += __shfl_xor_sync(maca::kFullWaveMask, sum[i], delta, 16);
          if (r == 0) s.sums[wave][4 * gq + i] = sum[i];
        }
        ComputeSync();
        if (ComputeThread() < 16) {
          int const row = ComputeThread();
          float total = s.row_sum[row] * s.alpha[row];
          for (int w = 0; w < kWaves; ++w) total += s.sums[w][row];
          s.row_sum[row] = total;
        }
        // O rows 4*gq+i are rescaled, then O += P V over this tile's keys.
#pragma unroll
        for (int j = 0; j < kDimAtomsPerWave; ++j)
#pragma unroll
          for (int i = 0; i < 4; ++i) output[j][i] *= s.alpha[4 * gq + i];
#pragma unroll
        for (int j = 0; j < kDimAtomsPerWave; ++j) {
          int const d = (wave * kDimAtomsPerWave + j) * 16 + r;
#pragma unroll
          for (int k0 = 0; k0 < KvTile; k0 += 32) {
            uint4 a = *reinterpret_cast<uint4 const*>(s.prob + r * KvTile + k0 + 8 * gq);
            uint4 v = *reinterpret_cast<uint4 const*>(s.value_t + d * KvTile + k0 + 8 * gq);
            output[j] = backend::mx::Mma(backend::mx::Lo(a), backend::mx::Lo(v), output[j]);
            output[j] = backend::mx::Mma(backend::mx::Hi(a), backend::mx::Hi(v), output[j]);
          }
        }
        ComputeSync();
      }

#pragma unroll
      for (int j = 0; j < kDimAtomsPerWave; ++j) {
        int const d = (wave * kDimAtomsPerWave + j) * 16 + r;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int const local = 4 * gq + i, row = query_begin + local;
          if (row >= kQueryExtent) continue;
          float const total = s.row_sum[local];
          float const value = total > 0 ? output[j][i] / total : 0;
          if (cmax == 1 || Tokens > 1) {
            p.context[((std::size_t(b) * Tokens + row / QPerKV) * p.heads_kv * QPerKV + g * QPerKV + row % QPerKV) *
                          D + d] = Element(value);
          } else {
            std::size_t const base = ((std::size_t(b) * p.heads_kv + g) * cmax + c) * kQueryExtent + row;
            p.partial[base * D + d] = value;
            if (d == 0) p.lse[base] = s.row_max[local] + log2f(total);
          }
        }
      }
      ComputeSync();
    }
  }
};

}  // namespace tilemega::codegen::mx
