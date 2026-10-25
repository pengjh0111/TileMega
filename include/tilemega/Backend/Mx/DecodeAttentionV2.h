// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Mx/DecodeGemmV2.h>
#include <tilemega/Backend/Mx/KvFragmentLayoutV2.h>
#include <tilemega/Backend/Mx/RowTransformV2.h>

namespace tilemega::backend::mx {

template <int Threads = 576>
struct MxAttentionShapeV2 { static constexpr int kThreads = Threads; int b, hkv; };
template <int B, int Hkv, int Threads = 576>
struct MxStaticAttentionShapeV2 { static constexpr int kThreads = Threads, b = B, hkv = Hkv; };
struct MxAttentionOperandsV2 {
  Bf16 const *qkv, *q_norm, *k_norm, *cosine, *sine;
  Bf16 *k_cache, *v_cache, *ctx;
  float* partial;
  unsigned* tickets;
  int capacity, past;
  float epsilon;
};

struct MxAttentionGeometryV2 {
  int blocks, units, active, begin, end;
  TM_MX_HD constexpr MxAttentionGeometryV2(int heads, int past, int bmin, int grid, int cta)
      : blocks((past + 32) / 32), units(heads * blocks),
        active((units + bmin - 1) / bmin < grid ? (units + bmin - 1) / bmin : grid),
        begin(0), end(0) {
#if defined(__CUDA_ARCH__) || defined(__MACA_ARCH__)
    // Every launched CTA lies in this grid, including through opaque reads.
    __builtin_assume(cta >= 0 && cta < grid);
#endif
    begin = cta < active ? cta * units / active : units;
    end = cta < active ? (cta + 1) * units / active : units;
  }
  TM_MX_HD constexpr int Owner(int unit) const { return ((unit + 1) * active - 1) / units; }
  TM_MX_HD constexpr int FirstOwner(int head) const { return Owner(head * blocks); }
  TM_MX_HD constexpr int Producers(int head) const { return Owner((head + 1) * blocks - 1) - FirstOwner(head) + 1; }
};

// The fixed payload kind is derived from V_pf: 0 = none, 1 = K, 2 = K+V.
template <int D, int Q, int Bmin, bool QkNorm, int PrefetchKind = 2, int Capacity = 1088>
struct MxAttentionV2 {
  static_assert(D == 64 || D == 128);
  static_assert(Q >= 1 && Q <= 8);
  static_assert(Bmin == 1 || Bmin == 2 || Bmin == 4 || Bmin == 8);
  static_assert(PrefetchKind >= 0 && PrefetchKind <= 2);
  static constexpr int kChunks = D / 32, kAtoms = D / 16;
  static constexpr int kKLoads = 2 * kChunks, kVLoads = kAtoms;
  static constexpr int kSlots = (Capacity / 32 < 104 ? Capacity / 32 : 104);
  static constexpr int kWaveFloats = 64 + D / 2 + Q * (D + 2);
  static constexpr int kMergeFloats = kSlots * Q * 3;
  static constexpr int kScratchFloats = kMergeFloats > Q * (D + 2) ? kMergeFloats : Q * (D + 2);
  static constexpr int kSharedBytes = (8 * kWaveFloats + kScratchFloats + 4) * 4;
  using Layout = MxKvFragmentLayoutV2<D>;
  struct Prefetched {
    static constexpr int kLoads = (PrefetchKind ? kKLoads : 0) + (PrefetchKind == 2 ? kVLoads : 0);
    uint4 k[PrefetchKind ? 2 : 1][PrefetchKind ? kChunks : 1];
    uint4 v[PrefetchKind == 2 ? kAtoms : 1];
  };
  struct Block { uint4 k[2][kChunks], v[kAtoms]; bool valid; };

  __device__ __forceinline__ static void LoadK(Block& block, MxAttentionOperandsV2 const& p,
                                               int head, int j, int lane) {
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int dc = 0; dc < kChunks; ++dc)
        block.k[a][dc] = LoadGlobal16Gvm(p.k_cache + Layout::KWave(head, j, dc, a, lane, p.capacity));
    asm volatile("" ::: "memory");
  }
  __device__ __forceinline__ static void LoadV(Block& block, MxAttentionOperandsV2 const& p,
                                               int head, int j, int lane) {
#pragma unroll
    for (int n = 0; n < kAtoms; ++n)
      block.v[n] = LoadGlobal16Gvm(p.v_cache + Layout::VWave(head, j, n, lane, p.capacity));
    asm volatile("" ::: "memory");
  }
  template <class Shape>
  __device__ __forceinline__ static Prefetched Prefetch(MxAttentionOperandsV2 const& p, Shape const& s) {
    Prefetched pf{};
    unsigned const thread = StageThread();
    __builtin_assume(thread < unsigned(Shape::kThreads));
    int const wave = __builtin_mxc_readfirstlane(int(thread / 64));
    __builtin_assume(unsigned(wave) < unsigned(Shape::kThreads) / 64);
    if (wave < 8) {
      MxAttentionGeometryV2 const geo(s.b * s.hkv, p.past, Bmin, int(gridDim.x), StageBlock());
      int const first = min(geo.begin, geo.units - 1), head = first / geo.blocks;
      int const j = min(first % geo.blocks + wave, geo.blocks - 1), lane = int(thread % 64);
      Block block;
      if constexpr (PrefetchKind) {
        LoadK(block, p, head, j, lane);
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
          for (int dc = 0; dc < kChunks; ++dc) pf.k[a][dc] = block.k[a][dc];
      }
      if constexpr (PrefetchKind == 2) {
        LoadV(block, p, head, j, lane);
#pragma unroll
        for (int n = 0; n < kAtoms; ++n) pf.v[n] = block.v[n];
      }
    }
    return pf;
  }
  __device__ __forceinline__ static uint4 MaskV(uint4 x, int block, int past, int group) {
    int const valid = min(max(past + 1 - block * 32 - group * 8, 0), 8);
    unsigned mask[4];
#pragma unroll
    for (int w = 0; w < 4; ++w)
      mask[w] = (2 * w < valid ? 0xffffu : 0u) | (2 * w + 1 < valid ? 0xffff0000u : 0u);
    return uint4{x.x & mask[0], x.y & mask[1], x.z & mask[2], x.w & mask[3]};
  }

  template <bool First, class Shape>
  __device__ __forceinline__ static void Segment(Prefetched const& pf, Block& block,
                                                 MxAttentionOperandsV2 const& p, Shape const& s,
                                                 MxAttentionGeometryV2 const& geo, int head, int first, int end,
                                                 float* shared) {
    using Sync = codegen::mx::StageSyncV2<Shape::kThreads>;
    int const t = StageThread();
    __builtin_assume(unsigned(t) < unsigned(Shape::kThreads));
    int const wave = __builtin_mxc_readfirstlane(t / 64), lane = t % 64;
    __builtin_assume(unsigned(wave) < unsigned(Shape::kThreads) / 64);
    int const r = lane & 15, g = lane >> 4;
    float* const workspace = shared + min(wave, 7) * kWaveFloats;
    float* const exchange = workspace;
    auto* const new_value = reinterpret_cast<Bf16*>(workspace + 64);
    float* const wave_result = workspace + 64 + D / 2;
    int const count = max(0, (end - first - wave + 7) / 8);
    uint4 query[kChunks];
    uint4 new_key[kChunks];
    unsigned new_values[kAtoms];
    if (wave < 8 && count) {
      uint4 cosine[kChunks], sine[kChunks], weights[kChunks], key_weights[kChunks];
      uint4 new_v{};
      int const offset = p.past % 32, owner_r = 4 * (offset / 8) + offset % 4;
      bool const owner = (geo.blocks - 1 >= first + wave) && (geo.blocks - 1 - first - wave) % 8 == 0
                         && end == geo.blocks;
#pragma unroll
      for (int dc = 0; dc < kChunks; ++dc) {
        int const at = dc * 32 + 8 * g;
        cosine[dc] = LoadGlobal16Gvm(p.cosine + p.past * D + at);
        sine[dc] = LoadGlobal16Gvm(p.sine + p.past * D + at);
        query[dc] = LoadGlobal16Gvm(p.qkv + (head * (Q + 2) + min(r, Q - 1)) * D + at);
        weights[dc] = uint4{};
        if constexpr (QkNorm) weights[dc] = LoadGlobal16Gvm(p.q_norm + at);
        new_key[dc] = uint4{};
        key_weights[dc] = uint4{};
      }
      if (owner) {
#pragma unroll
        for (int dc = 0; dc < kChunks; ++dc) {
          int const at = dc * 32 + 8 * g;
          if (r == owner_r) new_key[dc] = LoadGlobal16Gvm(p.qkv + (head * (Q + 2) + Q) * D + at);
          if constexpr (QkNorm) key_weights[dc] = LoadGlobal16Gvm(p.k_norm + at);
        }
        if (lane < D / 8)
          new_v = LoadGlobal16Gvm(p.qkv + (head * (Q + 2) + Q + 1) * D + lane * 8);
      }
      if constexpr (First) {
        if constexpr (PrefetchKind) {
#pragma unroll
          for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int dc = 0; dc < kChunks; ++dc) block.k[a][dc] = pf.k[a][dc];
        } else LoadK(block, p, head, first + wave, lane);
        if constexpr (PrefetchKind == 2) {
#pragma unroll
          for (int n = 0; n < kAtoms; ++n) block.v[n] = pf.v[n];
        } else LoadV(block, p, head, first + wave, lane);
      } else if (!block.valid) {
        LoadK(block, p, head, first + wave, lane);
        LoadV(block, p, head, first + wave, lane);
      }
      // Prefix inputs precede fresh history; keep only that younger batch outstanding.
      asm volatile("" ::: "memory");
      if constexpr (First) {
        if constexpr (PrefetchKind == 0) __builtin_mxc_arrive_gvmcnt(kKLoads + kVLoads);
        else if constexpr (PrefetchKind == 1) __builtin_mxc_arrive_gvmcnt(kVLoads);
        else __builtin_mxc_arrive_gvmcnt(0);
      } else if (!block.valid) __builtin_mxc_arrive_gvmcnt(kKLoads + kVLoads);
      else __builtin_mxc_arrive_gvmcnt(0);
      if (owner) {
        MxRowTransformV2<D, QkNorm>(new_key, key_weights, cosine, sine, p.epsilon, exchange, lane);
        __syncwarp();
      }
      MxRowTransformV2<D, QkNorm>(query, weights, cosine, sine, p.epsilon, exchange, lane);
      __syncwarp();
      if (owner && lane < D / 8)
        *reinterpret_cast<uint4*>(new_value + lane * 8) = new_v;
      __syncwarp();
#pragma unroll
      for (int n = 0; n < kAtoms; ++n) {
        unsigned value = owner ? unsigned(new_value[16 * n + r].raw()) : 0u;
        // Retire this LDS value before K_next is issued. Combining a fresh
        // LDS load with a V fragment made MACA insert ARRIVE 4160 (MIR).
        asm volatile("" : "+r"(value));
        new_values[n] = value;
      }
      unsigned mask = 0u - unsigned(r < Q);
      asm volatile("" : "+r"(mask));
#pragma unroll
      for (int dc = 0; dc < kChunks; ++dc)
        query[dc] = uint4{query[dc].x & mask, query[dc].y & mask, query[dc].z & mask, query[dc].w & mask};

      MmaAccum out[kAtoms];
#pragma unroll
      for (int n = 0; n < kAtoms; ++n) out[n] = MmaAccum{0.f, 0.f, 0.f, 0.f};
      float maximum = -INFINITY, lambda = 0;
#pragma unroll 1
      for (int index = 0; index < count; ++index) {
        int const j = first + wave + 8 * index;
        int const next_head = index + 1 < count ? head : head + 1;
        int const next_j = index + 1 < count ? j + 8 : wave;
        bool const has_next = index + 1 < count || ((head + 1) * geo.blocks + wave < geo.end);
        block.valid = has_next;
        __builtin_mxc_arrive_gvmcnt(kVLoads);
        if (j == geo.blocks - 1) {
#pragma unroll
          for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int dc = 0; dc < kChunks; ++dc)
              if (a == offset % 8 / 4 && r == owner_r) block.k[a][dc] = new_key[dc];
          // DT-KV: the owner writes the complete block, including unchanged rows.
          // Keep whole-block store addresses out of the MMA live range.
          int const store_cap = Opaque(p.capacity);
#pragma unroll
          for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int dc = 0; dc < kChunks; ++dc)
              MxStore16V2(p.k_cache + Layout::KWave(head, j, dc, a, lane, store_cap), block.k[a][dc]);
        }
        float score[8];
#pragma unroll
        for (int a = 0; a < 2; ++a) {
          MmaAccum acc{0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int dc = 0; dc < kChunks; ++dc) {
            acc = Mma(Lo(block.k[a][dc]), Lo(query[dc]), acc);
            acc = Mma(Hi(block.k[a][dc]), Hi(query[dc]), acc);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i)
            score[4 * a + i] = j * 32 + 8 * g + 4 * a + i <= p.past
                               ? acc[i] * (codegen::mx::kLog2e / sqrtf(float(D))) : -INFINITY;
        }
        // X-A: one K slot is reused only after its S^T consumer.
        asm volatile("" :: "r"(score[7]) : "memory");
        if (has_next) LoadK(block, p, next_head, next_j, lane);
        float block_max = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) block_max = fmaxf(block_max, score[i]);
        exchange[g * 16 + r] = block_max;
        __syncwarp();
        block_max = fmaxf(fmaxf(exchange[r], exchange[16 + r]), fmaxf(exchange[32 + r], exchange[48 + r]));
        float const new_max = fmaxf(maximum, block_max), alpha = maca::Exp2(maximum - new_max);
        // All groups agree; only group zero publishes the O-layout scalars.
        __syncwarp();
        if (!g) exchange[r] = alpha;
        __syncwarp();
        float output_alpha[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) output_alpha[i] = exchange[4 * g + i];
        alignas(16) Bf16 probability[8];
        float block_sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float const e = maca::Exp2(score[i] - new_max);
          probability[i] = Bf16(e); block_sum += e;
        }
        uint4 const prob = *reinterpret_cast<uint4 const*>(probability);
        if (has_next) __builtin_mxc_arrive_gvmcnt(kKLoads);
        else __builtin_mxc_arrive_gvmcnt(0);
        if (j == geo.blocks - 1) {
#pragma unroll
          for (int n = 0; n < kAtoms; ++n) {
            alignas(16) Bf16 v[8];
            *reinterpret_cast<uint4*>(v) = block.v[n];
#pragma unroll
            for (int e = 0; e < 8; ++e)
              v[e] = Bf16::bitcast(g == offset / 8 && e == offset % 8 ? new_values[n] : unsigned(v[e].raw()));
            block.v[n] = *reinterpret_cast<uint4 const*>(v);
          }
        }
#pragma unroll
        for (int n = 0; n < kAtoms; ++n) {
#pragma unroll
          for (int i = 0; i < 4; ++i) out[n][i] *= output_alpha[i];
          uint4 const value = MaskV(block.v[n], j, p.past, g);
          out[n] = Mma(Lo(prob), Lo(value), out[n]);
          out[n] = Mma(Hi(prob), Hi(value), out[n]);
        }
        if (j == geo.blocks - 1) {
          int const store_cap = Opaque(p.capacity);
#pragma unroll
          for (int n = 0; n < kAtoms; ++n)
            MxStore16V2(p.v_cache + Layout::VWave(head, j, n, lane, store_cap), block.v[n]);
        }
        asm volatile("" :: "r"(out[kAtoms - 1][3]) : "memory");
        if (has_next) LoadV(block, p, next_head, next_j, lane);
        lambda = alpha * lambda + block_sum; maximum = new_max;
      }
      exchange[g * 16 + r] = lambda;
      __syncwarp();
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int const q = 4 * g + i;
        if (q < Q) {
#pragma unroll
          for (int n = 0; n < kAtoms; ++n) wave_result[q * (D + 2) + 16 * n + r] = out[n][i];
        }
      }
      if (!g && r < Q) {
        wave_result[r * (D + 2) + D] = maximum;
        wave_result[r * (D + 2) + D + 1] = ((exchange[r] + exchange[16 + r]) + exchange[32 + r]) + exchange[48 + r];
      }
    } else block.valid = false;
    Sync::CtaSync();
    // Each thread owns (q,d), and merges waves in their prescribed order.
    float* const merged = shared + 64 + D / 2;
    if (wave < 8) {
      for (int item = t; item < Q * (D + 2); item += 512) {
        int const q = item / (D + 2), d = item % (D + 2);
        float m = -INFINITY;
#pragma unroll
        for (int w = 0; w < 8; ++w)
          if (first + w < end) m = fmaxf(m, shared[w * kWaveFloats + 64 + D / 2 + q * (D + 2) + D]);
        float result = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          if (first + w < end) {
            float const* values = shared + w * kWaveFloats + 64 + D / 2 + q * (D + 2);
            result += values[d] * maca::Exp2(values[D] - m);
          }
        }
        // Keep wave zero's inputs intact until every output thread finishes.
        float* const scratch = shared + 8 * kWaveFloats;
        if (d == D) result = m;
        scratch[item] = result;
      }
    }
    Sync::CtaSync();
    if (wave < 8)
      for (int item = t; item < Q * (D + 2); item += 512) merged[item] = shared[8 * kWaveFloats + item];
    Sync::CtaSync();
    int const producers = geo.Producers(head), ordinal = StageBlock() - geo.FirstOwner(head);
    if (producers == 1) WriteContext(merged, p, head, t);
    else {
      float* const base = p.partial + head * kSlots * Q * (D + 2);
      if (wave < 8)
        for (int item = t; item < Q * (D + 2); item += 512)
          base[ordinal * Q * (D + 2) + item] = merged[item];
      unsigned* const flag = reinterpret_cast<unsigned*>(shared + kSharedBytes / 4 - 4);
      codegen::executor::LastArriverV2<Sync>::Run(p.tickets + head, unsigned(producers), flag, [&] {
        MergeBody<Sync>(base, p, head, producers, t, shared + 8 * kWaveFloats);
      });
    }
    Sync::CtaSync();
  }

  __device__ __forceinline__ static void WriteContext(float const* result, MxAttentionOperandsV2 const& p,
                                                      int head, int thread) {
    if (thread < 512)
      for (int item = thread; item < Q * D; item += 512) {
        int const q = item / D, d = item % D;
        p.ctx[(head * Q + q) * D + d] = Bf16(result[q * (D + 2) + d] / result[q * (D + 2) + D + 1]);
      }
  }
  template <class Sync>
  __device__ __forceinline__ static void MergeBody(float const* base, MxAttentionOperandsV2 const& p,
                                                  int head, int producers, int thread, float* scratch) {
    if (thread < 512) {
      for (int item = thread; item < Q * kSlots; item += 512) {
        int const q = item / kSlots, c = item % kSlots, at = (c * Q + q) * (D + 2);
        unsigned m_bits = 0xff800000u, l_bits = 0;
        if (c < producers) {
          // D+2 is even, so each contiguous m/l pair is eight-byte aligned.
          using Pair = unsigned __attribute__((ext_vector_type(2)));
          using GlobalPair = __attribute__((address_space(1))) Pair const;
          Pair const ml = *(GlobalPair*)(base + at + D);
          m_bits = ml[0]; l_bits = ml[1];
        }
        asm volatile("" ::: "memory");
        __builtin_mxc_arrive_gvmcnt(0);
        scratch[item * 3] = __uint_as_float(m_bits);
        scratch[item * 3 + 1] = __uint_as_float(l_bits);
      }
    }
    Sync::CtaSync();
    if (thread < 512) {
      for (int q = thread; q < Q; q += 512) {
        float m = -INFINITY;
#pragma unroll
        for (int c = 0; c < kSlots; ++c) m = fmaxf(m, scratch[(q * kSlots + c) * 3]);
        scratch[q * kSlots * 3 + 2] = m;
      }
    }
    Sync::CtaSync();
    if (thread < 512) {
      for (int item = thread; item < Q * kSlots; item += 512) {
        int const q = item / kSlots;
        float const m = scratch[q * kSlots * 3 + 2];
        scratch[item * 3] = maca::Exp2(scratch[item * 3] - m);
      }
    }
    Sync::CtaSync();
    if (thread < 512) {
      for (int q = thread; q < Q; q += 512) {
        float lambda = 0;
#pragma unroll
        for (int c = 0; c < kSlots; ++c)
          lambda += scratch[(q * kSlots + c) * 3] * scratch[(q * kSlots + c) * 3 + 1];
        scratch[q * kSlots * 3 + 2] = lambda;
      }
    }
    Sync::CtaSync();
    if (thread < 512) {
      for (int item = thread; item < Q * D; item += 512) {
        int const q = item / D, d = item % D;
        float sum = 0;
        for (int start = 0; start < producers; start += 32) {
          float values[32];
#pragma unroll
          for (int c = 0; c < 32; ++c)
            values[c] = LoadFloatGvm(base + ((min(start + c, producers - 1) * Q + q) * (D + 2) + d));
          // Keep the complete bounded load batch ahead of its ordered reduction.
          asm volatile("" ::: "memory");
          __builtin_mxc_arrive_gvmcnt(0);
#pragma unroll
          for (int c = 0; c < 32; ++c)
            if (start + c < producers) sum += values[c] * scratch[(q * kSlots + start + c) * 3];
        }
        p.ctx[(head * Q + q) * D + d] = Bf16(sum / scratch[q * kSlots * 3 + 2]);
      }
    }
  }

  template <class Shape, class Next>
  __device__ __forceinline__ static auto Run(Prefetched const& pf, MxAttentionOperandsV2 const& p,
                                            Shape const& s, char* smem, Next next) {
    using Sync = codegen::mx::StageSyncV2<Shape::kThreads>;
    MxAttentionGeometryV2 const geo(s.b * s.hkv, p.past, Bmin, int(gridDim.x), StageBlock());
    float* const shared = reinterpret_cast<float*>(smem);
    Block block;
    block.valid = false;
    if (geo.begin < geo.end) {
      int const head = geo.begin / geo.blocks;
      Segment<true>(pf, block, p, s, geo, head, geo.begin % geo.blocks,
                    min(geo.end - head * geo.blocks, geo.blocks), shared);
#pragma unroll 1
      for (int h = head + 1; h * geo.blocks < geo.end; ++h)
        Segment<false>(pf, block, p, s, geo, h, 0, min(geo.end - h * geo.blocks, geo.blocks), shared);
    }
    Sync::CtaSync();
    using Result = decltype(next());
    unsigned const thread = StageThread();
    __builtin_assume(thread < unsigned(Shape::kThreads));
    if (thread < 512) return next();
    return Result{};
  }
};

}  // namespace tilemega::backend::mx
