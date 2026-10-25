// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Mx/DecodeGemmV2.h>

namespace tilemega::backend::mx {

struct MxHeadPartialV2 { float value; unsigned index; };
struct MxHeadOperandsV2 {
  MxGemmOperandsV2 gemm;
  MxHeadPartialV2* partial;
  unsigned* ticket;
  int* tokens;
  int capacity, past;
};

__device__ __forceinline__ MxHeadPartialV2 MxBetterV2(MxHeadPartialV2 a, MxHeadPartialV2 b) {
  return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}
__device__ __forceinline__ MxHeadPartialV2 MxHeadLoadV2(MxHeadPartialV2 const* p) {
  using Vector = unsigned __attribute__((ext_vector_type(2)));
  using Global = __attribute__((address_space(1))) Vector const;
  Vector const v = *(Global*)p;
  return {__uint_as_float(v[0]), v[1]};
}
__device__ __forceinline__ void MxHeadStoreV2(MxHeadPartialV2* p, MxHeadPartialV2 v) {
  using Vector = unsigned __attribute__((ext_vector_type(2)));
  using Global = __attribute__((address_space(1))) Vector;
  *(Global*)p = Vector{__float_as_uint(v.value), v.index};
}

template <int TN, int WN, int D, int Pw = D>
struct MxHead {
  static_assert(WN == 4 || WN == 8);
  using Core = MxGemm<TN, WN, D, MxOpV2::kStore, true, MxPart::kTileRange, Pw>;
  using Prefetched = typename Core::Prefetched;
  using Lane = typename Core::Lane;
  using Ring = typename Core::Ring;
  static constexpr int kAtoms = TN / 16, kWK = 8 / WN;

  TM_MX_HD static constexpr int SmemBytes(int b) {
    int const core = MxGemmSmemV2(b, TN, WN).bytes;
    return ((core > b * 128 * 8 ? core : b * 128 * 8) + 31) / 16 * 16;
  }
  template <class Shape>
  __device__ __forceinline__ static Prefetched Prefetch(MxHeadOperandsV2 const& p, Shape const& s) {
    return Core::Prefetch(p.gemm, s);
  }

  template <bool First, class Shape>
  __device__ __forceinline__ static void Round(Prefetched const& pf, Ring& ring, float& inverse,
                                               MxHeadPartialV2& best, MxHeadOperandsV2 const& p,
                                               Shape const& s, float* shared, MxGemmSmemV2 layout,
                                               MxGeometryV2<TN, WN, MxPart::kTileRange> const& geo,
                                               Lane const& l, int round) {
    using Sync = codegen::mx::StageSyncV2<Shape::kThreads>;
    auto const segment = geo.Segment(round, l.group);
    MmaAccum acc[kAtoms];
#pragma unroll
    for (int j = 0; j < kAtoms; ++j) acc[j] = MmaAccum{0.f, 0.f, 0.f, 0.f};
    float ss = 0;
    if (__builtin_mxc_readfirstlane(int(l.Compute() && segment.active))) {
      int const count = MxWaveChunksV2(segment, l.wave_k, kWK);
      int consumed = 0;
      if constexpr (First && Pw == D) {
        Core::PrimePrefetched(pf, ring, segment, p.gemm, s, l);
      } else if constexpr (First && Pw > 0) {
        Core::template Dispatch<D>(min(D, max(count - Pw, 0)), [&](auto c) {
          Core::template Prefix<decltype(c)::value>(pf, ring, segment, p.gemm, s, l, count, acc, ss);
        });
        consumed = min(Pw, count);
      } else if constexpr (First && Pw == 0) {
        Core::template Dispatch<D>(min(D, count), [&](auto c) {
          Core::template Prime<decltype(c)::value>(ring, segment, p.gemm, s, l, 0);
        });
      }
      int const left = count - consumed;
      Core::template Steady<First>(ring, segment, consumed, left, p.gemm, s, l, acc, ss);
      Core::template TailLoop<First>(ring, segment, consumed, left, p.gemm, s, l, acc, ss);
      auto const following = geo.Segment(round + 1, l.group);
      int const next_count = round + 1 < geo.rounds ? MxWaveChunksV2(following, l.wave_k, kWK) : 0;
      Core::template Dispatch<D>(min(D, next_count), [&](auto c) {
        Core::template Prime<decltype(c)::value>(ring, following, p.gemm, s, l, 0);
      });
    } else Core::template Prime<0>(ring, segment, p.gemm, s, l, 0);
    if constexpr (First) {
      if (l.Compute()) shared[layout.squares + l.wave * 64 + l.lane] = ss;
    }
    if constexpr (kWK == 1) {
      if (l.Compute() && segment.active) {
        if constexpr (First) {
          __syncwarp();
          float const* sq = shared + layout.squares + l.wave * 64 + l.r;
          inverse = maca::Rsqrt(((sq[0] + sq[32]) + (sq[16] + sq[48])) / float(s.k) + p.gemm.epsilon);
        }
        Accumulate(acc, inverse, best, segment.tile, s.b, l);
      }
    } else {
      int const rounds = MxReduceRounds(s.b, TN), phase_waves = rounds == 1 ? 8 : 4;
      for (int phase = 0; phase < rounds; ++phase) {
        Sync::CtaSync();
        bool const selected = l.Compute() && l.wave >= phase * phase_waves && l.wave < (phase + 1) * phase_waves;
        if (selected && l.r < s.b) {
#pragma unroll
          for (int j = 0; j < kAtoms; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
              shared[((l.wave - phase * phase_waves) * s.b + l.r) * TN + j * 16 + 4 * l.g + e] = acc[j][e];
        }
        Sync::CtaSync();
        if (selected && !l.wave_k && segment.active) {
          if constexpr (First) {
            float sum = 0;
#pragma unroll
            for (int w = 0; w < 2; ++w)
#pragma unroll
              for (int g = 0; g < 4; ++g)
                sum += shared[layout.squares + (l.group * 2 + w) * 64 + 16 * g + l.r];
            inverse = maca::Rsqrt(sum / float(s.k) + p.gemm.epsilon);
          }
#pragma unroll
          for (int j = 0; j < kAtoms; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              int const index = ((l.group * 2 - phase * phase_waves) * s.b + l.r) * TN + j * 16 + 4 * l.g + e;
              acc[j][e] = (0.f + shared[index]) + shared[index + s.b * TN];
            }
          Accumulate(acc, inverse, best, segment.tile, s.b, l);
        }
        Sync::CtaSync();
      }
    }
  }

  __device__ __forceinline__ static void Accumulate(MmaAccum const (&acc)[kAtoms], float inverse,
                                                    MxHeadPartialV2& best, int tile, int batch, Lane const& l) {
    if (l.r < batch) {
#pragma unroll
      for (int j = 0; j < kAtoms; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          best = MxBetterV2(best, {float(Bf16(acc[j][e] * inverse)), unsigned(tile * TN + j * 16 + 4 * l.g + e)});
    }
  }

  template <class Shape, class Next>
  __device__ __forceinline__ static auto Run(Prefetched pf, MxHeadOperandsV2 const& p,
                                            Shape const& s, char* smem, Next next) {
    using Sync = codegen::mx::StageSyncV2<Shape::kThreads>;
    Lane const l;
    MxGeometryV2<TN, WN, MxPart::kTileRange> const geo(s.n, s.k, int(gridDim.x), StageBlock());
    MxGemmSmemV2 const layout(s.b, TN, WN);
    float* const shared = reinterpret_cast<float*>(smem);
    Ring ring{pf.w};
    float inverse = 1.f;
    MxHeadPartialV2 best{-INFINITY, ~0u};
    if (geo.rounds) Round<true>(pf, ring, inverse, best, p, s, shared, layout, geo, l, 0);
#pragma unroll 1
    for (int round = 1; round < geo.rounds; ++round)
      Round<false>(pf, ring, inverse, best, p, s, shared, layout, geo, l, round);
    Sync::CtaSync();
    auto* const pairs = reinterpret_cast<MxHeadPartialV2*>(smem);
    if (l.Compute() && l.r < s.b) pairs[l.r * 32 + l.wave * 4 + l.g] = best;
    Sync::CtaSync();
    for (int stride = 16; stride; stride /= 2) {
      if (l.Compute() && l.t < s.b * stride) {
        int const row = l.t / stride, col = l.t % stride;
        pairs[row * 32 + col] = MxBetterV2(pairs[row * 32 + col], pairs[row * 32 + col + stride]);
      }
      Sync::CtaSync();
    }
    if (l.Compute() && l.t < s.b) MxHeadStoreV2(p.partial + l.t * gridDim.x + blockIdx.x, pairs[l.t * 32]);
    auto* const flag = reinterpret_cast<unsigned*>(smem + SmemBytes(s.b) - 16);
    codegen::executor::LastArriverV2<Sync>::Run(p.ticket, unsigned(gridDim.x), flag, [&] {
      if (l.Compute()) {
        // All B*G pairs are issued in parallel; padding gives a fixed tree.
        for (int index = l.t; index < s.b * 128; index += 512) {
          int const row = index / 128, col = index % 128;
          pairs[index] = col < int(gridDim.x) ? MxHeadLoadV2(p.partial + row * gridDim.x + col)
                                             : MxHeadPartialV2{-INFINITY, ~0u};
        }
      }
      Sync::CtaSync();
      for (int stride = 64; stride; stride /= 2) {
        if (l.Compute())
          for (int index = l.t; index < s.b * stride; index += 512) {
            int const row = index / stride, col = index % stride;
            pairs[row * 128 + col] = MxBetterV2(pairs[row * 128 + col], pairs[row * 128 + col + stride]);
          }
        Sync::CtaSync();
      }
      if (l.Compute() && l.t < s.b) p.tokens[l.t * p.capacity + p.past + 1] = int(pairs[l.t * 128].index);
    });
    Sync::CtaSync();
    using Result = decltype(next());
    if (l.Compute()) return next();
    return Result{};
  }
};

}  // namespace tilemega::backend::mx
