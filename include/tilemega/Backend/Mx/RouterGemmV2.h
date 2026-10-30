// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Mx/DecodeGemmV2.h>
#include <tilemega/Backend/Mx/MoeDispatchV2.h>
#include <tilemega/Backend/Mx/MoeCombineV2.h>
namespace tilemega::backend::mx {
struct MxMoeRouterOperandsV2 {
  Bf16 const *a,*w,*residual;
  Bf16* out;
  float* partial;
  MxMoeGlobalTicketV2* tickets;
  MxSharedTokensV2 *a_tokens,*residual_tokens;
  int a_stride,residual_stride;
  float epsilon;
  MxMoeGlobalTicketV2* router_ticket;
  MxMoeDispatchOperands dispatch;
};
// Pw is derived from geometry/resources, not an additional search dimension.
// The recorded §4.12 difference permits instances with different Pw payloads.
template <int TN, int D, MxPart Part, class DispatchBody, int Pw = D>
struct MxRouterGemmV2 {
  static constexpr int WN=1;
  static constexpr MxOpV2 Op=MxOpV2::kStore;
  static constexpr bool Norm=true;
  static constexpr int kAtoms = TN / 16, kWK = 8 / WN, kGroupThreads = kWK * 64;
  static constexpr int kLoads = kAtoms + 1, kTN = TN, kWN = WN, kDepth = D, kPrefetch = Pw;
  static_assert(TN == 16 || TN == 32 || TN == 64 || TN == 128);
  static_assert(WN == 1 || WN == 2 || WN == 4 || WN == 8);
  static_assert(D >= 1 && D <= 4 && MxDepthFits(TN, D));
  static_assert(Part != MxPart::kStreamK || WN == 1);
  static_assert(Op != MxOpV2::kSwiGLU || TN >= 32);
  static_assert(Pw == 0 || Pw >= D);
  static_assert((D - 1) * kLoads <= 63);

  struct Prefetched {
    static constexpr int kLoads = Pw * kAtoms;
    static constexpr int kStorage = Pw > D ? Pw : D;
    uint4 w[kStorage][kAtoms];
  };
  struct Lane {
    int t, wave, lane, r, g, group, wave_k, local;
    __device__ __forceinline__ Lane(int threads) {
      unsigned const thread = StageThread();
      // Restore launch bounds hidden by Opaque; retain stream-K's allocation guard.
      __builtin_assume(thread < unsigned(threads));
      t = int(thread); wave = __builtin_mxc_readfirstlane(int(thread / 64));
      if constexpr (Part == MxPart::kTileRange)
        __builtin_assume(unsigned(wave) < unsigned(threads) / 64);
      lane = int(thread % 64); r = lane & 15; g = lane >> 4;
      group = wave < 8 ? wave / kWK : 0;
      wave_k = wave % kWK; local = wave_k * 64 + lane;
    }
    __device__ __forceinline__ bool Compute() const { return wave < 8; }
  };
  // Incoming weights become the ring storage, so register allocation does
  // not have to keep a second D-chunk payload at the first loop boundary.
  struct Ring { uint4 (&w)[Prefetched::kStorage][kAtoms]; uint4 a[D]; };
  static constexpr int kResidualVectors = (16 * TN / 8 + kGroupThreads - 1) / kGroupThreads;
  struct Residual { uint4 vector[kResidualVectors]; uint2 fragment[kAtoms]; };

  template <int Limit, class F>
  __device__ __forceinline__ static void Dispatch(int count, F f) {
    if constexpr (Limit == 0) f(std::integral_constant<int, 0>{});
    else if (__builtin_mxc_readfirstlane(int(count == Limit))) f(std::integral_constant<int, Limit>{});
    else Dispatch<Limit - 1>(count, f);
  }

  __device__ __forceinline__ static int Chunk(MxSegmentV2 s, Lane const& l, int i) {
    return s.k0 + min(l.wave_k + i * kWK, s.k1 - s.k0 - 1);
  }
  template <class Shape>
  __device__ __forceinline__ static Bf16 const* ARow(MxMoeRouterOperandsV2 const& p, Shape const& s, int row) {
    row = min(row, s.b - 1);
    return p.a + std::size_t(p.a_tokens ? p.a_tokens[row] : row) * p.a_stride;
  }
  template <class Shape>
  __device__ __forceinline__ static Bf16 const* ResidualRow(MxMoeRouterOperandsV2 const& p, Shape const& s, int row) {
    row = min(row, s.b - 1);
    return p.residual + std::size_t(p.residual_tokens ? p.residual_tokens[row] : row) * p.residual_stride;
  }
  __device__ __forceinline__ static void Weight(uint4 (&w)[kAtoms], MxMoeRouterOperandsV2 const& p,
                                               int kc, int tile, int chunk, Lane const& l) {
    Bf16 const* base = p.w + (std::size_t(tile) * kc + chunk) * TN * 32 + l.lane * 8;
#pragma unroll
    for (int j = 0; j < kAtoms; ++j) w[j] = LoadGlobal16Gvm(base + j * 512);
  }
  template <class Shape>
  __device__ __forceinline__ static Prefetched Prefetch(MxMoeRouterOperandsV2 const& p, Shape const& s) {
    Lane const l(Shape::kThreads);
    Prefetched pf{};
    if (l.Compute()) {
      MxGeometryV2<TN, WN, Part> const geo(s.n, s.k, int(gridDim.x), StageBlock());
      auto segment = geo.Segment(0, l.group);
      if (__builtin_mxc_readfirstlane(int(segment.active && MxWaveChunksV2(segment, l.wave_k, kWK) > 0))) {
        segment.tile = min(segment.tile, geo.tiles - 1);
        if (!segment.active) segment.k0 = 0, segment.k1 = geo.kc;
#pragma unroll
        for (int i = 0; i < Pw; ++i) Weight(pf.w[i], p, geo.kc, segment.tile, Chunk(segment, l, i), l);
        asm volatile("" ::: "memory");
      }
    }
    return pf;
  }

  template <int Slot, class Shape>
  __device__ __forceinline__ static void Load(Ring& ring, MxSegmentV2 segment, int i,
                                             MxMoeRouterOperandsV2 const& p, Shape const& s, Lane const& l) {
    int const chunk = Chunk(segment, l, i);
    Weight(ring.w[Slot], p, s.k / 32, segment.tile, chunk, l);
    ring.a[Slot] = LoadGlobal16Gvm(ARow(p, s, l.r) + chunk * 32 + 8 * l.g);
    asm volatile("" ::: "memory");
  }
  template <int Count, class Shape>
  __device__ __forceinline__ static void Prime(Ring& ring, MxSegmentV2 segment,
                                              MxMoeRouterOperandsV2 const& p, Shape const& s, Lane const& l, int base) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (([&] {
        if constexpr (int(I) < Count) Load<int(I)>(ring, segment, base + int(I), p, s, l);
        else {
          // Do not hoist an inactive arm's full zero ring across the live one.
          unsigned zero = 0;
          asm volatile("" : "+r"(zero));
#pragma unroll
          for (int j = 0; j < kAtoms; ++j) ring.w[I][j] = uint4{zero, zero, zero, zero};
          ring.a[I] = uint4{zero, zero, zero, zero};
        }
      }()), ...);
    }(std::make_index_sequence<D>{});
  }
  template <class Shape>
  __device__ __forceinline__ static void PrimePrefetched(Prefetched const& pf, Ring& ring,
                                                         MxSegmentV2 segment, MxMoeRouterOperandsV2 const& p,
                                                         Shape const& s, Lane const& l) {
    static_assert(Pw == D);
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (([&] {
#pragma unroll
        for (int j = 0; j < kAtoms; ++j) ring.w[I][j] = pf.w[I][j];
        ring.a[I] = LoadGlobal16Gvm(ARow(p, s, l.r) + Chunk(segment, l, int(I)) * 32 + 8 * l.g);
      }()), ...);
    }(std::make_index_sequence<D>{});
    asm volatile("" ::: "memory");
  }
  __device__ __forceinline__ static float Square(uint4 a) {
    unsigned const words[4] = {a.x, a.y, a.z, a.w};
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float const lo = __uint_as_float(words[i] << 16), hi = __uint_as_float(words[i] & 0xffff0000u);
      sum += lo * lo; sum += hi * hi;
    }
    return sum;
  }
  template <bool Statistics = Norm>
  __device__ __forceinline__ static void Consume(uint4 const (&w)[kAtoms], uint4 a,
                                                MmaAccum (&acc)[kAtoms], float& ss, unsigned valid) {
    unsigned mask = 0u - valid;
    asm volatile("" : "+r"(mask));
    uint4 const x{a.x & mask, a.y & mask, a.z & mask, a.w & mask};
#pragma unroll
    for (int j = 0; j < kAtoms; ++j) {
      acc[j] = Mma(Lo(w[j]), Lo(x), acc[j]);
      acc[j] = Mma(Hi(w[j]), Hi(x), acc[j]);
    }
    if constexpr (Statistics) {
      uint4 stats = x;
      // Every atom's completed MMA precedes scalar statistics expansion.
#pragma unroll
      for (int j = 0; j < kAtoms; ++j)
        asm volatile("" : "+r"(stats.x), "+r"(stats.y), "+r"(stats.z), "+r"(stats.w) : "r"(acc[j][3]));
      ss += Square(stats);
      asm volatile("" :: "r"(ss) : "memory");
    }
  }

  template <int Ahead, class Shape>
  __device__ __forceinline__ static void Prefix(Prefetched const& pf, Ring& ring, MxSegmentV2 segment,
                                               MxMoeRouterOperandsV2 const& p, Shape const& s, Lane const& l,
                                               int count, MmaAccum (&acc)[kAtoms], float& ss) {
    uint4 a[Pw ? Pw : 1];
#pragma unroll
    for (int i = 0; i < Pw; ++i) a[i] = LoadGlobal16Gvm(ARow(p, s, l.r) + Chunk(segment, l, i) * 32 + 8 * l.g);
    asm volatile("" ::: "memory");
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (([&] {
        constexpr int i = int(I), start = Pw - D;
        constexpr int issued = i <= start ? 0 : (i - start < Ahead ? i - start : Ahead);
        constexpr int wait = Pw - 1 - i + issued * kLoads;
        static_assert(wait <= 63);
        __builtin_mxc_arrive_gvmcnt(wait);
        Consume(pf.w[i], a[i], acc, ss, unsigned(i < count && l.r < s.b));
        if constexpr (i >= start && i - start < Ahead)
          Load<i - start>(ring, segment, Pw + i - start, p, s, l);
      }()), ...);
    }(std::make_index_sequence<Pw>{});
  }

  template <bool Statistics = Norm, class Shape>
  __device__ __forceinline__ static void Steady(Ring& ring, MxSegmentV2 segment, int base, int count,
                                                 MxMoeRouterOperandsV2 const& p, Shape const& s, Lane const& l,
                                                 MmaAccum (&acc)[kAtoms], float& ss) {
    int const full = count / D;
#pragma unroll 1
    for (int group = 0; group + 1 < full; ++group) {
      [&]<std::size_t... I>(std::index_sequence<I...>) {
        (([&] {
          __builtin_mxc_arrive_gvmcnt((D - 1) * kLoads);
          Consume<Statistics>(ring.w[I], ring.a[I], acc, ss, unsigned(l.r < s.b));
          Load<int(I)>(ring, segment, base + (group + 1) * D + int(I), p, s, l);
        }()), ...);
      }(std::make_index_sequence<D>{});
    }
  }

  // Dispatch only wait immediates; sharing the data path avoids duplicate rings.
  template <bool Statistics = Norm, class Shape>
  __device__ __forceinline__ static void TailLoop(Ring& ring, MxSegmentV2 segment, int base, int count,
                                                MxMoeRouterOperandsV2 const& p, Shape const& s, Lane const& l,
                                                MmaAccum (&acc)[kAtoms], float& ss) {
    int const full = count / D, tail = count % D;
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (([&] {
        constexpr int i = int(I);
        if (__builtin_mxc_readfirstlane(int(full || i < tail))) {
          Dispatch<D - 1>(tail, [&](auto c) {
            constexpr int t = decltype(c)::value, ahead = i < t ? i : t;
            if (__builtin_mxc_readfirstlane(int(full != 0)))
              __builtin_mxc_arrive_gvmcnt((D - 1 - i + ahead) * kLoads);
            else __builtin_mxc_arrive_gvmcnt((t > i ? t - 1 - i : 0) * kLoads);
          });
          Consume<Statistics>(ring.w[i], ring.a[i], acc, ss, unsigned(l.r < s.b));
          if (__builtin_mxc_readfirstlane(int(full && i < tail)))
            Load<i>(ring, segment, base + full * D + i, p, s, l);
        }
      }()), ...);
    }(std::make_index_sequence<D>{});
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (([&] {
        constexpr int i = int(I);
        if (__builtin_mxc_readfirstlane(int(full && i < tail))) {
          Dispatch<D - 1>(tail, [&](auto c) {
            constexpr int t = decltype(c)::value;
            __builtin_mxc_arrive_gvmcnt((t > i ? t - 1 - i : 0) * kLoads);
          });
          Consume<Statistics>(ring.w[i], ring.a[i], acc, ss, unsigned(l.r < s.b));
        }
      }()), ...);
    }(std::make_index_sequence<D>{});
  }

  template <class Shape>
  __device__ __forceinline__ static Residual Early(MxSegmentV2 segment, MxMoeRouterOperandsV2 const& p,
                                                  Shape const& s, Lane const& l) {
    Residual residual{};
    if constexpr (Op == MxOpV2::kResidual) {
      if (l.Compute() && segment.active) {
        if constexpr (kWK == 1) {
          if (l.r < s.b) {
#pragma unroll
            for (int j = 0; j < kAtoms; ++j)
              residual.fragment[j] = MxLoad8V2(ResidualRow(p, s, l.r) + segment.tile * TN + j * 16 + 4 * l.g);
          }
        } else {
#pragma unroll
          for (int i = 0; i < kResidualVectors; ++i) {
            int const item = l.local + i * kGroupThreads;
            if (item < s.b * TN / 8)
              residual.vector[i] = LoadGlobal16Gvm(ResidualRow(p, s, item / (TN / 8)) + segment.tile * TN + item % (TN / 8) * 8);
          }
        }
      }
    }
    return residual;
  }

  template <class Shape>
  __device__ __forceinline__ static void Epilogue(float const* result, Residual const& residual,
                                                 MxSegmentV2 segment, MxMoeRouterOperandsV2 const& p,
                                                 Shape const& s, Lane const& l) {
    if (l.Compute() && segment.active) {
      int const stride = TN + 1, vectors = Op == MxOpV2::kSwiGLU ? TN / 16 : TN / 8;
#pragma unroll
      for (int i = 0; i < kResidualVectors; ++i) {
        int const item = l.local + i * kGroupThreads;
        if (item < s.b * vectors) {
          int const row = item / vectors, c8 = item % vectors * 8;
          float const inverse = Norm ? maca::Rsqrt(result[row * stride + TN] / float(s.k) + p.epsilon) : 1.f;
          alignas(16) Bf16 output[8], old[8];
          if constexpr (Op == MxOpV2::kResidual) *reinterpret_cast<uint4*>(old) = residual.vector[i];
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            if constexpr (Op == MxOpV2::kSwiGLU) {
              int const gate = 32 * (c8 / 16) + c8 % 16 + e;
              output[e] = MxSwiGLUV2(result[row * stride + gate] * inverse,
                                      result[row * stride + gate + 16] * inverse);
            } else if constexpr (Op == MxOpV2::kResidual)
              output[e] = MxResidualV2(result[row * stride + c8 + e] * inverse, old[e]);
            else output[e] = Bf16(result[row * stride + c8 + e] * inverse);
          }
          int const width = Op == MxOpV2::kSwiGLU ? s.n / 2 : s.n;
          MxStore16V2(p.out + std::size_t(row) * width + segment.tile * (Op == MxOpV2::kSwiGLU ? TN / 2 : TN) + c8,
                       *reinterpret_cast<uint4 const*>(output));
#if TILEMEGA_MX_DIAG
          if(p.dispatch.route_logits && p.dispatch.debug_step<64)
            MxStore16V2(p.dispatch.route_logits+
                ((std::size_t(p.dispatch.debug_step)*p.dispatch.layers+p.dispatch.layer)*s.b+row)*s.n+
                segment.tile*TN+c8,*reinterpret_cast<uint4 const*>(output));
#endif
        }
      }
    }
  }

  template <class Shape>
  __device__ __forceinline__ static void Direct(MmaAccum const (&acc)[kAtoms], float inverse,
                                               Residual const& residual, MxSegmentV2 segment,
                                               MxMoeRouterOperandsV2 const& p, Shape const& s, Lane const& l) {
    if (segment.active && l.r < s.b) {
#pragma unroll
      for (int j = 0; j < (Op == MxOpV2::kSwiGLU ? kAtoms / 2 : kAtoms); ++j) {
        alignas(8) Bf16 output[4], old[4];
        if constexpr (Op == MxOpV2::kResidual) *reinterpret_cast<uint2*>(old) = residual.fragment[j];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          if constexpr (Op == MxOpV2::kSwiGLU) output[e] = MxSwiGLUV2(acc[2 * j][e] * inverse, acc[2 * j + 1][e] * inverse);
          else if constexpr (Op == MxOpV2::kResidual) output[e] = MxResidualV2(acc[j][e] * inverse, old[e]);
          else output[e] = Bf16(acc[j][e] * inverse);
        }
        int const width = Op == MxOpV2::kSwiGLU ? s.n / 2 : s.n;
        MxStore8V2(p.out + std::size_t(l.r) * width + segment.tile * (Op == MxOpV2::kSwiGLU ? TN / 2 : TN)
                     + j * 16 + 4 * l.g, *reinterpret_cast<uint2 const*>(output));
      }
    }
  }

  template <class Shape>
  __device__ __forceinline__ static void Finish(MmaAccum const (&acc)[kAtoms], float ss,
                                               Residual const& residual, MxSegmentV2 segment,
                                               MxMoeRouterOperandsV2 const& p, Shape const& s, Lane const& l,
                                               float* shared, MxGemmSmemV2 layout,
                                               MxGeometryV2<TN, WN, Part> const& geo) {
    using Sync = codegen::mx::StageSyncV2<Shape::kThreads>;
    if (l.Compute()) shared[layout.squares + l.wave * 64 + l.lane] = ss;
    int const rounds = MxReduceRounds(s.b, TN), phase_waves = rounds == 1 ? 8 : 4, stride = TN + 1;
    if constexpr (WN > 1) {
      // A whole wave group lies in one phase. Its result workspace can be
      // reused by a later phase after every lane has finished the epilogue.
      float* const phase_result = shared + layout.result + (l.group % layout.result_slots) * s.b * stride;
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
        if (selected) {
          for (int index = l.local; index < s.b * stride; index += kGroupThreads) {
            int const row = index / stride, col = index % stride;
            float sum = 0;
            if (col < TN) {
              for (int w = 0; w < kWK; ++w)
                sum += shared[((l.group * kWK + w - phase * phase_waves) * s.b + row) * TN + col];
            } else if constexpr (Norm) {
              for (int w = 0; w < kWK; ++w)
                for (int g = 0; g < 4; ++g)
                  sum += shared[layout.squares + (l.group * kWK + w) * 64 + 16 * g + row];
            }
            phase_result[index] = sum;
          }
        }
        Sync::CtaSync();
        MxSegmentV2 own = segment;
        own.active = own.active && selected;
        Epilogue(phase_result, residual, own, p, s, l);
        Sync::CtaSync();
      }
      return;
    }
    float* const result = shared + layout.result + l.group * s.b * stride;
    for (int phase = 0; phase < rounds; ++phase) {
      Sync::CtaSync();
      if (l.Compute() && l.wave >= phase * phase_waves && l.wave < (phase + 1) * phase_waves && l.r < s.b) {
#pragma unroll
        for (int j = 0; j < kAtoms; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            shared[((l.wave - phase * phase_waves) * s.b + l.r) * TN + j * 16 + 4 * l.g + e] = acc[j][e];
      }
      Sync::CtaSync();
      if (l.Compute()) {
        for (int index = l.local; index < s.b * TN; index += kGroupThreads) {
          int const row = index / TN, col = index % TN;
          float sum = phase ? result[row * stride + col] : 0.f;
          for (int w = 0; w < kWK; ++w) {
            int const wave = l.group * kWK + w;
            if (wave >= phase * phase_waves && wave < (phase + 1) * phase_waves)
              sum += shared[((wave - phase * phase_waves) * s.b + row) * TN + col];
          }
          result[row * stride + col] = sum;
        }
      }
      Sync::CtaSync();
    }
    if (l.Compute() && l.local < s.b) {
      float sum = 0;
      if constexpr (Norm)
        for (int w = 0; w < kWK; ++w)
          for (int g = 0; g < 4; ++g) sum += shared[layout.squares + (l.group * kWK + w) * 64 + 16 * g + l.local];
      result[l.local * stride + TN] = sum;
    }
    Sync::CtaSync();
    if constexpr (Part == MxPart::kTileRange) {
      Epilogue(result,residual,segment,p,s,l);
      Complete(p,s,shared,layout);
    }
    else {
      bool const whole = segment.k0 == 0 && segment.k1 == geo.kc;
      if (whole) {
        Epilogue(result,residual,segment,p,s,l);
        Complete(p,s,shared,layout);
      }
      else {
        int const first = geo.FirstOwner(segment.tile), producers = geo.Producers(segment.tile);
        int const slots = MxMaxContributors(geo.tiles, geo.grid);
        float* const base = p.partial + std::size_t(segment.tile) * slots * s.b * stride;
        if (l.Compute())
          for (int index = l.t; index < s.b * stride; index += 512)
            base[std::size_t(geo.cta - first) * s.b * stride + index] = result[index];
        codegen::executor::LastArriverV2<Sync>::Run((unsigned*)(p.tickets + segment.tile), unsigned(producers),
            reinterpret_cast<unsigned*>(shared + layout.flags), [&] {
          if (l.Compute()) {
            for (int index = l.t; index < s.b * stride; index += 512) {
              float sum = 0;
              for (int begin = 0; begin < producers; begin += 32) {
                float values[32];
#pragma unroll
                for (int j = 0; j < 32; ++j)
                  values[j] = LoadFloatGvm(base + std::size_t(min(begin + j, producers - 1)) * s.b * stride + index);
#pragma unroll
                for (int j = 0; j < 32; ++j) if (begin + j < producers) sum += values[j];
              }
              result[index] = sum;
            }
          }
          Sync::CtaSync();
          Epilogue(result, residual, segment, p, s, l);
          Complete(p,s,shared,layout);
        });
      }
    }
  }

  template<class Shape>
  __device__ __forceinline__ static void Complete(MxMoeRouterOperandsV2 const& p,Shape const& s,
      float* shared,MxGemmSmemV2 layout) {
    using Sync=codegen::mx::StageSyncV2<Shape::kThreads>;
    codegen::executor::LastArriverV2<Sync>::Run((unsigned*)p.router_ticket,unsigned(s.n/TN),
        reinterpret_cast<unsigned*>(shared+layout.flags+1),[&] {
          DispatchBody::Run(p.dispatch,reinterpret_cast<char*>(shared));
        });
  }

  template <bool First, class Shape>
  __device__ __forceinline__ static void Round(Prefetched const& pf, Ring& ring, float& inverse,
                                              MxMoeRouterOperandsV2 const& p, Shape const& s, float* shared,
                                              MxGemmSmemV2 layout, MxGeometryV2<TN, WN, Part> const& geo,
                                              Lane const& l, int round) {
      auto const segment = geo.Segment(round, l.group);
      MmaAccum acc[kAtoms];
#pragma unroll
      for (int j = 0; j < kAtoms; ++j) acc[j] = MmaAccum{0.f, 0.f, 0.f, 0.f};
      float ss = 0;
      Residual const residual = Early(segment, p, s, l);
      // This predicate is uniform per wave; a masked branch retains both rings.
      if (__builtin_mxc_readfirstlane(int(l.Compute() && segment.active))) {
        int const count = MxWaveChunksV2(segment, l.wave_k, kWK);
        int consumed = 0;
        if constexpr (First && Pw == D) {
          PrimePrefetched(pf, ring, segment, p, s, l);
        } else if constexpr (First && Pw > 0) {
            int const ahead = min(D, max(count - Pw, 0));
            Dispatch<D>(ahead, [&](auto c) { Prefix<decltype(c)::value>(pf, ring, segment, p, s, l, count, acc, ss); });
            consumed = min(Pw, count);
        } else if constexpr (First && Pw == 0) {
          Dispatch<D>(min(D, count), [&](auto c) { Prime<decltype(c)::value>(ring, segment, p, s, l, 0); });
        }
        int const left = count - consumed;
        Steady<Norm && (First || kWK > 1)>(ring, segment, consumed, left, p, s, l, acc, ss);
        TailLoop<Norm && (First || kWK > 1)>(ring, segment, consumed, left, p, s, l, acc, ss);
        auto const following = geo.Segment(round + 1, l.group);
        int const next_count = round + 1 < geo.rounds ? MxWaveChunksV2(following, l.wave_k, kWK) : 0;
        Dispatch<D>(min(D, next_count), [&](auto c) { Prime<decltype(c)::value>(ring, following, p, s, l, 0); });
        if constexpr (kWK == 1) {
          if constexpr (Norm) {
            if constexpr (First) {
              shared[layout.squares + l.wave * 64 + l.lane] = ss;
              __syncwarp();
              float const* square = shared + layout.squares + l.wave * 64 + l.r;
              float const sum = (square[0] + square[32]) + (square[16] + square[48]);
              inverse = maca::Rsqrt(sum / float(s.k) + p.epsilon);
            }
          }
          Direct(acc, inverse, residual, segment, p, s, l);
        }
      }
      else Prime<0>(ring, segment, p, s, l, 0);
      if constexpr (kWK > 1) Finish(acc, ss, residual, segment, p, s, l, shared, layout, geo);
  }

  template <class Shape, class Next>
  __device__ __forceinline__ static auto Run(Prefetched pf, MxMoeRouterOperandsV2 const& p,
                                            Shape const& s, char* smem, Next next) {
    using Sync = codegen::mx::StageSyncV2<Shape::kThreads>;
    using NextPrefetched = decltype(next());
    static_assert((D - 1) * kLoads + NextPrefetched::kLoads <= 63);
    Lane const l(Shape::kThreads);
    MxGeometryV2<TN, WN, Part> const geo(s.n, s.k, int(gridDim.x), StageBlock());
    MxGemmSmemV2 const layout(s.b, TN, WN);
    float* const shared = reinterpret_cast<float*>(smem);
#if TILEMEGA_MX_DIAG
    if(StageBlock()==0 && l.Compute() && p.dispatch.route_input && p.dispatch.debug_step<64) {
      for(int item=l.t;item<s.b*s.k/8;item+=512) {
        auto const packet=LoadGlobal16Gvm(p.a+item*8);
        __builtin_mxc_arrive_gvmcnt(0);
        MxStore16V2(p.dispatch.route_input+
            (std::size_t(p.dispatch.debug_step)*p.dispatch.layers+p.dispatch.layer)*s.b*s.k+item*8,packet);
      }
    }
#endif
    Ring ring{pf.w};
    float inverse = 1.f;
    if (geo.rounds) Round<true>(pf, ring, inverse, p, s, shared, layout, geo, l, 0);
#pragma unroll 1
    for (int round = 1; round < geo.rounds; ++round)
      Round<false>(pf, ring, inverse, p, s, shared, layout, geo, l, round);
    // P-B: all result stores finish before the next body issues any loads.
    Sync::CtaSync();
    if (l.Compute()) return next();
    return NextPrefetched{};
  }
};

}
