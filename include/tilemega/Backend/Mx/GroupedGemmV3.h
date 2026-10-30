// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Mx/DecodeGemmV2.h>
#include <tilemega/Backend/DmEpilogueValue.h>
#include <tilemega/Backend/Mx/MoeCombineV3.h>
namespace tilemega::backend::mx {
struct MxGroupedUnitV3 {
  unsigned active=0,expert=0,tile=0,count=0,input_row=0;
};
template <bool Fp32>
struct MxGroupedGemmOperandsV3 {
  Bf16 const *a,*w;
  Bf16* out;
  MxSharedRouteTableV3<Fp32>* table;
  unsigned expert_stride;
  float epsilon;
  MxGroupedUnitV3 unit;
  MxMoeCombineOperandsV3<Fp32> combine;
  Bf16 const* router_logits;
#if TILEMEGA_MX_DIAG
  unsigned *route_debug;
  void* route_weights;
  unsigned debug_step,layer,layers;
  bool skip_compute;
#endif
};
template<class Rms,class Gate>
struct MxMoeGateChainV3 {
  static_assert(Rms::kKind==codegen::DmEpilogueKind::kDeferredRMSNorm);
  static_assert(Gate::kKind==codegen::DmEpilogueKind::kGatePair && Gate::kUnit==16);
  __device__ __forceinline__ static Bf16 Apply(float gate,float up,float inverse) {
    DmEpilogueInputs inputs;inputs.rstd=inverse;
    float const g=DmEpilogueValue<arch::Xcore1000,Rms>::Apply(gate,inputs);
    float const u=DmEpilogueValue<arch::Xcore1000,Rms>::Apply(up,inputs);
    return Bf16(DmEpilogueValue<arch::Xcore1000,Gate>::Apply(g,inputs,u));
  }
};
// Pw is derived from geometry/resources, not an additional search dimension.
// The recorded §4.12 difference permits instances with different Pw payloads.
template <int TN, int WN, int D, MxOpV2 Op, bool Norm, class GateChain, int Pw = D, bool Fp32 = false>
struct MxGroupedGemmV3 {
  using Operands = MxGroupedGemmOperandsV3<Fp32>;
  __device__ __forceinline__ static unsigned Units(Operands const& p) {
    return __builtin_mxc_readfirstlane(Op == MxOpV2::kSwiGLU ? p.table->units_gate_up : p.table->units_down);
  }
  static constexpr MxPart Part = MxPart::kTileRange;
  static constexpr int kAtoms = TN / 16, kWK = 8 / WN, kGroupThreads = kWK * 64;
  static constexpr int kLoads = kAtoms + 1, kTN = TN, kWN = WN, kDepth = D, kPrefetch = Pw;
  static_assert(TN == 16 || TN == 32 || TN == 64 || TN == 128);
  static_assert(WN == 1 || WN == 2 || WN == 4 || WN == 8);
  static_assert(D >= 1 && D <= 4 && MxDepthFits(TN, D, 192));
  static_assert(Part != MxPart::kStreamK || WN == 1);
  static_assert(Op == MxOpV2::kStore || Op == MxOpV2::kSwiGLU);
  static_assert(Norm == (Op == MxOpV2::kSwiGLU));
  static_assert(Op != MxOpV2::kSwiGLU || TN >= 32);
  static_assert(Pw == 0 || Pw >= D);
  static_assert((D - 1) * kLoads <= 63);

  struct Prefetched {
    static constexpr int kLoads = Pw * kAtoms;
    static constexpr int kStorage = Pw > D ? Pw : D;
    uint4 w[kStorage][kAtoms];
    MxGroupedUnitV3 unit;
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
  struct Ring { uint4 (&w)[Prefetched::kStorage][kAtoms]; uint4 a[D]; MxGroupedUnitV3 unit; };
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
  __device__ __forceinline__ static MxGroupedUnitV3 Unit(Operands const& p,
      Shape const& s, MxSegmentV2 segment, Lane const& l) {
    MxGroupedUnitV3 result{};
    if(l.Compute() && segment.active) {
      int const tiles=s.n/TN;
      result.active=segment.tile/tiles;
      result.tile=segment.tile%tiles;
      result.expert=p.table->expert[result.active];
      result.count=p.table->count[result.active];
      unsigned const row=min(unsigned(l.r),result.count-1);
      if constexpr(Op==MxOpV2::kSwiGLU)
        result.input_row=p.table->row[result.active][row]>>3;
      else result.input_row=result.active*s.b+row;
    }
    return result;
  }
  template <class Shape>
  __device__ __forceinline__ static Bf16 const* ARow(Operands const& p, Shape const& s, int) {
    return p.a+std::size_t(p.unit.input_row)*s.k;
  }
  template <class Shape>
  __device__ __forceinline__ static Bf16* OutputRow(Operands const& p, Shape const& s, int row) {
    if constexpr(Op==MxOpV2::kSwiGLU)
      return p.out+std::size_t(p.unit.active*s.b+row)*(s.n/2);
    else {
      unsigned const packed=p.table->row[p.unit.active][row];
      return p.out+std::size_t(packed)*s.n;
    }
  }
  __device__ __forceinline__ static void Weight(uint4 (&w)[kAtoms], Operands const& p,
                                               int kc, int tile, int chunk, Lane const& l) {
    Bf16 const* base = p.w + std::size_t(p.unit.expert)*p.expert_stride + (std::size_t(p.unit.tile)*kc+chunk)*TN*32 + l.lane*8;
#pragma unroll
    for (int j = 0; j < kAtoms; ++j) w[j] = LoadGlobal16Gvm(base + j * 512);
  }
  template <class Shape>
  __device__ __forceinline__ static Prefetched Prefetch(Operands const& p, Shape const& s) {
    static_assert(Shape::kThreads == 512);
    Lane const l(Shape::kThreads);
    Prefetched pf{};
    if constexpr(Pw>0) {
      MxGeometryV2<TN,WN,Part> const geo(Units(p)*TN,s.k,int(gridDim.x),StageBlock());
      auto const segment=geo.Segment(0,l.group);
      pf.unit=Unit(p,s,segment,l);
      if(__builtin_mxc_readfirstlane(int(l.Compute()&&segment.active&&MxWaveChunksV2(segment,l.wave_k,kWK)>0))) {
        auto bound=p;bound.unit=pf.unit;
#pragma unroll
        for(int i=0;i<Pw;++i)Weight(pf.w[i],bound,geo.kc,segment.tile,Chunk(segment,l,i),l);
        asm volatile("":::"memory");
      }
    }
    return pf;
  }

  template <int Slot, class Shape>
  __device__ __forceinline__ static void Load(Ring& ring, MxSegmentV2 segment, int i,
                                             Operands const& p, Shape const& s, Lane const& l) {
    int const chunk = Chunk(segment, l, i);
    Weight(ring.w[Slot], p, s.k / 32, segment.tile, chunk, l);
    ring.a[Slot] = LoadGlobal16Gvm(ARow(p, s, l.r) + chunk * 32 + 8 * l.g);
    asm volatile("" ::: "memory");
  }
  template <int Count, class Shape>
  __device__ __forceinline__ static void Prime(Ring& ring, MxSegmentV2 segment,
                                              Operands const& p, Shape const& s, Lane const& l, int base) {
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
                                                         MxSegmentV2 segment, Operands const& p,
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
                                               Operands const& p, Shape const& s, Lane const& l,
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
        Consume(pf.w[i], a[i], acc, ss, unsigned(i < count && l.r < int(p.unit.count)));
        if constexpr (i >= start && i - start < Ahead)
          Load<i - start>(ring, segment, Pw + i - start, p, s, l);
      }()), ...);
    }(std::make_index_sequence<Pw>{});
  }

  template <bool Statistics = Norm, class Shape>
  __device__ __forceinline__ static void Steady(Ring& ring, MxSegmentV2 segment, int base, int count,
                                                 Operands const& p, Shape const& s, Lane const& l,
                                                 MmaAccum (&acc)[kAtoms], float& ss) {
    int const full = count / D;
#pragma unroll 1
    for (int group = 0; group + 1 < full; ++group) {
      [&]<std::size_t... I>(std::index_sequence<I...>) {
        (([&] {
          __builtin_mxc_arrive_gvmcnt((D - 1) * kLoads);
          Consume<Statistics>(ring.w[I], ring.a[I], acc, ss, unsigned(l.r < int(p.unit.count)));
          Load<int(I)>(ring, segment, base + (group + 1) * D + int(I), p, s, l);
        }()), ...);
      }(std::make_index_sequence<D>{});
    }
  }

  // Dispatch only wait immediates; sharing the data path avoids duplicate rings.
  template <bool Statistics = Norm, class Shape>
  __device__ __forceinline__ static void TailLoop(Ring& ring, MxSegmentV2 segment, int base, int count,
                                                Operands const& p, Shape const& s, Lane const& l,
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
          Consume<Statistics>(ring.w[i], ring.a[i], acc, ss, unsigned(l.r < int(p.unit.count)));
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
          Consume<Statistics>(ring.w[i], ring.a[i], acc, ss, unsigned(l.r < int(p.unit.count)));
        }
      }()), ...);
    }(std::make_index_sequence<D>{});
  }

  template <class Shape>
  __device__ __forceinline__ static Residual Early(MxSegmentV2, Operands const&,
      Shape const&, Lane const&) { return {}; }
  template <class Shape>
  __device__ __forceinline__ static void Epilogue(float const* result, Residual const& residual,
                                                 MxSegmentV2 segment, Operands const& p,
                                                 Shape const& s, Lane const& l) {
    if (l.Compute() && segment.active) {
      int const stride = TN + 1, vectors = Op == MxOpV2::kSwiGLU ? TN / 16 : TN / 8;
#pragma unroll
      for (int i = 0; i < kResidualVectors; ++i) {
        int const item = l.local + i * kGroupThreads;
        if (item < int(p.unit.count) * vectors) {
          int const row = item / vectors, c8 = item % vectors * 8;
          float const inverse = Norm ? maca::Rsqrt(result[row * stride + TN] / float(s.k) + p.epsilon) : 1.f;
          alignas(16) Bf16 output[8], old[8];
          if constexpr (Op == MxOpV2::kResidual) *reinterpret_cast<uint4*>(old) = residual.vector[i];
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            if constexpr (Op == MxOpV2::kSwiGLU) {
              int const gate = 32 * (c8 / 16) + c8 % 16 + e;
              output[e] = GateChain::Apply(result[row*stride+gate],result[row*stride+gate+16],inverse);
            } else if constexpr (Op == MxOpV2::kResidual)
              output[e] = MxResidualV2(result[row * stride + c8 + e] * inverse, old[e]);
            else output[e] = Bf16(result[row * stride + c8 + e] * inverse);
          }
          int const width = Op == MxOpV2::kSwiGLU ? s.n / 2 : s.n;
          MxStore16V2(OutputRow(p,s,row) + p.unit.tile * (Op == MxOpV2::kSwiGLU ? TN / 2 : TN) + c8,
                       *reinterpret_cast<uint4 const*>(output));
        }
      }
    }
  }

  template <class Shape>
  __device__ __forceinline__ static void Direct(MmaAccum const (&acc)[kAtoms], float inverse,
                                               Residual const& residual, MxSegmentV2 segment,
                                               Operands const& p, Shape const& s, Lane const& l) {
    if (segment.active && l.r < int(p.unit.count)) {
#pragma unroll
      for (int j = 0; j < (Op == MxOpV2::kSwiGLU ? kAtoms / 2 : kAtoms); ++j) {
        alignas(8) Bf16 output[4], old[4];
        if constexpr (Op == MxOpV2::kResidual) *reinterpret_cast<uint2*>(old) = residual.fragment[j];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          if constexpr (Op == MxOpV2::kSwiGLU) output[e] = GateChain::Apply(acc[2*j][e],acc[2*j+1][e],inverse);
          else if constexpr (Op == MxOpV2::kResidual) output[e] = MxResidualV2(acc[j][e] * inverse, old[e]);
          else output[e] = Bf16(acc[j][e] * inverse);
        }
        int const width = Op == MxOpV2::kSwiGLU ? s.n / 2 : s.n;
        MxStore8V2(OutputRow(p,s,l.r) + p.unit.tile * (Op == MxOpV2::kSwiGLU ? TN / 2 : TN)
                     + j * 16 + 4 * l.g, *reinterpret_cast<uint2 const*>(output));
      }
    }
  }

  template <class Shape>
  __device__ __forceinline__ static void Finish(MmaAccum const (&acc)[kAtoms], float ss,
                                               Residual const& residual, MxSegmentV2 segment,
                                               Operands const& p, Shape const& s, Lane const& l,
                                               float* shared, MxGemmSmemV2 layout,
                                               MxGeometryV2<TN, WN, Part> const& geo) {
    using Sync = typename Shape::Sync;
    if (l.Compute()) shared[layout.squares + l.wave * 64 + l.lane] = ss;
    int const rounds = MxReduceRounds(s.b, TN), phase_waves = rounds == 1 ? 8 : 4, stride = TN + 1;
    if constexpr (WN > 1) {
      // A whole wave group lies in one phase. Its result workspace can be
      // reused by a later phase after every lane has finished the epilogue.
      float* const phase_result = shared + layout.result + (l.group % layout.result_slots) * s.b * stride;
      for (int phase = 0; phase < rounds; ++phase) {
        Sync::LdsSync();
        bool const selected = l.Compute() && l.wave >= phase * phase_waves && l.wave < (phase + 1) * phase_waves;
        if (selected && l.r < int(p.unit.count)) {
#pragma unroll
          for (int j = 0; j < kAtoms; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
              shared[((l.wave - phase * phase_waves) * s.b + l.r) * TN + j * 16 + 4 * l.g + e] = acc[j][e];
        }
        Sync::LdsSync();
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
        Sync::LdsSync();
        MxSegmentV2 own = segment;
        own.active = own.active && selected;
        Epilogue(phase_result, residual, own, p, s, l);
        Sync::LdsSync();
      }
      return;
    }
    float* const result = shared + layout.result + l.group * s.b * stride;
    for (int phase = 0; phase < rounds; ++phase) {
      Sync::LdsSync();
      if (l.Compute() && l.wave >= phase * phase_waves && l.wave < (phase + 1) * phase_waves && l.r < int(p.unit.count)) {
#pragma unroll
        for (int j = 0; j < kAtoms; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            shared[((l.wave - phase * phase_waves) * s.b + l.r) * TN + j * 16 + 4 * l.g + e] = acc[j][e];
      }
      Sync::LdsSync();
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
      Sync::LdsSync();
    }
    if (l.Compute() && l.local < s.b) {
      float sum = 0;
      if constexpr (Norm)
        for (int w = 0; w < kWK; ++w)
          for (int g = 0; g < 4; ++g) sum += shared[layout.squares + (l.group * kWK + w) * 64 + 16 * g + l.local];
      result[l.local * stride + TN] = sum;
    }
    Sync::LdsSync();
    Epilogue(result,residual,segment,p,s,l);
  }

  template <bool First, class Shape>
  __device__ __forceinline__ static void Round(Prefetched const& pf, Ring& ring, float& inverse,
                                              Operands const& operands, Shape const& s, float* shared,
                                              MxGemmSmemV2 layout, MxGeometryV2<TN, WN, Part> const& geo,
                                              Lane const& l, int round, int& inverse_expert) {
      auto p=operands;p.unit=ring.unit;
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
        Steady<Norm>(ring, segment, consumed, left, p, s, l, acc, ss);
        TailLoop<Norm>(ring, segment, consumed, left, p, s, l, acc, ss);
        auto following=geo.Segment(round+1,l.group);
        following.active=following.active&&round+1<geo.rounds;
        int const next_count=round+1<geo.rounds?MxWaveChunksV2(following,l.wave_k,kWK):0;
        auto following_ops=operands;
        following_ops.unit=Unit(operands,s,following,l);
        ring.unit=following_ops.unit;
        Dispatch<D>(min(D,next_count),[&](auto c){Prime<decltype(c)::value>(ring,following,following_ops,s,l,0);});
        if constexpr (kWK == 1) {
          if constexpr (Norm) {
            if (inverse_expert != int(p.unit.active)) {
              shared[layout.squares + l.wave * 64 + l.lane] = ss;
              __syncwarp();
              float const* square = shared + layout.squares + l.wave * 64 + l.r;
              float const sum = (square[0] + square[32]) + (square[16] + square[48]);
              inverse = maca::Rsqrt(sum / float(s.k) + p.epsilon);
              inverse_expert=int(p.unit.active);
            }
          }
          Direct(acc, inverse, residual, segment, p, s, l);
        }
      }
      else Prime<0>(ring, segment, p, s, l, 0);
      if constexpr (kWK > 1) Finish(acc, ss, residual, segment, p, s, l, shared, layout, geo);
  }

  template <class Shape, class Next>
  __device__ __forceinline__ static auto Run(Prefetched pf, Operands const& p,
                                            Shape const& s, char* smem, Next next) {
    using Sync = typename Shape::Sync;
    static_assert(Shape::kThreads == 512);
    using NextPrefetched = decltype(next());
    static_assert((D - 1) * kLoads + NextPrefetched::kLoads <= 63);
    Lane const l(Shape::kThreads);
    MxGeometryV2<TN, WN, Part> const geo(Units(p)*TN, s.k, int(gridDim.x), StageBlock());
    MxGemmSmemV2 const layout(s.b, TN, WN);
    float* const shared = reinterpret_cast<float*>(smem);
    Ring ring{pf.w,{},pf.unit};
    if constexpr(Pw==0)ring.unit=Unit(p,s,geo.Segment(0,l.group),l);
    float inverse = 1.f;
    int inverse_expert=-1;
    if (geo.rounds) Round<true>(pf, ring, inverse, p, s, shared, layout, geo, l, 0, inverse_expert);
#pragma unroll 1
    for (int round = 1; round < geo.rounds; ++round)
      Round<false>(pf, ring, inverse, p, s, shared, layout, geo, l, round, inverse_expert);
    if constexpr (Op == MxOpV2::kStore)
      MxMoeCombineTileV3<TN,Fp32>::Complete(p.combine,geo.begin,geo.end,
          p.table->n_active,s.b,s.n,smem);
    else Sync::CtaSync();
    if (l.Compute()) return next();
    return NextPrefetched{};
  }
};

template <class Core, int B, int Hidden, int Intermediate, int TnGate, int TnDown,
          int Experts, bool NormalizeTopK>
struct MxMoeGateUpV3 : Core {
  using Prefetched = typename Core::Prefetched;
  template <class Shape, class Next>
  __device__ __forceinline__ static auto Run(Prefetched pf, MxGroupedGemmOperandsV3<false> const& p,
                                            Shape const& s, char* smem, Next next) {
    using Build = MxBuildRouteTableV3<B,Hidden,Intermediate,TnGate,TnDown,Experts,NormalizeTopK>;
    Build::Run(p.router_logits,p.table,smem
#if TILEMEGA_MX_DIAG
        ,p.route_debug,p.route_weights,p.debug_step,p.layer,p.layers
#endif
    );
#if TILEMEGA_MX_DIAG
    // Down still requires this layer's table when GU arithmetic is skipped.
    if (p.skip_compute) {
      Shape::Sync::CtaSync();
      return next();
    }
#endif
    return Core::Run(pf,p,s,smem,next);
  }
};

}
