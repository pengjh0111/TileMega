// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Backend/Mx/DecodeGemm.h
//
// MX-4 decode GEMM stage body (MX4 prompt §3.5): Y[B][N] = op(X[B][K] W^T)
// for B <= 16 rows in one 16-row MMA atom, run by every CTA of the static
// decode program (G CTAs x 512 threads, one CTA per AP).
//
// W is packed tile_pages(xcore_mma, TN, 32): each (tile, 32-K chunk) is TN/16
// consecutive 1 KiB wave loads (ServingGemmMma.h FragmentOffset with TK=32),
// and a tile's chunks are adjacent. Work is the unit u = t*Kc + k (tile t,
// chunk k); CTA c owns units [cU/G, (c+1)U/G), which kTileRange rounds to
// whole tiles. A CTA walks its range as segments (one tile, a chunk range);
// its 8 waves interleave over a segment's chunks (wave w: k0+w, k0+w+8, ...).
//
// Main loop (rules 1-6 of §3.5): every global load goes through address
// space 1; the steady loop is straight-line, with out-of-range chunks
// clamped to the segment's last chunk and their A fragment selected to zero;
// D chunks are in flight in a ring with compile-time slots, each slot
// consumed before it is refilled (otherwise the old and new values are live
// together, the allocator copies the pending load at the back edge and the
// MetaX wait inserter drains there -- experiments/mx4/shape_probe.cu). The
// refills of a segment's last group load the next segment's first chunks, so
// they are in flight across the CTA reduction (rule 5).
//
// A segment ends with a fixed-order reduction over the waves in shared
// memory. A whole tile goes straight to the epilogue; otherwise the CTA
// stores its [B][TN+1] partial (the last column is A's sum of squares) and
// the tile's last arriver sums the n(t) partials in contributor order.
#pragma once

#include <tilemega/Backend/Mx/DecodeCommon.h>
#include <tilemega/Backend/Mx/DecodeShapes.h>
#include <tilemega/Backend/Mx/ServingGemmMma.h>
#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Codegen/executor/LastArriver.cuh>
#include <tilemega/Codegen/tasks/mx/AttentionCommon.h>
#include <tilemega/Target/Maca/Bf16.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

namespace tilemega::backend::mx {

/// The epilogue values of the serving GEMM (ServingEpilogue.h
/// ServingEpilogueValue), same expressions and rounding points, force-inlined:
/// the plan build left those as calls with stack frames (measured).
__device__ __forceinline__ tilemega::Bf16 MxResidual(float acc, tilemega::Bf16 residual) {
  auto const rounded = tilemega::Bf16(acc);
  return tilemega::Bf16(float(rounded) + float(residual));
}
__device__ __forceinline__ tilemega::Bf16 MxSwiGLU(float gate, float up) {
  float const g = float(tilemega::Bf16(gate));
  float const v = float(tilemega::Bf16(up));
  auto const activated = tilemega::Bf16(g / (1.0f + expf(-g)));
  return tilemega::Bf16(float(activated) * v);
}

// Tokens per V^T column write (RopeEpilogue): 32 tokens = one 64-byte line.
// Measured on Qwen3-0.6B B16 (TILEMEGA_MX_DIAG trace, per layer): 2-byte
// stores 44 us Qkv + 131 us attention; 8 tokens 53 + 125; 32 tokens 52 + 58;
// 64 tokens 76 + 59. Partial 64-byte lines are what the attention paid for.
#ifndef TILEMEGA_MX_VT_GROUP
#define TILEMEGA_MX_VT_GROUP 32
#endif

template <int D, int kAtoms>
struct MxGemmPrefetch {
  uint4 w[D][kAtoms];
};

template <int TN, int D, MxDecodeOp Op, bool Norm, MxPart Part, class Rope = MxNoRope>
struct MxDecodeGemm {
  using Element = tilemega::Bf16;
  static constexpr int kAtoms = TN / 16, kWaves = kDecodeWaves, kThreads = kDecodeThreads;
  static constexpr int kTN = TN, kDepth = D;
  static constexpr MxPart kPart = Part;
  static_assert(TN == 16 || TN == 32 || TN == 64 || TN == 128, "TN in {16, 32, 64, 128}");
  static_assert(D >= 1 && D <= 4 && MxDepthFits(TN, D), "ring depth 1..4 within 160 VGPRs (rule 6)");
  static_assert(Op != MxDecodeOp::kSwiGLU || TN >= 32, "GateUp: TN in {32, 64, 128}");
  static_assert(Op != MxDecodeOp::kArgmax || Part == MxPart::kTileRange, "Head is kTileRange");
  static_assert(Op != MxDecodeOp::kQkvRope ||
                    ((TN == Rope::kHeadDim || TN == 2 * Rope::kHeadDim) && TN <= 128 &&
                     ((Rope::kQ + 2) * Rope::kHeadDim) % TN == 0),
                "Qkv: TN in {D, 2D}, <= 128, divides (Q+2)D so a tile holds whole heads");
  /// Loads per chunk and lane: TN/16 W fragments and one A fragment.
  static constexpr int kLoads = kAtoms + 1;
  static_assert((D - 1) * kLoads <= 63, "arrive_gvmcnt takes n <= 63");
  using Prefetched = MxGemmPrefetch<D, kAtoms>;

  /// One tile's chunk range [k0, k1).
  struct Segment {
    int t, k0, k1;
  };

  template <class Shape>
  struct Geometry {
    int kc, tiles;
    long long units, begin, end;
    __device__ __forceinline__ Geometry(Shape const& s) : kc(s.k / 32), tiles(s.n / TN) {
      units = (long long)tiles * kc;
      long long const c = UniformOpaque(blockIdx.x), g = gridDim.x;
      if constexpr (Part == MxPart::kStreamK) {
        begin = c * units / g;
        end = (c + 1) * units / g;
      } else {
        begin = c * tiles / g * kc;
        end = (c + 1) * tiles / g * kc;
      }
    }
    __device__ __forceinline__ Segment First() const {
      int const t = int(begin / kc);
      return {t, int(begin - (long long)t * kc), int(min(end - (long long)t * kc, (long long)kc))};
    }
    __device__ __forceinline__ bool HasNext(Segment const& s) const { return (long long)(s.t + 1) * kc < end; }
    /// The next segment, or `s` itself at the end of the range (the ring's
    /// last refills then re-read the segment's last chunk).
    __device__ __forceinline__ Segment Next(Segment const& s) const {
      if (!HasNext(s)) return s;
      return {s.t + 1, 0, int(min(end - (long long)(s.t + 1) * kc, (long long)kc))};
    }
    /// The first CTA covering unit u.
    __device__ __forceinline__ int Owner(long long u) const { return int(((u + 1) * gridDim.x - 1) / units); }
  };

  struct Lane {
    int lane, wave, r, g;
    // readfirstlane: the wave index is uniform over a wave, and saying so
    // keeps every wave-uniform branch below from being treated as
    // divergent (both arms executed, both results live).
    __device__ __forceinline__ Lane()
        : lane(Opaque(threadIdx.x % 64)), wave(UniformOpaque(threadIdx.x / 64)),
          r(lane & 15), g(lane >> 4) {}
  };

  __device__ __forceinline__ static uint4 Load(Element const* p) { return LoadGlobal16Gvm(p); }

  /// This wave's chunk count in a segment.
  __device__ __forceinline__ static int Count(Segment const& s, Lane const& l) {
    int const length = s.k1 - s.k0;
    return length > l.wave ? (length - l.wave + kWaves - 1) / kWaves : 0;
  }
  /// The wave's j-th chunk of a segment, clamped to the segment's last.
  __device__ __forceinline__ static int Chunk(Segment const& s, Lane const& l, int j) {
    return s.k0 + min(l.wave + j * kWaves, s.k1 - s.k0 - 1);
  }
  __device__ __forceinline__ static Element const* WAddress(MxGemmOperands const& p, int kc, int t, int chunk, Lane const& l) {
    return p.w + ((std::size_t(t) * kc + chunk) * TN * 32 + std::size_t(l.lane) * 8);
  }
  template <class Shape>
  __device__ __forceinline__ static Element const* AAddress(MxGemmOperands const& p, Shape const& s, int chunk, Lane const& l) {
    return p.a + std::size_t(min(l.r, s.b - 1)) * s.k + chunk * 32 + 8 * l.g;
  }

  /// The cross-barrier prefetch: this wave's first D chunks of the CTA's
  /// first segment, W only (A is produced by the previous stage).
  template <class Shape>
  __device__ __forceinline__ static Prefetched Prefetch(MxGemmOperands const& p, Shape const& s) {
    Geometry<Shape> const geo(s);
    Lane const l;
    Segment const seg = geo.begin < geo.end ? geo.First() : Segment{0, 0, 1};
    Prefetched pf;
    [&]<std::size_t... k>(std::index_sequence<k...>) {
      (([&] {
         Element const* w = WAddress(p, geo.kc, seg.t, Chunk(seg, l, int(k)), l);
#pragma unroll
         for (int j = 0; j < kAtoms; ++j) pf.w[k][j] = Load(w + j * 512);
         asm volatile("" ::: "memory");
       }()),
       ...);
    }(std::make_index_sequence<D>{});
    return pf;
  }

  /// Squares of A's eight BF16 values.
  __device__ __forceinline__ static float SquareA(uint4 a) {
    unsigned const words[4] = {a.x, a.y, a.z, a.w};
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float const lo = __uint_as_float(words[i] << 16), hi = __uint_as_float(words[i] & 0xffff0000u);
      sum += lo * lo;
      sum += hi * hi;
    }
    return sum;
  }

  template <class Shape>
  __device__ __forceinline__ static void Run(Prefetched const& pf, MxGemmOperands const& p, Shape const& s, char* smem) {
    MxGemmSmem const layout(s.b, TN);
    float* const f = reinterpret_cast<float*>(smem);
    Geometry<Shape> const geo(s);
    Lane const l;
    if constexpr (Op == MxDecodeOp::kArgmax) {
      if (threadIdx.x < s.b) {
        f[layout.best_value + threadIdx.x] = -INFINITY;
        reinterpret_cast<int*>(f)[layout.best_index + threadIdx.x] = INT_MAX;
      }
    }
    // n(t) and the contributor ordinal assume every CTA owns a unit.
    if (Part == MxPart::kStreamK && geo.units < gridDim.x) maca::Trap();
    if (geo.begin < geo.end) {
      MmaAccum acc[kAtoms];
      float ss = 0;
      uint4 w[D][kAtoms], a[D];
      Segment cur = geo.First(), next = geo.Next(cur);
      int count = Count(cur, l), groups = (count + D - 1) / D;
      auto zero = [&] {
#pragma unroll
        for (int j = 0; j < kAtoms; ++j) acc[j] = MmaAccum{0.f, 0.f, 0.f, 0.f};
        ss = 0;
      };
      // Slot k of group g: this segment's chunk g*D+k, valid or not.
      auto consume = [&](auto slot, int g) {
        constexpr int k = decltype(slot)::value;
        // Zero A past the wave's chunks with a mask: a select of the vector
        // was lowered to a branch, a join inside the loop.
        unsigned mask = 0u - unsigned(l.wave + (g * D + k) * kWaves < cur.k1 - cur.k0);
        asm volatile("" : "+r"(mask));
        uint4 const x{a[k].x & mask, a[k].y & mask, a[k].z & mask, a[k].w & mask};
        if constexpr (Norm) ss += l.r < s.b ? SquareA(x) : 0.f;
#pragma unroll
        for (int j = 0; j < kAtoms; ++j) {
          acc[j] = Mma(Lo(x), Lo(w[k][j]), acc[j]);
          acc[j] = Mma(Hi(x), Hi(w[k][j]), acc[j]);
        }
      };
      // Refill slot k after group g: chunk (g+1)*D+k of this segment, or,
      // past the segment's groups, of the next one. Selects, no branch.
      auto refill = [&](auto slot, int g) {
        constexpr int k = decltype(slot)::value;
        int const i = (g + 1) * D + k;
        bool const here = i < groups * D;
        Segment const target{here ? cur.t : next.t, here ? cur.k0 : next.k0, here ? cur.k1 : next.k1};
        int const chunk = Chunk(target, l, here ? i : i - groups * D);
        Element const* wp = WAddress(p, geo.kc, target.t, chunk, l);
#pragma unroll
        for (int j = 0; j < kAtoms; ++j) w[k][j] = Load(wp + j * 512);
        a[k] = Load(AAddress(p, s, chunk, l));
      };
      auto group = [&](int g) {
        [&]<std::size_t... k>(std::index_sequence<k...>) {
          (([&] {
             __builtin_mxc_arrive_gvmcnt((D - 1) * kLoads);
             consume(std::integral_constant<int, int(k)>{}, g);
             refill(std::integral_constant<int, int(k)>{}, g);
           }()),
           ...);
        }(std::make_index_sequence<D>{});
      };
      auto refill_all = [&] {
        [&]<std::size_t... k>(std::index_sequence<k...>) {
          ((refill(std::integral_constant<int, int(k)>{}, -1)), ...);
        }(std::make_index_sequence<D>{});
      };

      zero();
      // The prefetched W, then this segment's A for the same D chunks.
#pragma unroll
      for (int k = 0; k < D; ++k)
#pragma unroll
        for (int j = 0; j < kAtoms; ++j) w[k][j] = pf.w[k][j];
      [&]<std::size_t... k>(std::index_sequence<k...>) {
        (([&] {
           a[k] = Load(AAddress(p, s, Chunk(cur, l, int(k)), l));
           asm volatile("" ::: "memory");
         }()),
         ...);
      }(std::make_index_sequence<D>{});
      int first_group = 0;
      if (count > 0) {
        // First group, peeled: its A loads were issued after all of W, so
        // slot k waits for the later A's and this group's k refills only.
        [&]<std::size_t... k>(std::index_sequence<k...>) {
          (([&] {
             __builtin_mxc_arrive_gvmcnt((D - 1 - int(k)) + int(k) * kLoads);
             consume(std::integral_constant<int, int(k)>{}, 0);
             refill(std::integral_constant<int, int(k)>{}, 0);
           }()),
           ...);
        }(std::make_index_sequence<D>{});
        first_group = 1;
      } else {
        refill_all();  // wave-uniform: no chunk here; load the next segment's
      }
      for (;;) {
#pragma unroll 1
        for (int g = first_group; g < groups; ++g) group(g);
        Finish(acc, ss, cur, geo, l, p, s, f, layout);
        if (!geo.HasNext(cur)) break;
        cur = next;
        next = geo.Next(cur);
        count = Count(cur, l);
        groups = (count + D - 1) / D;
        zero();
        first_group = 0;
        if (count == 0) refill_all();
      }
    }
    if constexpr (Op == MxDecodeOp::kArgmax) HeadFinish(p, s, f, layout);
  }

  /// Segment end: reduce over the waves into res[B][TN+1], then epilogue or
  /// partial + last arriver.
  template <class Shape>
  __device__ __forceinline__ static void Finish(MmaAccum (&acc)[kAtoms], float& ss, Segment const& seg, Geometry<Shape> const& geo,
                                Lane const& lane, MxGemmOperands const& p, Shape const& s, float* f,
                                MxGemmSmem const& layout) {
    // The addresses below do not depend on the main loop, and the compiler
    // computed them all before it, live across it (~100 MTregs at TN=128,
    // measured with experiments/mx4/mir_pressure.py): derive them from lane
    // coordinates it cannot see through, after the loop.
    Lane l = lane;
    asm volatile("" : "+r"(l.lane), "+r"(l.wave), "+r"(l.r), "+r"(l.g));
    int const b = s.b, stride = TN + 1;
    float* const buf = f + layout.buf;
    float* const ssb = f + layout.ss;
    float* const res = f + layout.res;
    auto write = [&](int slot) {
#pragma unroll
      for (int j = 0; j < kAtoms; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int const row = 4 * l.g + i;
          if (row < b) buf[(slot * b + row) * TN + 16 * j + l.r] = acc[j][i];
        }
      if constexpr (Norm) ssb[slot * 64 + l.lane] = ss;
    };
    int waves = kWaves;
    if (MxReduceRounds(b, TN) == 2) {
      // Second half writes; first half adds wave w + W/2 into its registers.
      __syncthreads();
      if (l.wave >= kWaves / 2) write(l.wave - kWaves / 2);
      __syncthreads();
      if (l.wave < kWaves / 2) {
#pragma unroll
        for (int j = 0; j < kAtoms; ++j) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            int const row = 4 * l.g + i;
            if (row < b) acc[j][i] += buf[(l.wave * b + row) * TN + 16 * j + l.r];
          }
          asm volatile("" ::: "memory");  // one atom's loads at a time: the
        }                                 // next segment's ring is live
        if constexpr (Norm) ss += ssb[l.wave * 64 + l.lane];
      }
      waves = kWaves / 2;
    }
    __syncthreads();
    if (l.wave < waves) write(l.wave);
    __syncthreads();
    // These loops run while the next segment's ring is in flight: keep
    // their unrolling (and so their registers) small.
#pragma unroll 1
    for (int index = threadIdx.x; index < b * stride; index += kThreads) {
      int const row = index / stride, col = index % stride;
      float sum = 0;
      if (col < TN) {
        for (int v = 0; v < waves; ++v) sum += buf[(v * b + row) * TN + col];
      } else if constexpr (Norm) {
        // Lane (r, g) holds row r's squares of its K group g.
        for (int v = 0; v < waves; ++v)
          for (int g = 0; g < 4; ++g) sum += ssb[v * 64 + 16 * g + row];
      }
      res[index] = sum;
    }
    __syncthreads();
    if (Part == MxPart::kTileRange || (seg.k0 == 0 && seg.k1 == geo.kc)) {
      Epilogue(res, seg.t, p, s, f, layout);
      return;
    }
    long long const first_unit = (long long)seg.t * geo.kc;
    int const first = geo.Owner(first_unit);
    int const producers = geo.Owner(first_unit + geo.kc - 1) - first + 1;
    int const contributors = MxMaxContributors(geo.tiles, gridDim.x);
    float* const tile = p.partial + std::size_t(seg.t) * contributors * b * stride;
    float* const own = tile + std::size_t(int(blockIdx.x) - first) * b * stride;
#pragma unroll 1
    for (int index = threadIdx.x; index < b * stride; index += kThreads) own[index] = res[index];
    unsigned* const flag = reinterpret_cast<unsigned*>(f + layout.flag);
    codegen::executor::LastArriver::Run(&p.tickets[seg.t], unsigned(producers), flag, [&] {
#pragma unroll 1
      for (int index = threadIdx.x; index < b * stride; index += kThreads) {
        float sum = 0;
#pragma unroll 4
        for (int o = 0; o < producers; ++o) sum += LoadFloatGvm(tile + std::size_t(o) * b * stride + index);
        res[index] = sum;
      }
      __syncthreads();
      Epilogue(res, seg.t, p, s, f, layout);
    });
  }

  template <class Shape>
  __device__ __forceinline__ static float Inverse(float const* res, int row, Shape const& s, float epsilon) {
    if constexpr (Norm) return maca::Rsqrt(res[row * (TN + 1) + TN] / float(s.k) + epsilon);
    else return 1.f;
  }

  /// res[B][TN+1] of tile t, in shared memory, all threads.
  template <class Shape>
  __device__ __forceinline__ static void Epilogue(float const* res, int t, MxGemmOperands const& p, Shape const& s, float* f,
                                  MxGemmSmem const& layout) {
    int const b = s.b, stride = TN + 1;
    if constexpr (Op == MxDecodeOp::kResidual) {
      for (int item = threadIdx.x; item < b * TN / 8; item += kThreads) {
        int const row = item / (TN / 8), c8 = item % (TN / 8) * 8;
        float const inv = Inverse(res, row, s, p.epsilon);
        Element* const out = p.out + std::size_t(row) * s.n + std::size_t(t) * TN + c8;
        alignas(16) Element x[8], y[8];
        *reinterpret_cast<uint4*>(x) = Load(out);
#pragma unroll
        for (int e = 0; e < 8; ++e)
          y[e] = MxResidual(res[row * stride + c8 + e] * inv, x[e]);
        StoreGlobal16(out, *reinterpret_cast<uint4 const*>(y));
      }
      __syncthreads();
    } else if constexpr (Op == MxDecodeOp::kSwiGLU) {
      // Atom 2j is gate, 2j+1 up (interleave_u = 16).
      for (int item = threadIdx.x; item < b * TN / 16; item += kThreads) {
        int const row = item / (TN / 16), c8 = item % (TN / 16) * 8;
        float const inv = Inverse(res, row, s, p.epsilon);
        int const gate = 32 * (c8 / 16) + c8 % 16;
        alignas(16) Element y[8];
#pragma unroll
        for (int e = 0; e < 8; ++e)
          y[e] = MxSwiGLU(res[row * stride + gate + e] * inv, res[row * stride + gate + 16 + e] * inv);
        StoreGlobal16(p.out + std::size_t(row) * (s.n / 2) + std::size_t(t) * (TN / 2) + c8,
                      *reinterpret_cast<uint4 const*>(y));
      }
      __syncthreads();
    } else if constexpr (Op == MxDecodeOp::kQkvRope) {
      RopeEpilogue(res, t, p, s, f, layout);
    } else {
      ArgmaxEpilogue(res, t, p, s, f, layout);
    }
  }

  /// Qkv: v = bf16(acc*inv), per head of D columns; Q heads get q-norm and
  /// RoPE into qkv's Q rows, K heads k-norm and RoPE into the K cache row
  /// `past`, V heads go to the V^T cache column `past`. One thread per (row,
  /// head, 8 dimensions), in RotateRow's arithmetic (AttentionCommon.h); the
  /// head's sum of squares goes through shared memory in WarpSum32's order.
  template <class Shape>
  __device__ __forceinline__ static void RopeEpilogue(float const* res, int t, MxGemmOperands const& p, Shape const& s, float* f,
                                      MxGemmSmem const& layout) {
    using codegen::mx::RotateEight;
    using codegen::mx::SquareEight;
    using codegen::mx::TreeSum32;
    constexpr int kD = Rope::kHeadDim, kQ = Rope::kQ, kLanes = kD / 8, kHeads = TN / kD;
    int const b = s.b, stride = TN + 1, items = b * kHeads * kLanes;
    float* const square = f + layout.square;
    int const item = threadIdx.x, row = item / (kHeads * kLanes), hh = item / kLanes % kHeads;
    int const d = item % kLanes * 8, partner = (d + kD / 2) % kD;
    int const head = t * kHeads + hh, group = head / (kQ + 2), j = head % (kQ + 2);
    bool const active = item < items;
    alignas(16) Element x[8], y[8], w[8], partner_w[8], cos[8], sin[8];
    float inv = 1;
    if (active) {
      inv = Inverse(res, row, s, p.epsilon);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        x[e] = Element(res[row * stride + hh * kD + d + e] * inv);
        y[e] = Element(res[row * stride + hh * kD + partner + e] * inv);
      }
      if constexpr (Rope::kQkNorm) square[item] = SquareEight(x);
    }
    if constexpr (Rope::kQkNorm) __syncthreads();
    if (active) {
      std::size_t const bg = std::size_t(row) * Rope::kHeadsKv + group;
      if (j == kQ + 1) {
        // V heads: the loop below.
      } else {
        float inverse = 1;
        if constexpr (Rope::kQkNorm) {
          float const* const head_squares = square + (item - item % kLanes);
          float const sum = TreeSum32([&](int lane) {
            float const v = lane < kLanes ? head_squares[lane] : 0.f;
            asm volatile("" ::: "memory");  // one pair of loads at a time
            return v;
          });
          inverse = maca::Rsqrt(sum / kD + p.epsilon);
          Element const* norm = j < kQ ? p.q_norm : p.k_norm;
          *reinterpret_cast<uint4*>(w) = Load(norm + d);
          *reinterpret_cast<uint4*>(partner_w) = Load(norm + partner);
        }
        *reinterpret_cast<uint4*>(cos) = Load(p.cosine + std::size_t(p.past) * kD + d);
        *reinterpret_cast<uint4*>(sin) = Load(p.sine + std::size_t(p.past) * kD + d);
        RotateEight<Rope::kQkNorm>(x, y, w, partner_w, cos, sin, inverse, d < kD / 2);
        Element* const target = j < kQ ? p.out + (bg * (kQ + 2) + j) * kD + d
                                       : p.k_cache + (bg * p.capacity + p.past) * kD + d;
        StoreGlobal16(target, *reinterpret_cast<uint4 const*>(x));
      }
    }
    // V heads -> V^T[b][g][d][past]: each dimension's tokens are contiguous.
    // The column is written as the aligned group of TILEMEGA_MX_VT_GROUP
    // tokens (one 64-byte line) that holds it, read, modified and stored
    // whole, one (row, d) per thread: as 2-byte stores (B x hkv x D of them,
    // 16 K at B=16) the partial lines cost the next stage, the attention that
    // streams this cache, ~80 us per layer (measured, TILEMEGA_MX_DIAG). Only
    // this thread writes the row this step, and the attention reads it after
    // the barrier.
#if !TILEMEGA_MX_DIAG_NO_VT
    constexpr int kGroup = TILEMEGA_MX_VT_GROUP;  // tokens per read-modify-write group
    static_assert(kGroup % 8 == 0 && kGroup <= 64, "whole 16-byte vectors");
    int const at = p.past / kGroup * kGroup, slot = p.past % kGroup;
    for (int index = threadIdx.x; index < b * kHeads * kD; index += kThreads) {
      int const vrow = index / (kHeads * kD), vh = index / kD % kHeads, dd = index % kD;
      int const vhead = t * kHeads + vh;
      if (vhead % (kQ + 2) != kQ + 1) continue;  // uniform over a wave: kD >= 64 threads per head
      std::size_t const vbg = std::size_t(vrow) * Rope::kHeadsKv + vhead / (kQ + 2);
      Element* const target = p.vt_cache + (vbg * kD + dd) * p.capacity + at;
      Element const value = Element(res[vrow * stride + vh * kD + dd] * Inverse(res, vrow, s, p.epsilon));
      uint4 words[kGroup / 8];
#pragma unroll
      for (int q = 0; q < kGroup / 8; ++q) words[q] = Load(target + 8 * q);
#pragma unroll
      for (int q = 0; q < kGroup / 8; ++q) {
        alignas(16) Element v[8];
        *reinterpret_cast<uint4*>(v) = words[q];
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = 8 * q + e == slot ? value : v[e];  // no runtime index: registers
        StoreGlobal16(target + 8 * q, *reinterpret_cast<uint4 const*>(v));
      }
    }
#endif
    __syncthreads();
  }

  /// Head tile: v = float(bf16(acc*inv)); the CTA's per-row best (value,
  /// smallest index on ties) in shared memory.
  template <class Shape>
  __device__ __forceinline__ static void ArgmaxEpilogue(float const* res, int t, MxGemmOperands const& p, Shape const& s, float* f,
                                        MxGemmSmem const& layout) {
    int const b = s.b, stride = TN + 1, items = b * TN / 8;
    float* const cand_value = f + layout.cand_value;
    int* const cand_index = reinterpret_cast<int*>(f) + layout.cand_index;
    float* const best_value = f + layout.best_value;
    int* const best_index = reinterpret_cast<int*>(f) + layout.best_index;
    int const item = threadIdx.x;
    if (item < items) {
      int const row = item / (TN / 8), c8 = item % (TN / 8) * 8;
      float const inv = Inverse(res, row, s, p.epsilon);
      float best = -INFINITY;
      int index = INT_MAX;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float const v = float(Element(res[row * stride + c8 + e] * inv));
        if (v > best) {
          best = v;
          index = t * TN + c8 + e;
        }
      }
      cand_value[item] = best;
      cand_index[item] = index;
    }
    __syncthreads();
    if (int(threadIdx.x) < b) {
      int const row = threadIdx.x;
      float best = best_value[row];
      int index = best_index[row];
      for (int c = 0; c < TN / 8; ++c) {
        float const v = cand_value[row * (TN / 8) + c];
        int const i = cand_index[row * (TN / 8) + c];
        if (v > best || (v == best && i < index)) {
          best = v;
          index = i;
        }
      }
      best_value[row] = best;
      best_index[row] = index;
    }
    __syncthreads();
  }

  /// After the CTA's tiles: its (value, index) per row into the head
  /// partial; the last of the G CTAs writes tokens[b][past+1].
  template <class Shape>
  __device__ __forceinline__ static void HeadFinish(MxGemmOperands const& p, Shape const& s, float* f, MxGemmSmem const& layout) {
    int const b = s.b, grid = gridDim.x;
    if (int(threadIdx.x) < b) {
      p.head_value[threadIdx.x * grid + blockIdx.x] = f[layout.best_value + threadIdx.x];
      p.head_index[threadIdx.x * grid + blockIdx.x] = reinterpret_cast<int*>(f)[layout.best_index + threadIdx.x];
    }
    unsigned* const flag = reinterpret_cast<unsigned*>(f + layout.flag);
    codegen::executor::LastArriver::Run(p.head_ticket, unsigned(grid), flag, [&] {
      if (int(threadIdx.x) < b) {
        int const row = threadIdx.x;
        float best = -INFINITY;
        int index = INT_MAX;
        for (int c = 0; c < grid; ++c) {
          float const v = LoadFloatGvm(p.head_value + row * grid + c);
          int const i = *(__attribute__((address_space(1))) int const*)(p.head_index + row * grid + c);
          if (v > best || (v == best && i < index)) {
            best = v;
            index = i;
          }
        }
        p.tokens[std::size_t(row) * p.capacity + p.past + 1] = index;
      }
    });
  }
};

/// §3.7: every thread of every CTA copies 16 B vectors of
/// x[b][:] = embed[tokens[b][past]][:].
template <class Shape>
__device__ __forceinline__ void MxDecodeEmbed(tilemega::Bf16 const* embed, int const* tokens, tilemega::Bf16* x, Shape const& s,
                              int hidden, int capacity, int past) {
  int const vectors = s.b * hidden / 8;
  for (int v = blockIdx.x * kDecodeThreads + threadIdx.x; v < vectors; v += gridDim.x * kDecodeThreads) {
    int const row = v / (hidden / 8), c8 = v % (hidden / 8) * 8;
    int const token = *(__attribute__((address_space(1))) int const*)(tokens + std::size_t(row) * capacity + past);
    StoreGlobal16(x + std::size_t(row) * hidden + c8, LoadGlobal16Gvm(embed + std::size_t(token) * hidden + c8));
  }
}

}  // namespace tilemega::backend::mx
