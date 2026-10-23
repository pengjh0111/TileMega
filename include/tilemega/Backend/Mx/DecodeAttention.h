// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Backend/Mx/DecodeAttention.h
//
// MX-4 decode attention stage body (MX4 prompt §3.6). An item is (b, g, c):
// batch row b, KV head g and a chunk c of T_c tokens of L = past + 1. T_c is
// the smallest multiple of 32, >= Tmin and <= 256, that leaves at most G*W
// items; item j runs on CTA j mod G, wave j / G, so the items first spread
// over every AP. Waves do not share data or synchronize until the CTA's end.
//
// Each wave runs its item through MMAs, 32 tokens (two M atoms) per block:
//   S^T[token][q] = K[token] . Q[q]^T   A = K cache rows, B = Q (q >= Q zero)
//   O[q][d] = sum_t P[q][t] V[t][d]     A = P, B = V^T (the vt cache layout)
// Atom 0's row 4g+i holds token 8g+i and atom 1's token 8g+4+i, so lane
// (q, g) of S^T's C fragment holds tokens 8g..8g+7 of query q: exactly PV's A
// fragment (elements 0-3 to the first MMA, 4-7 to the second). P needs no
// shared memory and no shuffle. Pass 1 computes S^T for all of the item's
// blocks (<= 8 blocks, 64 VGPRs) and the row maxima; pass 2 forms P =
// exp2(s - m) and accumulates PV; l sums the FP32 P. The 4 lane groups of a
// query exchange m and l through wave-private shared memory.
//
// C = 1 writes ctx = bf16(O / l). Otherwise each item stores (O, m, l) and
// LastArriver::RunMany lets the CTA that completes a (b, g) merge its C
// partials in chunk order.
#pragma once

#include <tilemega/Backend/Mx/DecodeCommon.h>
#include <tilemega/Backend/Mx/DecodeShapes.h>
#include <tilemega/Backend/Mx/ServingGemmMma.h>
#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Codegen/executor/LastArriver.cuh>
#include <tilemega/Codegen/tasks/mx/AttentionCommon.h>
#include <tilemega/Target/Maca/Bf16.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>

#include <cmath>
#include <cstdint>

namespace tilemega::backend::mx {

struct MxAttentionShape {
  int b, hkv;
};
template <int B, int Hkv>
struct MxStaticAttentionShape {
  static constexpr int b = B, hkv = Hkv;
};

struct MxAttentionOperands {
  tilemega::Bf16 const* qkv;       // [B][hkv][Q+2][D]; Q rows normed and rotated
  tilemega::Bf16 const* k_cache;   // [B][hkv][cap][D]
  tilemega::Bf16 const* vt_cache;  // [B][hkv][D][cap]
  tilemega::Bf16* ctx;             // [B][hkv*Q][D]
  float* partial;                  // [B*hkv][ceil(cap/32)][Q][D+2]: O, m, l
  unsigned* tickets;               // [B*hkv]
  int capacity, past;
};

template <int D, int Q, int Tmin, int MaxBlocks = 8>
struct MxDecodeAttention {
  static_assert(D == 64 || D == 128, "D in {64, 128}");
  static_assert(Q >= 1 && Q <= 8, "Q <= 8");
  static_assert(Tmin >= 1 && Tmin <= 256, "Tmin in [1, 256]");
  static_assert(MaxBlocks >= 1 && MaxBlocks <= 8, "T_c <= 256");
  using Element = tilemega::Bf16;
  static constexpr int kThreads = 512, kWaves = 8;
  static constexpr int kChunks = D / 32;  // 32-dimension MMA chunks of a K row
  static constexpr int kDimAtoms = D / 16;
  static constexpr int kMaxItems = 32;    // per CTA (RunMany flags)
  /// Wave-private exchange (m and l of 16 queries x 4 lane groups) and the
  /// CTA's RunMany flags.
  static constexpr int kWaveBytes = 2 * 4 * 16 * 4;
  static constexpr int kSharedBytes = kWaves * kWaveBytes + kMaxItems * 4;
  static_assert(kWaveBytes <= 2048, "<= 2 KiB per wave (§3.6)");

  struct Prefetched {
    uint4 k[2][kChunks];
    bool valid;
  };

  /// The item geometry of one step, uniform over the grid.
  template <class Shape>
  struct Plan {
    int length, tokens, chunks, items, slots;
    __device__ __forceinline__ Plan(Shape const& s, int past) {
      length = past + 1;
      slots = int(gridDim.x) * kWaves;
      int const heads = s.b * s.hkv;
      tokens = MxAttentionChunkTokens(heads, length, Tmin, slots);
      chunks = (length + tokens - 1) / tokens;
      items = heads * chunks;
    }
  };

  __device__ __forceinline__ static uint4 Load(Element const* p) { return LoadGlobal16Gvm(p); }
  __device__ __forceinline__ static int Wave() { return UniformOpaque(threadIdx.x / 64); }
  __device__ __forceinline__ static int Cta() { return UniformOpaque(blockIdx.x); }

  /// K fragments of one block: lane (r, g) loads, for atom a, the row of
  /// token base + 8(r>>2) + 4a + (r&3) (clamped to end-1), 8 dimensions per
  /// 32-dimension chunk.
  __device__ __forceinline__ static void LoadK(uint4 (&k)[2][kChunks], Element const* head, int base, int end,
                                               int lane) {
    int const r = lane & 15, g = lane >> 4;
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      int const token = min(base + 8 * (r >> 2) + 4 * a + (r & 3), end - 1);
#pragma unroll
      for (int dc = 0; dc < kChunks; ++dc) k[a][dc] = Load(head + std::size_t(token) * D + dc * 32 + 8 * g);
    }
    // Blocks issue in order: the scheduler moved one block's loads ahead of
    // the previous one's, and the wait inserter then drained to < 1 block.
    asm volatile("" ::: "memory");
  }

  template <class Shape>
  __device__ __forceinline__ static Prefetched Prefetch(MxAttentionOperands const& p, Shape const& s) {
    Plan<Shape> const plan(s, p.past);
    Prefetched pf;
    int const j = Cta() + int(gridDim.x) * Wave();
    int const c = j % plan.chunks, head = j / plan.chunks;
    int const base = c * plan.tokens, end = min(base + plan.tokens, plan.length);
    // The block holding `past` was written by the previous stage.
    pf.valid = j < plan.items && !(p.past >= base && p.past < base + 32);
    if (pf.valid) {
      LoadK(pf.k, p.k_cache + std::size_t(head) * p.capacity * D, base, end, threadIdx.x % 64);
    } else {
      // Defined when not loaded: left undefined, the merge took the previous
      // step's registers, which kept the old prefetch live through Run (MIR).
#pragma unroll
      for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int dc = 0; dc < kChunks; ++dc) pf.k[a][dc] = uint4{0u, 0u, 0u, 0u};
    }
    return pf;
  }

  template <class Shape>
  __device__ __forceinline__ static void Run(Prefetched const& pf, MxAttentionOperands const& p, Shape const& s,
                                             char* smem) {
    Plan<Shape> const plan(s, p.past);
    int const wave = Wave(), lane = threadIdx.x % 64, r = lane & 15, g = lane >> 4, cta = Cta();
    float* const exchange = reinterpret_cast<float*>(smem + wave * kWaveBytes);
    unsigned* const flags = reinterpret_cast<unsigned*>(smem + kWaves * kWaveBytes);
    int const c_max = (p.capacity + 31) / 32;
    // The wave's first item takes the prefetch and the body for its block
    // count. More items than G*W slots (none of the MX-4 cells: B*hkv*C <=
    // 640) loop through the MaxBlocks body, which masks the blocks it lacks;
    // a single call site with the prefetch kept it live across the loop and
    // spilled (measured).
    int const first = cta + int(gridDim.x) * wave;
    if (first < plan.items) Item(pf, true, p, s, plan, first, lane, r, g, exchange, c_max);
#pragma unroll 1
    for (int j = first + plan.slots; j < plan.items; j += plan.slots)
      Blocks<MaxBlocks>(pf, false, p, plan, j, lane, r, g, exchange, c_max);
    if (plan.chunks > 1) {
      int const count = plan.items > cta ? (plan.items - cta + int(gridDim.x) - 1) / int(gridDim.x) : 0;
      if (count > kMaxItems) maca::Trap();
      auto item = [&](int i) { return cta + int(gridDim.x) * i; };
      codegen::executor::LastArriver::RunMany(
          count, flags, [&](int i) { return p.tickets + item(i) / plan.chunks; },
          [&](int) { return unsigned(plan.chunks); },
          [&](int i) { Merge(p, s, plan, item(i) / plan.chunks, c_max); });
    }
  }

  /// One item, dispatched once on its block count so that each body is
  /// straight-line: loads are unconditional and the wait inserter sees no
  /// join between them (a guarded load drained the ring, measured in MIR).
  template <class Shape>
  __device__ __forceinline__ static void Item(Prefetched const& pf, bool use_pf, MxAttentionOperands const& p,
                                              Shape const& s, Plan<Shape> const& plan, int j, int lane, int r, int g,
                                              float* exchange, int c_max) {
    int const c = j % plan.chunks, base = c * plan.tokens, end = min(base + plan.tokens, plan.length);
    int const blocks = __builtin_mxc_readfirstlane((end - base + 31) / 32);
    if (blocks > MaxBlocks) maca::Trap();  // more blocks than the plan's MaxBlocks
    Dispatch<MaxBlocks>(blocks, pf, use_pf, p, plan, j, lane, r, g, exchange, c_max);
  }

  /// A uniform if/else chain from the largest body down. The structurizer
  /// linearizes it (a switch with a return per case became a chain too), so
  /// the inputs of every body placed later, the prefetch among them, stay
  /// live through the bodies before it; this order puts the MaxBlocks body,
  /// the one with the most registers, last (MIR: 32 MTregs at its peak).
  template <int N, class Shape>
  __device__ __forceinline__ static void Dispatch(int blocks, Prefetched const& pf, bool use_pf,
                                                  MxAttentionOperands const& p, Plan<Shape> const& plan, int j,
                                                  int lane, int r, int g, float* exchange, int c_max) {
    if constexpr (N >= 1) {
      if (blocks >= N) {  // == N; an == chain was folded back into a switch
        Blocks<N>(pf, use_pf, p, plan, j, lane, r, g, exchange, c_max);
      } else {
        Dispatch<N - 1>(blocks, pf, use_pf, p, plan, j, lane, r, g, exchange, c_max);
      }
    }
  }

  /// V^T mask of lane group g in a block: tokens >= end are zeroed when the
  /// fragment is consumed (the cache past L is uninitialized memory; masking
  /// at the load would wait for it there).
  __device__ __forceinline__ static uint4 MaskV(uint4 x, int block_base, int end, int g) {
    int const valid = min(max(end - (block_base + 8 * g), 0), 8);
    unsigned words[4];
#pragma unroll
    for (int w = 0; w < 4; ++w)
      words[w] = (2 * w < valid ? 0x0000ffffu : 0u) | (2 * w + 1 < valid ? 0xffff0000u : 0u);
    return uint4{x.x & words[0], x.y & words[1], x.z & words[2], x.w & words[3]};
  }

  template <int NB, class Shape>
  __device__ __forceinline__ static void Blocks(Prefetched const& pf, bool use_pf, MxAttentionOperands const& p,
                                                Plan<Shape> const& plan, int j, int lane, int r, int g,
                                                float* exchange, int c_max) {
    int const head = j / plan.chunks, c = j % plan.chunks;
    int const base = c * plan.tokens, end = min(base + plan.tokens, plan.length);
    Element const* const keys = p.k_cache + std::size_t(head) * p.capacity * D;
    Element const* const values = p.vt_cache + std::size_t(head) * D * p.capacity;
    float const scale = codegen::mx::kLog2e / sqrtf(float(D));

    // Q as B fragments: lane (q, g), 8 dimensions per 32-dimension chunk.
    uint4 query[kChunks];
    Element const* const qrow = p.qkv + (std::size_t(head) * (Q + 2) + min(r, Q - 1)) * D + 8 * g;
#pragma unroll
    for (int dc = 0; dc < kChunks; ++dc) query[dc] = Load(qrow + dc * 32);

    // Pass 1: S^T for every block, K two blocks in flight.
    float score[NB][2][4];
    uint4 key[2][2][kChunks];
    // Uniform by construction; readfirstlane lets the compiler branch on it.
    // use_pf stays outside: the builtin is not folded, and a constant false
    // inside it kept the prefetch live through the extra-items loop (MIR).
    if (use_pf && __builtin_mxc_readfirstlane(int(pf.valid))) {
#pragma unroll
      for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int dc = 0; dc < kChunks; ++dc) key[0][a][dc] = pf.k[a][dc];
    } else {
      LoadK(key[0], keys, base, end, lane);
    }
    if constexpr (NB > 1) LoadK(key[1], keys, base + 32, end, lane);
    {
      // q >= Q columns are zero.
      unsigned mask = 0u - unsigned(r < Q);
      asm volatile("" : "+r"(mask));
#pragma unroll
      for (int dc = 0; dc < kChunks; ++dc)
        query[dc] = uint4{query[dc].x & mask, query[dc].y & mask, query[dc].z & mask, query[dc].w & mask};
    }
#pragma unroll
    for (int block = 0; block < NB; ++block) {
      uint4 (&k)[2][kChunks] = key[block % 2];
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        MmaAccum acc{0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int dc = 0; dc < kChunks; ++dc) {
          acc = Mma(Lo(k[a][dc]), Lo(query[dc]), acc);
          acc = Mma(Hi(k[a][dc]), Hi(query[dc]), acc);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int const token = base + 32 * block + 8 * g + 4 * a + i;
          score[block][a][i] = token < end ? acc[i] * scale : -INFINITY;
        }
      }
      // The refill follows this block's MMAs (an input of the last score):
      // hoisted above them it needed a second set of registers and the wait
      // inserter shortened the ring.
      if (block + 2 < NB) {
        asm volatile("" ::"r"(score[block][1][3]));
        LoadK(k, keys, base + 32 * (block + 2), end, lane);
      }
    }

    // V^T fragments of one block: lane (d, g) loads V^T[d][block + 8g .. +7]
    // per dimension atom.
    // The atom stride goes through UniformOpaque here: otherwise the compiler
    // forms all D/16 row offsets (16n + r) * cap ahead of the program's loops
    // and keeps them live through pass 1 (stage_bench spilled).
    std::size_t const atom_stride = std::size_t(UniformOpaque(16 * p.capacity));
    Element const* const lane_values = values + std::size_t(r) * p.capacity;
    uint4 value[2][kDimAtoms];
    auto load_v = [&](uint4 (&v)[kDimAtoms], int block_base) {
      // Groups past `end` re-read the last valid group (in bounds; masked).
      int const first = block_base + 8 * g, at = first < end ? first : (end - 1) / 8 * 8;
#pragma unroll
      for (int n = 0; n < kDimAtoms; ++n) v[n] = Load(lane_values + n * atom_stride + at);
      asm volatile("" ::: "memory");  // blocks issue in order (see LoadK)
    };
    load_v(value[0], base);
    if constexpr (NB > 1) load_v(value[1], base + 32);

    // Row maximum of query r across the 4 lane groups.
    float maximum = -INFINITY;
#pragma unroll
    for (int block = 0; block < NB; ++block)
#pragma unroll
      for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int i = 0; i < 4; ++i) maximum = fmaxf(maximum, score[block][a][i]);
    float* const ex_max = exchange;
    float* const ex_sum = exchange + 64;
    ex_max[16 * g + r] = maximum;
    __syncwarp();
    maximum = fmaxf(fmaxf(ex_max[r], ex_max[16 + r]), fmaxf(ex_max[32 + r], ex_max[48 + r]));

    // Pass 2: P = exp2(s - m), O += P V^T, l = sum of FP32 P.
    MmaAccum out[kDimAtoms];
#pragma unroll
    for (int n = 0; n < kDimAtoms; ++n) out[n] = MmaAccum{0.f, 0.f, 0.f, 0.f};
    float sum = 0;
#pragma unroll
    for (int block = 0; block < NB; ++block) {
      alignas(16) Element prob[8];
#pragma unroll
      for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float const e = maca::Exp2(score[block][a][i] - maximum);
          sum += e;
          prob[4 * a + i] = Element(e);
        }
      uint4 const pa = *reinterpret_cast<uint4 const*>(prob);
      uint4 (&v)[kDimAtoms] = value[block % 2];
#pragma unroll
      for (int n = 0; n < kDimAtoms; ++n) {
        uint4 const x = MaskV(v[n], base + 32 * block, end, g);
        out[n] = Mma(Lo(pa), Lo(x), out[n]);
        out[n] = Mma(Hi(pa), Hi(x), out[n]);
      }
      if (block + 2 < NB) {
        asm volatile("" ::"r"(out[kDimAtoms - 1][3]));  // after this block's MMAs (see pass 1)
        load_v(v, base + 32 * (block + 2));
      }
    }
    ex_sum[16 * g + r] = sum;
    __syncwarp();
    // O layout: lane (d, g) holds O[q = 4g+i][16n + d].
    if (plan.chunks == 1) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int const q = 4 * g + i;
        if (q < Q) {
          float const l = ((ex_sum[q] + ex_sum[16 + q]) + ex_sum[32 + q]) + ex_sum[48 + q];
          Element* const row = p.ctx + (std::size_t(head) * Q + q) * D;
#pragma unroll
          for (int n = 0; n < kDimAtoms; ++n) row[16 * n + r] = Element(out[n][i] / l);
        }
      }
    } else {
      float* const item = p.partial + (std::size_t(head) * c_max + c) * Q * (D + 2);
      // One lane base and constant offsets (immediates): per-query addresses
      // were otherwise formed ahead of the step loop and spilled.
      float* const lane_item = item + 4 * g * (D + 2) + r;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (4 * g + i < Q) {
#pragma unroll
          for (int n = 0; n < kDimAtoms; ++n) lane_item[i * (D + 2) + 16 * n] = out[n][i];
        }
      }
      if (g == 0 && r < Q) {
        item[r * (D + 2) + D] = maximum;
        item[r * (D + 2) + D + 1] = ((ex_sum[r] + ex_sum[16 + r]) + ex_sum[32 + r]) + ex_sum[48 + r];
      }
    }
    __syncwarp();
  }

  /// The CTA that completed (b, g): every thread takes (q, d) = (t / D,
  /// t mod D) with stride kThreads and merges the C chunks in order.
  template <class Shape>
  __device__ __forceinline__ static void Merge(MxAttentionOperands const& p, Shape const& s, Plan<Shape> const& plan,
                                               int head, int c_max) {
    float const* const base = p.partial + std::size_t(head) * c_max * Q * (D + 2);
    // Opaque: the per-thread addresses were otherwise formed ahead of the
    // step loop and spilled (cold code; nothing to rematerialize here).
    for (int t = Opaque(threadIdx.x); t < Q * D; t += kThreads) {
      int const q = t / D, d = t % D;
      float m = -INFINITY;
      for (int c = 0; c < plan.chunks; ++c) m = fmaxf(m, LoadFloatGvm(base + (c * Q + q) * (D + 2) + D));
      float lambda = 0, o = 0;
      for (int c = 0; c < plan.chunks; ++c) {
        float const* const item = base + (c * Q + q) * (D + 2);
        float const w = maca::Exp2(LoadFloatGvm(item + D) - m);
        lambda += LoadFloatGvm(item + D + 1) * w;
        o += LoadFloatGvm(item + d) * w;
      }
      p.ctx[(std::size_t(head) * Q + q) * D + d] = Element(o / lambda);
    }
  }
};

}  // namespace tilemega::backend::mx
