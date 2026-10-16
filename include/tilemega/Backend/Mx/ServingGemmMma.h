// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Backend/Mx/ServingGemmMma.h
//
// xcore1000 serving GEMM mainloop: C[TileM x TileN] = A[M,K] * B[N,K]^T with
// __builtin_mxc_mma_16x16x16bf16 (FP32 accumulate). The 128 compute threads
// are two 64-lane waves; wave w owns N atoms [w*kAtomsNPerWave, ...) for all
// M atoms, so A fragments are reused across a wave's N atoms.
//
// Fragment layout (P-9, verified): lane l holds A row l&15 / B column l&15 /
// C column l&15, and K (A, B) or M (C) elements 4*(l>>4)+i. One 16-byte load
// covers 8 consecutive K for one row; elements 0..3 feed the first MMA and
// 4..7 the second. A and B use the same K assignment, so each 32-K chunk is
// summed exactly once, in a fixed order (C-2).
//
// Operands stream from global memory straight into registers: P-10 measured
// 97 % of DRAM bandwidth at 2 waves/AP with >= 8 outstanding 16-byte loads
// per lane, against 78 % for the ldg_b128_bsm shared-memory ring.
#pragma once

#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/Maca/Bf16.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>

#include <cstdint>
#include <cstring>

namespace tilemega::backend {

namespace mx {
typedef float MmaAccum __attribute__((ext_vector_type(4)));
typedef __fp16 MmaOperand __attribute__((__vector_size__(4 * sizeof(__fp16))));

__device__ __forceinline__ MmaOperand Lo(uint4 v) {
  MmaOperand r;
  unsigned bits[2] = {v.x, v.y};
  std::memcpy(&r, bits, sizeof(r));
  return r;
}
__device__ __forceinline__ MmaOperand Hi(uint4 v) {
  MmaOperand r;
  unsigned bits[2] = {v.z, v.w};
  std::memcpy(&r, bits, sizeof(r));
  return r;
}
__device__ __forceinline__ MmaAccum Mma(MmaOperand a, MmaOperand b, MmaAccum c) {
  return __builtin_mxc_mma_16x16x16bf16(a, b, c);
}

/// Eight BF16 starting at `row + k`, zero outside [0, valid).
__device__ __forceinline__ uint4 LoadK8(tilemega::Bf16 const* row, int k, int valid) {
  if (k + 8 <= valid) return LoadGlobal16(row + k);
  alignas(16) tilemega::Bf16 values[8];
  for (int e = 0; e < 8; ++e)
    values[e] = k + e < valid ? row[k + e] : tilemega::Bf16::bitcast(0);
  return *reinterpret_cast<uint4 const*>(values);
}
}  // namespace mx

template <int TileM, int TileN, int TileK>
struct ServingGemmMma {
  using Element = tilemega::Bf16;
  static constexpr int kWaves = codegen::executor::kComputeThreads / maca::kWaveSize;
  static constexpr int kAtomsM = TileM / 16, kAtomsN = TileN / 16;
  static_assert(TileM % 16 == 0 && TileN % 16 == 0 && kAtomsN % kWaves == 0,
                "MMA tiles are whole 16x16 atoms split evenly over the waves");
  static constexpr int kAtomsNPerWave = kAtomsN / kWaves;
  /// 32-K chunks loaded before their MMAs: keep >= 8 loads per lane in flight.
  static constexpr int kLoadsPerChunk = kAtomsNPerWave + kAtomsM;
  static constexpr int kUnroll = kLoadsPerChunk >= 8 ? 1 : kLoadsPerChunk >= 4 ? 2 : 4;
  static constexpr int kChunkK = 32;

  /// Offset of lane `lane`'s 16 bytes of atom `atom`, 32-K chunk `chunk`
  /// inside one TileN x TileK weight tile in fragment order: a wave's load
  /// of one atom and chunk is 1 KiB contiguous (the loader's `xcore_mma`
  /// tile_pages layout, python/tilemega/serving/weights.py).
  __device__ static constexpr int FragmentOffset(int atom, int chunk, int lane) {
    return ((atom * (TileK / kChunkK) + chunk) * maca::kWaveSize + lane) * 8;
  }

  /// Accumulate the tile into FP32 shared memory, row-major [TileM][TileN],
  /// followed by TileM floats of epilogue scratch. `Tiled`: B is read from
  /// `weight_base` in fragment-packed tiles, zero-padded to whole tiles, so B
  /// needs no bounds handling; the row form reads [N, K] rows. The
  /// fragment-packed stream measured ~2x the row form's DRAM rate (MX-T).
  template <bool Tiled = false, class Operands>
  __device__ static float* Dense(Operands const& p, int tile_m, int tile_n, char* shared) {
    using codegen::executor::ComputeSync;
    using codegen::executor::ComputeThread;
    int const lane = ComputeThread() % maca::kWaveSize;
    int const wave = ComputeThread() / maca::kWaveSize;
    int const r = lane & 15, g = lane >> 4;
    int const a_pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
    int const b_pitch = p.b_row_stride ? p.b_row_stride : p.k_total;

    // Rows past p.m and columns past p.n read the last valid row/column so
    // every load is unconditional: a load under a lane-divergent branch
    // serialized the mainloop (M = 1 leaves 15 of 16 A lanes invalid; the
    // predicated form streamed ~8 GB/s per CTA, measured). Their results are
    // zeroed when the tile is written.
    Element const* a_row[kAtomsM];
#pragma unroll
    for (int am = 0; am < kAtomsM; ++am)
      a_row[am] = p.a + std::int64_t(min(tile_m * TileM + am * 16 + r, p.m - 1)) * a_pitch + p.k_begin;
    Element const* b_row[kAtomsNPerWave];
#pragma unroll
    for (int j = 0; j < kAtomsNPerWave; ++j)
      b_row[j] = Tiled ? nullptr
                       : p.b + std::int64_t(min(tile_n * TileN + (wave * kAtomsNPerWave + j) * 16 + r, p.n - 1)) *
                                   b_pitch + p.k_begin;
    // Tiled: tile (tile_n, kk / TileK) of the packed weight, kk = k_begin + k.
    Element const* b_tiles = nullptr;
    if constexpr (Tiled) {
      static_assert(TileK % kChunkK == 0, "fragment-packed tiles hold whole 32-K chunks");
      int const kt = (p.k_total_full + TileK - 1) / TileK;
      b_tiles = p.weight_base + std::size_t(tile_n) * kt * TileN * TileK;
    }
    auto load_b = [&](int j, int k_chunk_start) -> uint4 {
      if constexpr (Tiled) {
        int const kk = p.k_begin + k_chunk_start;
        return LoadGlobal16(b_tiles + std::size_t(kk / TileK) * TileN * TileK +
                            FragmentOffset(wave * kAtomsNPerWave + j, (kk % TileK) / kChunkK, lane));
      } else {
        return LoadGlobal16(b_row[j] + k_chunk_start + 8 * g);
      }
    };

    mx::MmaAccum acc[kAtomsM][kAtomsNPerWave];
#pragma unroll
    for (int am = 0; am < kAtomsM; ++am)
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j) acc[am][j] = mx::MmaAccum{0.f, 0.f, 0.f, 0.f};
    auto mma = [&](uint4 const (&a)[kAtomsM], uint4 const (&b)[kAtomsNPerWave]) {
#pragma unroll
      for (int am = 0; am < kAtomsM; ++am)
#pragma unroll
        for (int j = 0; j < kAtomsNPerWave; ++j) {
          acc[am][j] = mx::Mma(mx::Lo(a[am]), mx::Lo(b[j]), acc[am][j]);
          acc[am][j] = mx::Mma(mx::Hi(a[am]), mx::Hi(b[j]), acc[am][j]);
        }
    };

    int const count = p.k_count, full = count / (kChunkK * kUnroll) * (kChunkK * kUnroll);
#pragma unroll 1
    for (int k0 = 0; k0 < full; k0 += kChunkK * kUnroll) {
      uint4 a[kUnroll][kAtomsM], b[kUnroll][kAtomsNPerWave];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        int const k = k0 + u * kChunkK + 8 * g;
#pragma unroll
        for (int j = 0; j < kAtomsNPerWave; ++j) b[u][j] = load_b(j, k0 + u * kChunkK);
#pragma unroll
        for (int am = 0; am < kAtomsM; ++am) a[u][am] = LoadGlobal16(a_row[am] + k);
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) mma(a[u], b[u]);
    }
    // K tail: predicated, fewer than kUnroll chunks.
#pragma unroll 1
    for (int k0 = full; k0 < count; k0 += kChunkK) {
      int const k = k0 + 8 * g;
      uint4 a[kAtomsM], b[kAtomsNPerWave];
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j) b[j] = Tiled ? load_b(j, k0) : mx::LoadK8(b_row[j], k, count);
#pragma unroll
      for (int am = 0; am < kAtomsM; ++am) a[am] = mx::LoadK8(a_row[am], k, count);
      mma(a, b);
    }

    auto* tile = reinterpret_cast<float*>(shared);
#pragma unroll
    for (int am = 0; am < kAtomsM; ++am)
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j) {
        int const col = (wave * kAtomsNPerWave + j) * 16 + r;
        bool const col_valid = tile_n * TileN + col < p.n;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int const row = am * 16 + 4 * g + i;
          tile[row * TileN + col] = col_valid && tile_m * TileM + row < p.m ? acc[am][j][i] : 0.f;
        }
      }
    ComputeSync();
    return tile;
  }
};

}  // namespace tilemega::backend
