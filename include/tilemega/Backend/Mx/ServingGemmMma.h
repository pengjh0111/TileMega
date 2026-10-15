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

  /// Accumulate the tile into FP32 shared memory, row-major [TileM][TileN],
  /// followed by TileM floats of epilogue scratch.
  template <class Operands>
  __device__ static float* Dense(Operands const& p, int tile_m, int tile_n, char* shared) {
    using codegen::executor::ComputeSync;
    using codegen::executor::ComputeThread;
    int const lane = ComputeThread() % maca::kWaveSize;
    int const wave = ComputeThread() / maca::kWaveSize;
    int const r = lane & 15, g = lane >> 4;
    int const a_pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
    int const b_pitch = p.b_row_stride ? p.b_row_stride : p.k_total;

    Element const* a_row[kAtomsM];
    bool a_valid[kAtomsM];
#pragma unroll
    for (int am = 0; am < kAtomsM; ++am) {
      int const m = tile_m * TileM + am * 16 + r;
      a_valid[am] = m < p.m;
      a_row[am] = p.a + (a_valid[am] ? std::int64_t(m) * a_pitch + p.k_begin : 0);
    }
    Element const* b_row[kAtomsNPerWave];
    bool b_valid[kAtomsNPerWave];
#pragma unroll
    for (int j = 0; j < kAtomsNPerWave; ++j) {
      int const n = tile_n * TileN + (wave * kAtomsNPerWave + j) * 16 + r;
      b_valid[j] = n < p.n;
      b_row[j] = p.b + (b_valid[j] ? std::int64_t(n) * b_pitch + p.k_begin : 0);
    }

    mx::MmaAccum acc[kAtomsM][kAtomsNPerWave];
#pragma unroll
    for (int am = 0; am < kAtomsM; ++am)
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j) acc[am][j] = mx::MmaAccum{0.f, 0.f, 0.f, 0.f};

    int const count = p.k_count;
    for (int k0 = 0; k0 < count; k0 += kChunkK * kUnroll) {
      uint4 a[kUnroll][kAtomsM], b[kUnroll][kAtomsNPerWave];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        int const k = k0 + u * kChunkK + 8 * g;
#pragma unroll
        for (int j = 0; j < kAtomsNPerWave; ++j)
          b[u][j] = b_valid[j] && k < count ? mx::LoadK8(b_row[j], k, count) : uint4{0, 0, 0, 0};
#pragma unroll
        for (int am = 0; am < kAtomsM; ++am)
          a[u][am] = a_valid[am] && k < count ? mx::LoadK8(a_row[am], k, count) : uint4{0, 0, 0, 0};
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u)
#pragma unroll
        for (int am = 0; am < kAtomsM; ++am)
#pragma unroll
          for (int j = 0; j < kAtomsNPerWave; ++j) {
            acc[am][j] = mx::Mma(mx::Lo(a[u][am]), mx::Lo(b[u][j]), acc[am][j]);
            acc[am][j] = mx::Mma(mx::Hi(a[u][am]), mx::Hi(b[u][j]), acc[am][j]);
          }
    }

    auto* tile = reinterpret_cast<float*>(shared);
#pragma unroll
    for (int am = 0; am < kAtomsM; ++am)
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j) {
        int const col = (wave * kAtomsNPerWave + j) * 16 + r;
#pragma unroll
        for (int i = 0; i < 4; ++i) tile[(am * 16 + 4 * g + i) * TileN + col] = acc[am][j][i];
      }
    ComputeSync();
    return tile;
  }
};

}  // namespace tilemega::backend
