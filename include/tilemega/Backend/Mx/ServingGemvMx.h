// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Backend/Mx/ServingGemvMx.h
//
// xcore1000 SIMT GEMV for M <= 4 rows: C[M x TileN] = A[M,K] * B[N,K]^T in
// FP32. Each 32-lane logical warp owns TileN/4 columns; a lane covers 8
// consecutive K per 256-K step, so a column's loads are fully coalesced.
// kUnroll steps of B (16 16-byte loads per lane) are issued before their
// FMAs to keep enough reads in flight (P-10). The 32-lane reduction uses xor
// offsets below 32, which stay inside the half wave (P-5).
#pragma once

#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Backend/Mx/ServingGemmMma.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/Maca/Bf16.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>

#include <cstdint>

namespace tilemega::backend {

template <int TileM, int TileN, int TileK>
struct ServingGemvMx {
  static_assert(TileN == 8 || TileN == 16 || TileN == 32 || TileN == 64);
  using Element = tilemega::Bf16;
  static constexpr int kRows = TileM < 4 ? TileM : 4, kWarps = 4;
  static constexpr int kColumns = TileN / kWarps, kUnroll = 16 / kColumns;
  static constexpr int kStepK = 32 * 8;

  template <class Operands>
  __device__ static bool Supported(Operands const& p, int tile_m) {
    int const rows = p.m - tile_m * TileM;
    return rows > 0 && rows <= kRows && p.k_begin >= 0 && p.k_count > 0 &&
           p.k_begin + p.k_count <= p.k_total;
  }

  template <class Operands>
  __device__ static float* Dense(Operands const& p, int tile_m, int tile_n, char* shared) {
    using codegen::executor::ComputeSync;
    using codegen::executor::ComputeThread;
    if (!Supported(p, tile_m) || !p.a || !p.b) maca::Trap();
    int const lane = ComputeThread() % 32, warp = ComputeThread() / 32;
    int const a_pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
    int const b_pitch = p.b_row_stride ? p.b_row_stride : p.k_total;
    int const rows = p.m - tile_m * TileM;
    int const count = p.k_count;

    Element const* b_row[kColumns];
    bool b_valid[kColumns];
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
      int const n = tile_n * TileN + warp * kColumns + c;
      b_valid[c] = n < p.n;
      b_row[c] = p.b + (b_valid[c] ? std::int64_t(n) * b_pitch + p.k_begin : 0);
    }
    Element const* a_row[kRows];
#pragma unroll
    for (int row = 0; row < kRows; ++row)
      a_row[row] = p.a + (row < rows ? std::int64_t(tile_m * TileM + row) * a_pitch + p.k_begin : 0);

    float acc[kColumns][kRows] = {};
    for (int first = 0; first < count; first += kStepK * kUnroll) {
      uint4 weights[kUnroll][kColumns];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        int const k = first + u * kStepK + lane * 8;
#pragma unroll
        for (int c = 0; c < kColumns; ++c)
          weights[u][c] = b_valid[c] && k < count ? mx::LoadK8(b_row[c], k, count) : uint4{0, 0, 0, 0};
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        int const k = first + u * kStepK + lane * 8;
        alignas(16) Element activation[kRows][8];
#pragma unroll
        for (int row = 0; row < kRows; ++row)
          *reinterpret_cast<uint4*>(activation[row]) =
              row < rows && k < count ? mx::LoadK8(a_row[row], k, count) : uint4{0, 0, 0, 0};
#pragma unroll
        for (int c = 0; c < kColumns; ++c) {
          alignas(16) Element b[8];
          *reinterpret_cast<uint4*>(b) = weights[u][c];
#pragma unroll
          for (int row = 0; row < kRows; ++row)
#pragma unroll
            for (int e = 0; e < 8; ++e)
              acc[c][row] = fmaf(float(activation[row][e]), float(b[e]), acc[c][row]);
        }
      }
    }

    auto* tile = reinterpret_cast<float*>(shared);
    for (int i = ComputeThread(); i < TileM * TileN; i += codegen::executor::kComputeThreads) tile[i] = 0;
    ComputeSync();
#pragma unroll
    for (int c = 0; c < kColumns; ++c)
#pragma unroll
      for (int row = 0; row < kRows; ++row) {
        float sum = acc[c][row];
#pragma unroll
        for (int shift = 16; shift; shift /= 2) sum += __shfl_xor_sync(maca::kFullWaveMask, sum, shift, 32);
        if (lane == 0) tile[row * TileN + warp * kColumns + c] = sum;
      }
    ComputeSync();
    return tile;
  }
};

}  // namespace tilemega::backend
