// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Backend/Mx/ServingGemvMx.h
//
// xcore1000 SIMT GEMV for M <= 4 rows: C[M x TileN] = A[M,K] * B[N,K]^T in
// FP32. Each 32-lane logical warp owns TileN/4 columns and walks them in
// passes of kGroup columns; a lane covers 8 consecutive K per 256-K step, so a
// column's loads are fully coalesced. A pass issues kGroup * kUnroll 16-byte
// weight loads per lane before their FMAs (16 in flight, P-10). The main loop
// loads unconditionally from clamped rows (a load under a branch costs ~30 %,
// measured); only the K tail is predicated. Keeping one pass of columns live
// holds the body under the register file: the all-columns-at-once form used
// 256 registers and spilled 472 bytes. The 32-lane reduction uses xor offsets
// below 32, which stay inside the half wave (P-5).
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
  static constexpr int kColumns = TileN / kWarps;
  static constexpr int kGroup = kColumns < 4 ? kColumns : 4, kUnroll = 16 / kGroup;
  static constexpr int kStepK = 32 * 8, kBlockK = kStepK * kUnroll;

  template <class Operands>
  __device__ static bool Supported(Operands const& p, int tile_m) {
    int const rows = p.m - tile_m * TileM;
    return rows > 0 && rows <= kRows && p.k_begin >= 0 && p.k_count > 0 &&
           p.k_begin + p.k_count <= p.k_total;
  }

  /// acc[c][row] += A[row, k..k+8) . B[column c, k..k+8) for one lane's span.
  __device__ static void Accumulate(uint4 const& weights, uint4 const (&activation)[kRows], float (&acc)[kRows]) {
    auto const* b = reinterpret_cast<Element const*>(&weights);
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      auto const* a = reinterpret_cast<Element const*>(&activation[row]);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[row] = fmaf(float(a[e]), float(b[e]), acc[row]);
    }
  }

  template <class Operands>
  __device__ static float* Dense(Operands const& p, int tile_m, int tile_n, char* shared) {
    using codegen::executor::ComputeSync;
    using codegen::executor::ComputeThread;
    if (!Supported(p, tile_m) || !p.a || !p.b) maca::Trap();
    int const lane = ComputeThread() % 32, warp = ComputeThread() / 32;
    std::int64_t const a_pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
    std::int64_t const b_pitch = p.b_row_stride ? p.b_row_stride : p.k_total;
    int const rows = p.m - tile_m * TileM;
    int const count = p.k_count, full = count / kBlockK * kBlockK;
    // Rows past `rows` read the last valid row; their sums are discarded.
    Element const* a_base = p.a + std::int64_t(tile_m * TileM) * a_pitch + p.k_begin;
    int const first_column = tile_n * TileN + warp * kColumns;

    auto* tile = reinterpret_cast<float*>(shared);
    // Rows kRows..TileM are never valid (Supported); the epilogue may still
    // scale them, so they hold zeros. Disjoint from the rows written below.
    for (int i = kRows * TileN + ComputeThread(); i < TileM * TileN; i += codegen::executor::kComputeThreads)
      tile[i] = 0;
#pragma unroll 1
    for (int group = 0; group < kColumns; group += kGroup) {
      // Columns past p.n read column p.n - 1; their sums are discarded.
      Element const* b_base = p.b + p.k_begin;
      int column[kGroup];
#pragma unroll
      for (int c = 0; c < kGroup; ++c) column[c] = min(first_column + group + c, p.n - 1);
      float acc[kGroup][kRows] = {};
#pragma unroll 1
      for (int first = 0; first < full; first += kBlockK) {
        uint4 weights[kUnroll][kGroup];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)
#pragma unroll
          for (int c = 0; c < kGroup; ++c)
            weights[u][c] = LoadGlobal16(b_base + column[c] * b_pitch + first + u * kStepK + lane * 8);
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          uint4 activation[kRows];
#pragma unroll
          for (int row = 0; row < kRows; ++row)
            activation[row] = LoadGlobal16(a_base + min(row, rows - 1) * a_pitch + first + u * kStepK + lane * 8);
#pragma unroll
          for (int c = 0; c < kGroup; ++c) Accumulate(weights[u][c], activation, acc[c]);
        }
      }
      // K tail: predicated, at most kUnroll steps.
#pragma unroll 1
      for (int k = full + lane * 8; k < count; k += kStepK) {
        uint4 activation[kRows];
#pragma unroll
        for (int row = 0; row < kRows; ++row)
          activation[row] = mx::LoadK8(a_base + min(row, rows - 1) * a_pitch, k, count);
#pragma unroll
        for (int c = 0; c < kGroup; ++c)
          Accumulate(mx::LoadK8(b_base + column[c] * b_pitch, k, count), activation, acc[c]);
      }
#pragma unroll
      for (int c = 0; c < kGroup; ++c)
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
          float sum = acc[c][row];
#pragma unroll
          for (int shift = 16; shift; shift /= 2) sum += __shfl_xor_sync(maca::kFullWaveMask, sum, shift, 32);
          bool const valid = row < rows && first_column + group + c < p.n;
          if (lane == 0) tile[row * TileN + warp * kColumns + group + c] = valid ? sum : 0.f;
        }
    }
    ComputeSync();
    return tile;
  }
};

}  // namespace tilemega::backend
