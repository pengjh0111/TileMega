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
#include <utility>

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

/// `Stages` is the number of 32-K chunks in flight, and it is a real pipeline
/// depth: the loop below keeps Stages-1 chunks loading while it runs the MMAs
/// of the oldest. Until MX-3 this parameter existed only in
/// ServingGemmConfig's shared-memory and legality arithmetic while the body
/// ran one chunk at a time, so the cost model priced a pipeline that was not
/// there (it divides the iteration latency by Stages-1 and drops Stages-1
/// iterations, TaskPriceParts.cpp:146 and :148).
///
/// Depth lives in registers, not shared memory. Measured on this C550
/// (experiments/mx3/pipeline_probe.cu, two runs within 1%): with two MMAs per
/// load the serial form takes 755 ns/load and depth 6 takes 226 ns, a 3.34x
/// gain that brings the loop within 16% of the 195 ns bandwidth bound. The
/// same probe with no MMA between the loads is already bandwidth-bound at
/// depth 1, and every depth is then slower -- the gain is in hiding MMA
/// latency, not in issuing loads sooner. A shared-memory ring was slower than
/// both at every depth.
/// Registers one chunk in flight costs a lane, and whether a depth fits.
/// Free functions so the solver and the codegen can ask before naming a
/// ServingGemmMma: a static_assert in the class body would fire on the query
/// itself. The budget is the 255-register file less the accumulators, the
/// operand staging and the addressing the mainloop already needs.
constexpr int ServingGemmLoadsPerChunk(int tile_m, int tile_n, int waves) {
  return tile_n / 16 / waves + tile_m / 16;
}
constexpr int ServingGemmPipelineVgprs(int tile_m, int tile_n, int stages, int waves) {
  return stages * ServingGemmLoadsPerChunk(tile_m, tile_n, waves) * 4;
}
constexpr bool ServingGemmDepthFits(int tile_m, int tile_n, int stages, int waves) {
  return stages >= 1 && ServingGemmPipelineVgprs(tile_m, tile_n, stages, waves) <= 160;
}

template <int TileM, int TileN, int TileK, int Stages = 1>
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
  /// Chunks in flight. One is the serial form this body had before MX-3.
  static constexpr int kDepth = Stages < 1 ? 1 : Stages;
  /// Registers the pipeline holds beyond the accumulators: each chunk in
  /// flight keeps kLoadsPerChunk 16-byte fragments per lane.
  static constexpr int kPipelineVgprs = kDepth * kLoadsPerChunk * 4;
  static constexpr bool kDepthFits =
      ServingGemmDepthFits(TileM, TileN, kDepth, kWaves);

  /// Offset of lane `lane`'s 16 bytes of atom `atom`, 32-K chunk `chunk`
  /// inside one TileN x TileK weight tile in fragment order: a wave's load
  /// of one atom and chunk is 1 KiB contiguous (the loader's `xcore_mma`
  /// tile_pages layout, python/tilemega/serving/weights.py).
  __device__ static constexpr int FragmentOffset(int atom, int chunk, int lane) {
    return ((atom * (TileK / kChunkK) + chunk) * maca::kWaveSize + lane) * 8;
  }

  /// Lane coordinates shared by every operand source: the lane's MMA row
  /// (`r`, an A row or a B column inside an atom), its 8-K group within a
  /// 32-K chunk (`g`) and its wave.
  struct Lane {
    int lane, wave, r, g;
    __device__ Lane()
        : lane(codegen::executor::ComputeThread() % maca::kWaveSize),
          wave(codegen::executor::ComputeThread() / maca::kWaveSize), r(lane & 15), g(lane >> 4) {}
  };

  /// The register-direct MMA loop over a source policy:
  ///   uint4 A(am, kc), B(j, kc)            full 32-K chunk at relative K kc;
  ///   uint4 ATail(am, kc, count), BTail(j, kc, count)   predicated chunk.
  /// Full blocks of kUnroll chunks load unconditionally (a load under a
  /// lane-divergent branch serialized the loop, measured); only the K tail is
  /// predicated. The FP32 tile, row-major [TileM][TileN] plus TileM floats of
  /// epilogue scratch, zeroes rows >= `rows` and columns >= `columns`.
  template <class Source>
  __device__ static float* Run(Source& source, Lane const& l, int count, int rows, int columns, char* shared) {
    using codegen::executor::ComputeSync;
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
    static_assert(kDepthFits,
                  "pipeline depth would exhaust the register file; the solver "
                  "must not offer this Stages for this tile");
    // Whole 32-K chunks run through a depth-kDepth pipeline; the ragged K tail
    // keeps the predicated path below. The MMAs stay in chunk order, so the
    // K-direction accumulation order is unchanged and C-2 stays bit-exact.
    int const chunks = count / kChunkK;
    int const groups = chunks / kDepth;
    uint4 a[kDepth][kAtomsM], b[kDepth][kAtomsNPerWave];
    auto load = [&](auto slot, int chunk) {
      constexpr int kAt = decltype(slot)::value;
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j) b[kAt][j] = source.B(j, chunk * kChunkK);
#pragma unroll
      for (int am = 0; am < kAtomsM; ++am) a[kAt][am] = source.A(am, chunk * kChunkK);
    };
    // Prologue: kDepth-1 chunks in flight before the first MMA.
    [&]<std::size_t... kPrime>(std::index_sequence<kPrime...>) {
      ((int(kPrime) < chunks
            ? load(std::integral_constant<int, int(kPrime)>{}, int(kPrime))
            : void()),
       ...);
    }(std::make_index_sequence<kDepth - 1>{});
    // The ring slot must be a compile-time constant: indexing these arrays
    // with a runtime `chunk % kDepth` puts the whole pipeline in local memory
    // and cost 5.8x the serial form when measured that way, so the chunk loop
    // is unrolled by kDepth and each copy owns one slot.
    // An index_sequence fold, not a `#pragma unroll` loop: the slot index has
    // to be a template argument for the arrays to stay in registers, and a
    // loop variable cannot be used in a constant expression.
    auto step = [&](auto stage, int g) {
      constexpr int d = decltype(stage)::value;
      int const ahead = g * kDepth + d + kDepth - 1;
      if (ahead < chunks)
        load(std::integral_constant<int, (d + kDepth - 1) % kDepth>{}, ahead);
      // Leave the newest kDepth-1 chunks outstanding; retirement is in order,
      // so the slot about to be read has landed.
      maca::WaitAsyncCopies<(kDepth - 1) * kLoadsPerChunk>();
      mma(a[d], b[d]);
    };
#pragma unroll 1
    for (int g = 0; g < groups; ++g)
      [&]<std::size_t... kStage>(std::index_sequence<kStage...>) {
        (step(std::integral_constant<int, int(kStage)>{}, g), ...);
      }(std::make_index_sequence<kDepth>{});
    // Whole chunks the groups did not cover.
#pragma unroll 1
    for (int chunk = groups * kDepth; chunk < chunks; ++chunk) {
      load(std::integral_constant<int, 0>{}, chunk);
      maca::WaitAsyncCopies<0>();
      mma(a[0], b[0]);
    }
    int const full = chunks * kChunkK;
#pragma unroll 1
    for (int k0 = full; k0 < count; k0 += kChunkK) {
      uint4 a[kAtomsM], b[kAtomsNPerWave];
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j) b[j] = source.BTail(j, k0, count);
#pragma unroll
      for (int am = 0; am < kAtomsM; ++am) a[am] = source.ATail(am, k0, count);
      mma(a, b);
    }
    auto* tile = reinterpret_cast<float*>(shared);
#pragma unroll
    for (int am = 0; am < kAtomsM; ++am)
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j) {
        int const col = (l.wave * kAtomsNPerWave + j) * 16 + l.r;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int const row = am * 16 + 4 * l.g + i;
          tile[row * TileN + col] = col < columns && row < rows ? acc[am][j][i] : 0.f;
        }
      }
    ComputeSync();
    return tile;
  }

  /// A weight tile in fragment order (`Tiled` B): tile `k_tile` of the
  /// packed row of N tiles starting at `tiles`, zero-padded to whole tiles.
  __device__ static uint4 TiledB(Element const* tiles, std::size_t k_tile, int chunk, Lane const& l, int j) {
    return LoadGlobal16(tiles + k_tile * TileN * TileK +
                        FragmentOffset(l.wave * kAtomsNPerWave + j, chunk, l.lane));
  }

  /// Dense operands: row-major A [M, K]; B as [N, K] rows or (`Tiled`)
  /// fragment-packed tiles from `weight_base`. Rows past p.m and columns past
  /// p.n read the last valid row/column (unconditional loads); Run zeroes
  /// their results.
  template <bool Tiled, class Operands>
  struct DenseSource {
    Operands const& p;
    Lane const& l;
    Element const* a_row[kAtomsM];
    Element const* b_row[kAtomsNPerWave];
    Element const* b_tiles = nullptr;
    __device__ DenseSource(Operands const& operands, Lane const& lane, int tile_m, int tile_n)
        : p(operands), l(lane) {
      int const a_pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
      int const b_pitch = p.b_row_stride ? p.b_row_stride : p.k_total;
#pragma unroll
      for (int am = 0; am < kAtomsM; ++am)
        a_row[am] = p.a + std::int64_t(min(tile_m * TileM + am * 16 + l.r, p.m - 1)) * a_pitch + p.k_begin;
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j)
        b_row[j] = Tiled ? nullptr
                         : p.b + std::int64_t(min(tile_n * TileN + (l.wave * kAtomsNPerWave + j) * 16 + l.r,
                                                  p.n - 1)) * b_pitch + p.k_begin;
      if constexpr (Tiled) {
        static_assert(TileK % kChunkK == 0, "fragment-packed tiles hold whole 32-K chunks");
        b_tiles = p.weight_base + std::size_t(tile_n) * ((p.k_total_full + TileK - 1) / TileK) * TileN * TileK;
      }
    }
    __device__ uint4 A(int am, int kc) const { return LoadGlobal16(a_row[am] + kc + 8 * l.g); }
    __device__ uint4 ATail(int am, int kc, int count) const { return mx::LoadK8(a_row[am], kc + 8 * l.g, count); }
    __device__ uint4 B(int j, int kc) const {
      if constexpr (Tiled) {
        int const kk = p.k_begin + kc;
        return TiledB(b_tiles, kk / TileK, (kk % TileK) / kChunkK, l, j);
      } else {
        return LoadGlobal16(b_row[j] + kc + 8 * l.g);
      }
    }
    // Tiled B is zero-padded past K; the A tail zeroes the products.
    __device__ uint4 BTail(int j, int kc, int count) const {
      if constexpr (Tiled) return B(j, kc);
      else return mx::LoadK8(b_row[j], kc + 8 * l.g, count);
    }
  };

  /// Dense GEMM tile. `Tiled`: B from fragment-packed tiles (the loader's
  /// xcore_mma tile_pages layout); measured ~2x the row form's DRAM rate.
  template <bool Tiled = false, class Operands>
  __device__ static float* Dense(Operands const& p, int tile_m, int tile_n, char* shared) {
    Lane const l;
    DenseSource<Tiled, Operands> source(p, l, tile_m, tile_n);
    return Run(source, l, p.k_count, p.m - tile_m * TileM, p.n - tile_n * TileN, shared);
  }
};

}  // namespace tilemega::backend
