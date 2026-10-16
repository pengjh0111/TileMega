// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Backend/Mx/ServingDmGemmMx.h
//
// xcore1000 DM (DNN/MoE) GEMM mainloop: the dense MMA mainloop of
// ServingGemmMma.h generalized over how operands are addressed. One K tile
// (TK = 32/64/128) is consumed per iteration as TK/32 chunks; lane l of a
// wave holds row l & 15 of an atom and K values 8*(l >> 4) .. +7 of a chunk
// (the 16x16x16 MMA layout, P-9). `LoadA(am, it, c)` and `LoadB(j, it, c)`
// return that lane's eight BF16 values for A atom `am` / the wave's B atom
// `j`, iteration `it`, chunk `c`:
//   - DenseA: [M, K] rows, optional gather (DmSourceRow) and SCA scaling
//     (bf16(x * scale) before the MMA, as DmScaledActivation);
//   - DenseB: [N, K] rows, or fragment-packed tiles (xcore_mma tile_pages);
//   - ConvA / ConvB: implicit GEMM over an NHWC activation with halo and
//     KRSC (or packed) weights, one ConvIterationCursor per lane chunk.
// Loads are unconditional from clamped addresses; invalid values are zeroed
// by select (a load under a divergent branch serialized the loop, MX-T). The
// FP32 result goes to shared memory row-major [TM][TN] for the DM epilogue.
#pragma once

#include <tilemega/Backend/ConvIteration.h>
#include <tilemega/Backend/DmActivationScale.h>
#include <tilemega/Backend/DmMoeOperands.h>
#include <tilemega/Backend/Mx/ServingGemmMma.h>
#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/Maca/Bf16.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>

#include <cstdint>

namespace tilemega::backend {

template <int TM, int TN, int TK>
struct ServingDmGemmMx {
  using Element = tilemega::Bf16;
  static constexpr int kWaves = codegen::executor::kComputeThreads / maca::kWaveSize;
  static constexpr int kAtomsM = TM / 16, kAtomsN = TN / 16, kAtomsNPerWave = kAtomsN / kWaves;
  static constexpr int kChunks = TK / 32;
  static_assert(TM % 16 == 0 && TN % 32 == 0 && TK % 32 == 0 && kAtomsN % kWaves == 0,
                "xcore1000 DM tiles: TM in 16s, TN in 32s, TK in 32s");
  /// Loads per lane per iteration; two iterations are issued together when
  /// one does not keep eight 16-byte loads in flight (P-10).
  static constexpr int kLoads = kChunks * (kAtomsM + kAtomsNPerWave);
  static constexpr int kUnroll = kLoads >= 8 ? 1 : 2;
  static constexpr int kSharedBytes = 4 * TM * TN + 8 * TM;

  __device__ static int Lane() { return codegen::executor::ComputeThread() % maca::kWaveSize; }
  __device__ static int Wave() { return codegen::executor::ComputeThread() / maca::kWaveSize; }
  /// Column (B row) of the wave's atom j owned by this lane, before clamping.
  __device__ static int BColumn(int tile_n, int j) { return tile_n * TN + (Wave() * kAtomsNPerWave + j) * 16 + (Lane() & 15); }
  __device__ static int ARow(int tile_m, int am) { return tile_m * TM + am * 16 + (Lane() & 15); }
  __device__ static int KLane() { return 8 * (Lane() >> 4); }

  template <class Operands, class LoadA, class LoadB>
  __device__ static float* Run(Operands const& p, int tile_m, int tile_n, int iterations, LoadA&& load_a,
                               LoadB&& load_b, char* shared) {
    using codegen::executor::ComputeSync;
    mx::MmaAccum acc[kAtomsM][kAtomsNPerWave];
#pragma unroll
    for (int am = 0; am < kAtomsM; ++am)
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j) acc[am][j] = mx::MmaAccum{0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int first = 0; first < iterations; first += kUnroll) {
      uint4 a[kUnroll][kChunks][kAtomsM], b[kUnroll][kChunks][kAtomsNPerWave];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        // A second iteration past the end reloads the last one; its MMAs are skipped.
        int const it = min(first + u, iterations - 1);
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
          // A first: ConvB reads ConvA's cursors for the same iteration.
#pragma unroll
          for (int am = 0; am < kAtomsM; ++am) a[u][c][am] = load_a(am, it, c);
#pragma unroll
          for (int j = 0; j < kAtomsNPerWave; ++j) b[u][c][j] = load_b(j, it, c);
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (first + u >= iterations) break;
#pragma unroll
        for (int c = 0; c < kChunks; ++c)
#pragma unroll
          for (int am = 0; am < kAtomsM; ++am)
#pragma unroll
            for (int j = 0; j < kAtomsNPerWave; ++j) {
              acc[am][j] = mx::Mma(mx::Lo(a[u][c][am]), mx::Lo(b[u][c][j]), acc[am][j]);
              acc[am][j] = mx::Mma(mx::Hi(a[u][c][am]), mx::Hi(b[u][c][j]), acc[am][j]);
            }
      }
    }
    auto* tile = reinterpret_cast<float*>(shared);
    int const r = Lane() & 15, g = Lane() >> 4;
#pragma unroll
    for (int am = 0; am < kAtomsM; ++am)
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j) {
        int const col = (Wave() * kAtomsNPerWave + j) * 16 + r;
        bool const col_valid = tile_n * TN + col < p.n;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int const row = am * 16 + 4 * g + i;
          tile[row * TN + col] = col_valid && tile_m * TM + row < p.m ? acc[am][j][i] : 0.f;
        }
      }
    ComputeSync();
    return tile;
  }

  /// bf16(x * scale) for values[first .. first+count) of A row `source_row`,
  /// channels `channel` .. +count (DmScaledActivation: the scale is per image
  /// and channel; invalid elements become zero).
  template <class Operands>
  __device__ static void Scale(Operands const& p, std::uint32_t source_row, std::uint32_t channel,
                               std::uint32_t channels, std::uint32_t image_rows, Element* values, int count,
                               bool valid) {
    using namespace codegen;
    unsigned dtype = 1;
    std::uint64_t pitch = channels;
    if (p.access.a_scale != kDmNoIndex) {
      dtype = p.dm_buffers.dtypes[p.access.a_scale];
      auto const& layout = p.dm_buffers.layouts[p.access.a_scale];
      if (layout.rank) pitch = layout.strides[0];
    }
    std::uint64_t const base = std::uint64_t(source_row / image_rows) * pitch;
    for (int e = 0; e < count; ++e) {
      bool const ok = valid && channel + e < channels;
      float const scale = !ok ? 0.f
                          : dtype == 0 ? float(reinterpret_cast<Element const*>(p.a_scale)[base + channel + e])
                                       : p.a_scale[base + channel + e];
      values[e] = Element(ok ? float(values[e]) * scale : 0.f);
    }
  }

  /// [M, K] activation rows (dense or gathered), K tile `it` of [k_begin, k_begin + k_count).
  template <class Operands>
  struct DenseA {
    Operands const& p;
    Element const* row[kAtomsM];
    std::uint32_t source[kAtomsM];
    bool valid[kAtomsM];
    int count;
    __device__ DenseA(Operands const& operands, int tile_m) : p(operands), count(operands.k_count) {
      int const pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
#pragma unroll
      for (int am = 0; am < kAtomsM; ++am) {
        int const m = ARow(tile_m, am);
        valid[am] = m < p.m;
        source[am] = DmSourceRow(p, std::uint32_t(min(m, p.m - 1)));
        row[am] = p.a + std::int64_t(source[am]) * pitch + p.k_begin;
      }
    }
    __device__ uint4 operator()(int am, int it, int c) const {
      int const k = it * TK + c * 32 + KLane();
      uint4 v = k + 8 <= count ? LoadGlobal16(row[am] + k) : mx::LoadK8(row[am], k, count);
      if constexpr (HasDmActivationScale<Operands>::value) {
        if (p.a_scale)
          Scale(p, source[am], std::uint32_t(p.k_begin + k), std::uint32_t(p.k_total), p.access.rows_per_batch,
                reinterpret_cast<Element*>(&v), 8, valid[am]);
      }
      return v;
    }
  };

  /// [N, K] weight rows, or fragment-packed tiles when `tiled`.
  template <class Operands>
  struct DenseB {
    Operands const& p;
    Element const* row[kAtomsNPerWave];
    Element const* tiles;
    int count;
    bool tiled;
    __device__ DenseB(Operands const& operands, int tile_n, bool tiled_b)
        : p(operands), tiles(nullptr), count(operands.k_count), tiled(tiled_b) {
      int const pitch = p.b_row_stride ? p.b_row_stride : p.k_total;
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j)
        row[j] = p.b + std::int64_t(min(BColumn(tile_n, j), p.n - 1)) * pitch + p.k_begin;
      if (tiled) {
        int const kt = (p.k_total_full + TK - 1) / TK;
        tiles = p.weight_base + (std::size_t(tile_n) * kt + p.k_begin / TK) * TN * TK;
      }
    }
    __device__ uint4 operator()(int j, int it, int c) const {
      if (tiled)
        return LoadGlobal16(tiles + std::size_t(it) * TN * TK +
                            ServingGemmMma<16, TN, TK>::FragmentOffset(Wave() * kAtomsNPerWave + j, c, Lane()));
      int const k = it * TK + c * 32 + KLane();
      return k + 8 <= count ? LoadGlobal16(row[j] + k) : mx::LoadK8(row[j], k, count);
    }
  };

  /// Implicit-GEMM activation: NHWC with halo, K iterations of `geometry`.
  /// Each lane chunk owns a cursor; Pack 4 (four padded channels) splits a
  /// lane's eight K values into two 8-byte halves with their own cursors.
  template <class Operands, int Pack>
  struct ConvA {
    static constexpr int kHalves = 8 / Pack;
    Operands const& p;
    std::int64_t base[kAtomsM];
    std::uint32_t row[kAtomsM];
    bool row_valid[kAtomsM];
    ConvIterationCursor cursor[kChunks][kHalves];
    int next;
    std::uint32_t channels, image_rows;
    __device__ ConvA(Operands const& operands, codegen::ConvDesc const& conv, codegen::DmBufferLayout const& layout,
                     ConvIterationGeometry const& geometry, int tile_m, std::uint64_t begin)
        : p(operands), next(0), channels(conv.c), image_rows(conv.p * conv.q) {
#pragma unroll
      for (int am = 0; am < kAtomsM; ++am) {
        int const m = ARow(tile_m, am);
        row_valid[am] = m < p.m;
        row[am] = std::uint32_t(min(m, p.m - 1));
        auto image = row[am] / (conv.p * conv.q), pixel = row[am] % (conv.p * conv.q);
        auto y = std::int64_t(pixel / conv.q) * conv.stride_h - conv.pad_h + layout.halo_top;
        auto x = std::int64_t(pixel % conv.q) * conv.stride_w - conv.pad_w + layout.halo_left;
        base[am] = std::int64_t(image) * layout.strides[0] + y * layout.strides[1] + x * layout.strides[2];
      }
#pragma unroll
      for (int c = 0; c < kChunks; ++c)
#pragma unroll
        for (int h = 0; h < kHalves; ++h)
          cursor[c][h] = ConvIterationCursor(geometry, conv, layout, begin, c * 32 + KLane() + h * Pack);
    }
    /// Iterations are consumed in order; the cursors advance once per iteration.
    __device__ void Seek(int it) {
      while (next < it) {
#pragma unroll
        for (int c = 0; c < kChunks; ++c)
#pragma unroll
          for (int h = 0; h < kHalves; ++h) cursor[c][h].Advance();
        ++next;
      }
    }
    __device__ uint4 operator()(int am, int it, int c) {
      Seek(it);
      uint4 v;
      if constexpr (Pack == 8) {
        bool const valid = row_valid[am] && cursor[c][0].Point().valid;
        v = LoadGlobal16(p.a + (valid ? base[am] + cursor[c][0].a_offset : 0));
        if (!valid) v = uint4{0, 0, 0, 0};
      } else {
        uint2 half[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          bool const valid = row_valid[am] && cursor[c][h].Point().valid;
          half[h] = *reinterpret_cast<uint2 const*>(p.a + (valid ? base[am] + cursor[c][h].a_offset : 0));
          if (!valid) half[h] = uint2{0, 0};
        }
        v = uint4{half[0].x, half[0].y, half[1].x, half[1].y};
      }
      if constexpr (HasDmActivationScale<Operands>::value) {
        if (p.a_scale) {
          // Each Pack-wide half sits at one filter position: channels
          // point.c .. +Pack (the vector packing never straddles a tail).
          auto* element = reinterpret_cast<Element*>(&v);
#pragma unroll
          for (int h = 0; h < kHalves; ++h) {
            auto const point = cursor[c][h].Point();
            Scale(p, row[am], point.c, channels, image_rows, element + h * Pack, Pack,
                  row_valid[am] && point.valid);
          }
        }
      }
      return v;
    }
  };

  /// KRSC weights for ConvA (row pitch R*S*Cp), or fragment-packed tiles.
  template <class Operands, int Pack>
  struct ConvB {
    static constexpr int kHalves = 8 / Pack;
    Operands const& p;
    ConvA<Operands, Pack> const& a;
    Element const* column[kAtomsNPerWave];
    bool column_valid[kAtomsNPerWave];
    Element const* tiles;
    bool tiled;
    __device__ ConvB(Operands const& operands, ConvA<Operands, Pack> const& activation, codegen::ConvDesc const& conv,
                     ConvIterationGeometry const& geometry, int tile_n, std::uint64_t begin, bool tiled_b)
        : p(operands), a(activation), tiles(nullptr), tiled(tiled_b) {
      auto const pitch = std::uint64_t(conv.r) * conv.s * geometry.channels;
#pragma unroll
      for (int j = 0; j < kAtomsNPerWave; ++j) {
        int const n = BColumn(tile_n, j);
        column_valid[j] = n < p.n;
        column[j] = p.b + std::uint64_t(min(n, p.n - 1)) * pitch;
      }
      if (tiled) tiles = p.weight_base + (std::uint64_t(tile_n) * geometry.iterations + begin) * TN * TK;
    }
    /// Called after ConvA for the same iteration (the cursors are shared).
    __device__ uint4 operator()(int j, int it, int c) const {
      if (tiled)
        return LoadGlobal16(tiles + std::size_t(it) * TN * TK +
                            ServingGemmMma<16, TN, TK>::FragmentOffset(Wave() * kAtomsNPerWave + j, c, Lane()));
      if constexpr (Pack == 8) {
        bool const valid = column_valid[j] && a.cursor[c][0].Point().valid;
        uint4 v = LoadGlobal16(column[j] + (valid ? a.cursor[c][0].b_offset : 0));
        return valid ? v : uint4{0, 0, 0, 0};
      } else {
        uint2 half[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          bool const valid = column_valid[j] && a.cursor[c][h].Point().valid;
          half[h] = *reinterpret_cast<uint2 const*>(column[j] + (valid ? a.cursor[c][h].b_offset : 0));
          if (!valid) half[h] = uint2{0, 0};
        }
        return uint4{half[0].x, half[0].y, half[1].x, half[1].y};
      }
    }
  };
};

}  // namespace tilemega::backend
