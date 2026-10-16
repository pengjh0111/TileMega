// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Backend/ServingConv.h (xcore1000)
//
// Implicit-GEMM convolution: the register-direct MMA loop over an im2col
// source. K is the issued iteration space of ConvIterationGeometry (each
// filter position's channel tail padded to a whole K tile), so split-K
// partitions iterations. A row m is an output pixel; input buffers carry
// their zero halo, so only the channel tail is predicated. A lane's eight
// K elements are eight channels of one filter position (16 bytes), or two
// positions of four packed RGB channels (two 8-byte loads). B is the KRSC
// weight with the same per-position padding, as rows or fragment-packed
// tiles over the iteration space.
#pragma once
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <tilemega/Backend/ConvIteration.h>
#include <tilemega/Backend/ServingDmGemm.h>

namespace tilemega::backend {

template <class Arch, int TM, int TN, int TK, int Stages>
struct ServingConv {
  using Gemm = ServingDmGemm<Arch, TM, TN, TK, Stages>;
  using Mma = typename Gemm::Mma;
  using Lane = typename Mma::Lane;
  using Element = typename Gemm::Element;
  static constexpr int kSharedBytes = Gemm::kSharedBytes;

  template <bool TiledB, class Operands>
  struct Source {
    Operands const& p;
    codegen::ConvDesc const& conv;
    ConvIterationGeometry geometry;
    Lane const& l;
    std::uint64_t begin;
    DmActivationScaler<Operands> scaler;
    std::int64_t row_base[Mma::kAtomsM];
    unsigned row[Mma::kAtomsM];
    int column[Mma::kAtomsNPerWave];
    std::int64_t delta_row, delta_column;
    std::uint64_t weight_pitch;
    Element const* b_tiles = nullptr;

    __device__ Source(Operands const& operands, codegen::ConvDesc const& c, codegen::DmBufferLayout const& layout,
                      ConvIterationGeometry g, Lane const& lane, int tile_m, int tile_n, std::uint64_t first, bool& ok)
        : p(operands), conv(c), geometry(g), l(lane), begin(first) {
      ok = scaler.Init(p, true);
      delta_row = std::int64_t(c.dilation_h) * layout.strides[1];
      delta_column = std::int64_t(c.dilation_w) * layout.strides[2];
      weight_pitch = std::uint64_t(c.r) * c.s * g.channels;
#pragma unroll
      for (int am = 0; am < Mma::kAtomsM; ++am) {
        row[am] = unsigned(min(tile_m * TM + am * 16 + l.r, p.m - 1));
        auto image = row[am] / (c.p * c.q), pixel = row[am] % (c.p * c.q);
        auto y = std::int64_t(pixel / c.q) * c.stride_h - c.pad_h + layout.halo_top;
        auto x = std::int64_t(pixel % c.q) * c.stride_w - c.pad_w + layout.halo_left;
        row_base[am] = std::int64_t(image) * layout.strides[0] + y * layout.strides[1] + x * layout.strides[2];
      }
#pragma unroll
      for (int j = 0; j < Mma::kAtomsNPerWave; ++j)
        column[j] = min(tile_n * TN + (l.wave * Mma::kAtomsNPerWave + j) * 16 + l.r, p.n - 1);
      if (TiledB) b_tiles = p.weight_base + std::uint64_t(tile_n) * g.iterations * TN * TK;
    }
    __device__ ConvIterationPoint Point(int k) const { return geometry.At(begin + k / TK, unsigned(k % TK)); }
    /// Four or eight elements of one filter position at element offset `k`.
    template <int Width>
    __device__ uint2 Piece(Element const* base, std::int64_t offset, bool valid) const {
      if (!valid) return uint2{0, 0};
      return *reinterpret_cast<uint2 const*>(base + offset);
    }
    __device__ uint4 A(int am, int kc) const {
      int const k = kc + 8 * l.g;
      if (geometry.channels >= 8) {
        auto const point = Point(k);
        uint4 v = point.valid ? LoadGlobal16(p.a + row_base[am] + std::int64_t(point.r) * delta_row +
                                             std::int64_t(point.s) * delta_column + point.c)
                              : uint4{0, 0, 0, 0};
        return scaler.Apply(v, row[am], point.c, geometry.channels, point.valid ? 8 : 0);
      }
      // Four packed channels per position: two independent 8-byte pieces.
      uint4 v;
      auto* halves = reinterpret_cast<uint2*>(&v);
      for (int h = 0; h < 2; ++h) {
        auto const point = Point(k + 4 * h);
        halves[h] = Piece<4>(p.a, row_base[am] + std::int64_t(point.r) * delta_row +
                                      std::int64_t(point.s) * delta_column + point.c, point.valid);
        if (scaler.scale) {
          uint4 part{halves[h].x, halves[h].y, 0, 0};
          part = scaler.Apply(part, row[am], point.c, geometry.channels, point.valid ? 4 : 0);
          halves[h] = uint2{part.x, part.y};
        }
      }
      return v;
    }
    __device__ uint4 ATail(int am, int kc, int) const { return A(am, kc); }
    __device__ uint4 B(int j, int kc) const {
      if (TiledB) return Mma::TiledB(b_tiles, begin + kc / TK, (kc % TK) / Mma::kChunkK, l, j);
      int const k = kc + 8 * l.g;
      Element const* base = p.b + std::uint64_t(column[j]) * weight_pitch;
      if (geometry.channels >= 8) {
        auto const point = Point(k);
        return point.valid ? LoadGlobal16(base + (std::uint64_t(point.r) * geometry.columns + point.s) *
                                                     geometry.channels + point.c)
                           : uint4{0, 0, 0, 0};
      }
      uint4 v;
      auto* halves = reinterpret_cast<uint2*>(&v);
      for (int h = 0; h < 2; ++h) {
        auto const point = Point(k + 4 * h);
        halves[h] = Piece<4>(base, std::int64_t((std::uint64_t(point.r) * geometry.columns + point.s) *
                                                    geometry.channels + point.c), point.valid);
      }
      return v;
    }
    __device__ uint4 BTail(int j, int kc, int) const { return B(j, kc); }
  };

  template <bool TiledB, class Operands>
  __device__ static float* Run(Operands const& p, codegen::ConvDesc const& conv,
                               codegen::DmBufferLayout const& layout, ConvIterationGeometry geometry, int tile_m,
                               int tile_n, std::uint64_t begin, int iterations, char* shared) {
    if (!p.a || !p.b || geometry.tile_k != TK || begin >= geometry.iterations || iterations <= 0 ||
        std::uint64_t(iterations) > geometry.iterations - begin || !conv.p || !conv.q ||
        (TiledB && !p.weight_base) || (geometry.channels != 4 && geometry.channels % 8)) {
      maca::Trap();
      return nullptr;
    }
    Lane const l;
    bool ok = true;
    Source<TiledB, Operands> source(p, conv, layout, geometry, l, tile_m, tile_n, begin, ok);
    if (!ok) maca::Trap();
    return Mma::Run(source, l, iterations * TK, p.m - tile_m * TM, p.n - tile_n * TN, shared);
  }
};
}  // namespace tilemega::backend
