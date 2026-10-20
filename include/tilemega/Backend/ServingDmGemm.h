// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Backend/ServingDmGemm.h (xcore1000)
//
// DM GEMM mainloop for gathered or scaled activations and expert-indirect
// weights: the register-direct MMA loop of Backend/Mx/ServingGemmMma.h over a
// DM operand source. A rows come from DmSourceRow (row gather, MoE binding
// rows or storage rows); B is [N, K] rows or fragment-packed tiles, already
// offset to the bound expert by ResolveDmMoeTile. SCA activation scales are
// applied to the eight A elements a lane holds and rounded to BF16 before the
// MMA, the same rounding point as the NVIDIA fragment transform.
#pragma once
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <tilemega/Backend/DmActivationScale.h>
#include <tilemega/Backend/Mx/ServingGemmMma.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Solver/DmGemmTraits.h>
#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Target/Maca/Bf16.h>
#include <cstdint>
#include <type_traits>

namespace tilemega::backend {

/// Scale lookup for one activation row (DmScaledActivation's addressing):
/// scale(source_row, column) = a_scale[(source_row / image_rows) * pitch + column].
template <class Operands>
struct DmActivationScaler {
  float const* scale = nullptr;
  unsigned dtype = 1, channels = 0, image_rows = 0;
  std::uint64_t pitch = 0;
  __device__ bool Init(Operands const& p, bool conv) {
    using namespace codegen;
    if constexpr (!HasDmActivationScale<Operands>::value) {
      return true;
    } else {
      if (!p.a_scale) return true;
      scale = p.a_scale;
      channels = conv ? p.convolutions[p.access.conv].c : unsigned(p.k_total);
      image_rows = conv ? p.convolutions[p.access.conv].p * p.convolutions[p.access.conv].q
                        : p.access.rows_per_batch;
      if (!image_rows) return false;
      pitch = channels;
      if (p.access.a_scale != kDmNoIndex) {
        if (p.access.a_scale >= p.dm_buffers.count || !p.dm_buffers.dtypes || !p.dm_buffers.layouts) return false;
        dtype = p.dm_buffers.dtypes[p.access.a_scale];
        auto const& layout = p.dm_buffers.layouts[p.access.a_scale];
        if (dtype > 1 || (layout.rank && (layout.rank != 2 || layout.logical[1] != channels ||
                                          layout.strides[1] != 1)))
          return false;
        if (layout.rank) pitch = layout.strides[0];
      }
      return true;
    }
  }
  /// Scale `count` (<= 8) activations whose channels are `column + i`;
  /// channels at or past `limit` or `channels` become zero.
  __device__ uint4 Apply(uint4 raw, unsigned source_row, unsigned column, unsigned limit, unsigned count) const {
    if (!scale) return raw;
    auto* values = reinterpret_cast<tilemega::Bf16*>(&raw);
    std::uint64_t const base = std::uint64_t(source_row / image_rows) * pitch;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      bool const valid = i < count && column + i < limit && column + i < channels;
      float const s = !valid ? 0.f
                      : dtype == 0 ? float(reinterpret_cast<tilemega::Bf16 const*>(scale)[base + column + i])
                                   : scale[base + column + i];
      values[i] = tilemega::Bf16(valid ? float(values[i]) * s : 0.f);
    }
    return raw;
  }
};

template <class Arch, int TM, int TN, int TK, int Stages>
struct ServingDmGemm {
  static_assert(std::is_same_v<Arch, arch::Xcore1000>, "xcore1000 DM GEMM only");
  static_assert(solver::DmServingBF16ShapeLegal(TM, TN, TK, Stages));
  using Element = tilemega::Bf16;
  using Mma = ServingGemmMma<TM, TN, TK, Stages>;
  using Lane = typename Mma::Lane;
  static constexpr int kThreads = 128;
  static constexpr int kSharedBytes = solver::DmServingBF16SmemBytes(TM, TN, TK, Stages);

  template <bool Tiled, class Operands>
  struct Source {
    Operands const& p;
    Lane const& l;
    DmActivationScaler<Operands> scaler;
    Element const* a_row[Mma::kAtomsM];
    unsigned source_row[Mma::kAtomsM];
    Element const* b_row[Mma::kAtomsNPerWave];
    Element const* b_tiles = nullptr;
    __device__ Source(Operands const& operands, Lane const& lane, int tile_m, int tile_n, bool& ok)
        : p(operands), l(lane) {
      int const a_pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
      int const b_pitch = p.b_row_stride ? p.b_row_stride : p.k_total;
      ok = scaler.Init(p, false);
#pragma unroll
      for (int am = 0; am < Mma::kAtomsM; ++am) {
        source_row[am] = DmSourceRow(p, std::uint32_t(min(tile_m * TM + am * 16 + l.r, p.m - 1)));
        a_row[am] = p.a + std::size_t(source_row[am]) * a_pitch + p.k_begin;
      }
#pragma unroll
      for (int j = 0; j < Mma::kAtomsNPerWave; ++j)
        b_row[j] = Tiled ? nullptr
                         : p.b + std::int64_t(min(tile_n * TN + (l.wave * Mma::kAtomsNPerWave + j) * 16 + l.r,
                                                  p.n - 1)) * b_pitch + p.k_begin;
      if (Tiled) b_tiles = p.weight_base + std::size_t(tile_n) * ((p.k_total_full + TK - 1) / TK) * TN * TK;
    }
    __device__ uint4 Scaled(int am, uint4 raw, int k, int count) const {
      return scaler.Apply(raw, source_row[am], unsigned(p.k_begin + k), unsigned(p.k_begin + count),
                          unsigned(count - k < 8 ? count - k : 8));
    }
    __device__ uint4 A(int am, int kc) const {
      int const k = kc + 8 * l.g;
      return Scaled(am, LoadGlobal16(a_row[am] + k), k, p.k_count);
    }
    __device__ uint4 ATail(int am, int kc, int count) const {
      int const k = kc + 8 * l.g;
      return k < count ? Scaled(am, mx::LoadK8(a_row[am], k, count), k, count) : uint4{0, 0, 0, 0};
    }
    __device__ uint4 B(int j, int kc) const {
      if (Tiled) {
        int const kk = p.k_begin + kc;
        return Mma::TiledB(b_tiles, kk / TK, (kk % TK) / Mma::kChunkK, l, j);
      }
      return LoadGlobal16(b_row[j] + kc + 8 * l.g);
    }
    __device__ uint4 BTail(int j, int kc, int count) const {
      return Tiled ? B(j, kc) : mx::LoadK8(b_row[j], kc + 8 * l.g, count);
    }
  };

  template <class Operands>
  __device__ static float* Dense(Operands const& p, int tm, int tn, char* shared, bool tiled_b = false) {
    int const a_pitch = p.a_row_stride ? p.a_row_stride : p.k_total;
    int const b_pitch = p.b_row_stride ? p.b_row_stride : p.k_total;
    if (!p.a || !p.b || p.k_begin < 0 || p.k_count <= 0 || p.k_begin % 8 || p.k_begin + p.k_count > p.k_total ||
        a_pitch % 8 || b_pitch % 8 || a_pitch < p.k_total || b_pitch < p.k_total ||
        (tiled_b && (!p.weight_base || p.k_total_full <= 0 || p.k_begin % TK))) {
      maca::Trap();
      return nullptr;
    }
    Lane const l;
    bool ok = true;
    auto run = [&](auto source) {
      if (!ok) maca::Trap();
      return Mma::Run(source, l, p.k_count, p.m - tm * TM, p.n - tn * TN, shared);
    };
    if (tiled_b) return run(Source<true, Operands>(p, l, tm, tn, ok));
    return run(Source<false, Operands>(p, l, tm, tn, ok));
  }
};

}  // namespace tilemega::backend
