// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Mx/MoeCombineV2.h>
#include <tilemega/Backend/Mx/RouteTableV3.h>

namespace tilemega::backend::mx {

template <bool Fp32>
struct MxMoeCombineOperandsV3 {
  Bf16 const* partials;
  MxSharedRouteTableV3<Fp32> const* table;
  Bf16 const* residual;
  Bf16* output;
  MxMoeGlobalTicketV2* tickets;
};

template <int TN, bool Fp32>
struct MxMoeCombineTileV3 {
  using Sync = codegen::mx::StageSyncV3<512>;

  using SharedLast = __attribute__((address_space(3))) unsigned;

  __device__ __forceinline__ static void Combine(MxMoeCombineOperandsV3<Fp32> const& p,
      SharedLast const* last, unsigned b, unsigned hidden) {
    unsigned const t = StageThread();
    __builtin_assume(t < 512);
    // Flatten all last tiles together; every output retains rank0..7 order.
    for (unsigned item = t; item < b * hidden / 8; item += 512) {
      unsigned const tile = item / (b * TN / 8), within = item % (b * TN / 8);
      if (!last[tile]) continue;
      unsigned const token = within / (TN / 8), column = tile * TN + (within % (TN / 8)) * 8;
      uint4 const old = LoadGlobal16Gvm(p.residual + token * hidden + column);
      uint4 partial[8];
      float weights[8];
#pragma unroll
      for (unsigned rank = 0; rank < 8; ++rank) {
        partial[rank] = LoadGlobal16Gvm(p.partials + (token * 8 + rank) * hidden + column);
        if constexpr (Fp32) weights[rank] = p.table->weight[token][rank];
        else weights[rank] = __uint_as_float(unsigned(p.table->weight[token][rank]) << 16);
      }
      __builtin_mxc_arrive_gvmcnt(0);
      alignas(16) Bf16 result[8], residual[8];
      *reinterpret_cast<uint4*>(residual) = old;
      float sum[8] = {};
#pragma unroll
      for (unsigned rank = 0; rank < 8; ++rank) {
        alignas(16) Bf16 values[8];
        *reinterpret_cast<uint4*>(values) = partial[rank];
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
          float product = __fmul_rn(weights[rank],float(values[e]));
          if constexpr (!Fp32) product = float(Bf16(product));
          sum[e] = __fadd_rn(sum[e],product);
        }
      }
#pragma unroll
      for (unsigned e = 0; e < 8; ++e)
        result[e] = Bf16(__fadd_rn(float(Bf16(sum[e])),float(residual[e])));
      MxStore16V2(p.output + token * hidden + column,*reinterpret_cast<uint4 const*>(result));
    }
  }

  __device__ __forceinline__ static void Complete(MxMoeCombineOperandsV3<Fp32> const& p,
      unsigned begin, unsigned end, unsigned active, unsigned b, unsigned hidden,
      char* smem) {
    unsigned const t = StageThread(), wave = t / 64, lane = t % 64, tiles = hidden / TN;
    __builtin_assume(t < 512);
    auto* const last = (SharedLast*)smem;
    Sync::CtaSync();
    if (wave == 0) {
      __threadfence();
      for (unsigned tile = lane; tile < tiles; tile += 64) {
        unsigned count = 0;
        for (unsigned unit = begin; unit < end; ++unit) count += unit % tiles == tile;
        last[tile] = 0;
        if (count) {
          unsigned const old = atomicAdd((unsigned*)(p.tickets + tile), count);
          if (old + count > active) maca::Trap();
          last[tile] = old + count == active;
        }
      }
    }
    Sync::CtaSync();
    unsigned any = 0;
    for (unsigned tile = 0; tile < tiles; ++tile) any |= last[tile];
    if (any) {
      if (wave == 0) __threadfence();
      Sync::CtaSync();
      Combine(p, last, b, hidden);
      if (wave == 0)
        for (unsigned tile = lane; tile < tiles; tile += 64)
          if (last[tile]) atomicExch((unsigned*)(p.tickets + tile), 0u);
    }
    Sync::CtaSync();
  }
};

}  // namespace tilemega::backend::mx
