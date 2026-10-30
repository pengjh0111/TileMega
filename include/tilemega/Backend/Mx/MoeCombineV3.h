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

  __device__ __forceinline__ static void Combine(MxMoeCombineOperandsV3<Fp32> const& p,
                                                unsigned tile, unsigned b, unsigned hidden) {
    unsigned const t = StageThread();
    __builtin_assume(t < 512);
    for (unsigned item = t; item < b * TN / 8; item += 512) {
      unsigned const token = item / (TN / 8), column = tile * TN + (item % (TN / 8)) * 8;
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

  // CP1 control retains per-unit completion; CP2 replaces this publication site.
  __device__ __forceinline__ static bool Arrive(MxMoeCombineOperandsV3<Fp32> const& p,
      unsigned tile, unsigned active, unsigned b, unsigned hidden, unsigned* shared_last) {
    return codegen::executor::LastArriverV2<Sync>::Run((unsigned*)(p.tickets+tile),active,shared_last,
        [&] { Combine(p,tile,b,hidden); });
  }
};

}  // namespace tilemega::backend::mx
