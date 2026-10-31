// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Mx/MoeCombineV2.h>
#include <tilemega/Backend/Mx/RouteTableV3.h>

namespace tilemega::backend::mx {

enum class MxCombineModeV3 { kOwner16, kOwnerTile, kUnitOwner, kUnitLa };
struct MxCombineStateV3 { unsigned first, end, generation, pad; };
using MxSharedCombineStateV3 = __attribute__((address_space(3))) MxCombineStateV3;
static_assert(sizeof(MxCombineStateV3)==16);

template <bool Fp32>
struct MxMoeCombineOperandsV3 {
  Bf16 const* partials;
  MxSharedRouteTableV3<Fp32> const* table;
  Bf16 const* residual;
  Bf16* output;
  MxMoeGlobalTicketV2* tickets;
  MxSharedCombineStateV3* shared = nullptr;
  unsigned generation = 1;
  codegen::mx::StageWatchV2 watch{};
};

template <int TN, bool Fp32, MxCombineModeV3 Mode = MxCombineModeV3::kOwner16>
struct MxMoeCombineTileV3 {
  using Sync = codegen::mx::StageSyncV3<512>;
  static constexpr int kTile = Mode==MxCombineModeV3::kOwnerTile || Mode==MxCombineModeV3::kUnitLa ? TN : 16;

  __device__ __forceinline__ static void Initialize(MxSharedCombineStateV3* s,unsigned hidden) {
    if(Sync::IsRole()) {
      unsigned const tiles=hidden/kTile,cta=StageBlock();
      s->first=(cta*tiles+103)/104;s->end=((cta+1)*tiles+103)/104;
      s->generation=0;s->pad=0;
    }
  }

  __device__ __forceinline__ static void Item(MxMoeCombineOperandsV3<Fp32> const& p,
      unsigned token,unsigned column,unsigned hidden) {
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

  __device__ __forceinline__ static void UnitOwner(MxMoeCombineOperandsV3<Fp32> const& p,unsigned tile) {
    __threadfence();
#pragma unroll
    for(unsigned j=0;j<TN/16;++j)atomicAdd((unsigned*)(p.tickets+tile*(TN/16)+j),1u);
  }

  __device__ __forceinline__ static void UnitLa(MxMoeCombineOperandsV3<Fp32> const& p,
      unsigned tile,unsigned active,unsigned b,unsigned hidden,unsigned* flag) {
    codegen::executor::LastArriverV2<Sync>::Run((unsigned*)(p.tickets+tile),active,flag,[&] {
      unsigned const t=StageThread();
      for(unsigned item=t;item<b*TN/8;item+=512)
        Item(p,item/(TN/8),tile*TN+item%(TN/8)*8,hidden);
    });
  }

  __device__ __forceinline__ static void WaitOwned(MxMoeCombineOperandsV3<Fp32> const& p,unsigned active) {
    if(Sync::IsRole()) {
      using Counter=__attribute__((address_space(1))) volatile unsigned const;
      for(unsigned tc=p.shared->first;tc<p.shared->end;++tc) {
#if TILEMEGA_WATCHDOG
        unsigned long long start=0,failures=0;
#endif
        for(;;) {
          unsigned const observed=*((Counter*)p.tickets+tc);
          if(observed>=active)break;
#if TILEMEGA_WATCHDOG
          auto const& w=p.watch;
          if(w.record&&w.limit_ns&&!(++failures&4095ull)) {
            unsigned long long const now=maca::GlobalTimerNs();
            if(!start)start=now;
            if(now-start>w.limit_ns&&atomicCAS(&w.record->fired,0ull,1ull)==0ull) {
              w.record->site=12;w.record->block=blockIdx.x;w.record->thread=threadIdx.x;
              w.record->waiter_stage=w.stage;w.record->need=active;w.record->value=observed;
              w.record->iteration=w.iteration;__threadfence_system();maca::Trap();
            }
          }
#endif
        }
      }
      __threadfence();
      Sync::Publish((unsigned*)&p.shared->generation,p.generation);
    }
    Sync::Await((unsigned const*)&p.shared->generation,p.generation);
  }

  __device__ __forceinline__ static void Complete(MxMoeCombineOperandsV3<Fp32> const& p,
      unsigned begin,unsigned end,unsigned active,unsigned b,unsigned hidden,char*) {
    unsigned const t=StageThread(),wave=t/64,lane=t%64,tiles=hidden/TN;
    __builtin_assume(t<512);
    Sync::CtaSync();
    if constexpr(Mode!=MxCombineModeV3::kUnitLa) {
      if constexpr(Mode!=MxCombineModeV3::kUnitOwner) {
        if(wave==0) {
          __threadfence();
          for(unsigned tc=lane;tc<hidden/kTile;tc+=64) {
            unsigned count=0;
            for(unsigned unit=begin;unit<end;++unit)count+=unit%tiles==tc/(TN/kTile);
            if(count)atomicAdd((unsigned*)(p.tickets+tc),count);
          }
        }
      }
      // Every CTA finishes producing before waiting; all 104 CTAs are resident.
      WaitOwned(p,active);
      unsigned const first=p.shared->first,owned=p.shared->end-first;
      if(t<owned*b*(kTile/8)) {
        unsigned const tc=first+t/(b*kTile/8),within=t%(b*kTile/8);
        Item(p,within/(kTile/8),tc*kTile+within%(kTile/8)*8,hidden);
      }
      if(Sync::IsRole())
        for(unsigned tc=first;tc<p.shared->end;++tc)p.tickets[tc]=0;
      // Only the owner resets; the stage barrier separates the next use.
      Sync::CtaSync();
    }
  }
};

}  // namespace tilemega::backend::mx
