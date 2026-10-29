// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Mx/DecodeGemmV2.h>
#include <tilemega/Codegen/executor/LastArriverV2.cuh>
namespace tilemega::backend::mx {
using MxMoeGlobalTicketV2=__attribute__((address_space(1))) unsigned;
struct MxMoeCombineOperandsV2 {
  Bf16 const* partials;
  void const* weights;
  Bf16 const* residual;
  Bf16* output;
  MxMoeGlobalTicketV2* tickets;
};
template<int TN,bool Fp32,int Threads>
struct MxMoeCombineTileV2 {
  static_assert(TN%16==0);
  using Sync=codegen::mx::StageSyncV2<Threads>;
  // The caller has published one expert's entire output tile before arriving.
  __device__ __forceinline__ static bool Arrive(MxMoeCombineOperandsV2 const& p,
      unsigned tile,unsigned active,unsigned b,unsigned hidden,unsigned* shared_last) {
    // Keep the global address space through the ABI field and LA inlining.
    unsigned* const ticket=(unsigned*)(p.tickets+tile);
    return codegen::executor::LastArriverV2<Sync>::Run(ticket,active,shared_last,[&] {
      unsigned const t=StageThread();
      __builtin_assume(t<unsigned(Threads));
      if(t<512) {
        for(unsigned item=t;item<b*TN/8;item+=512) {
          unsigned const token=item/(TN/8),column=tile*TN+(item%(TN/8))*8;
          uint4 const old=LoadGlobal16Gvm(p.residual+token*hidden+column);
          uint4 partial[8];
          float weights[8];
#pragma unroll
          for(unsigned rank=0;rank<8;++rank) {
            partial[rank]=LoadGlobal16Gvm(p.partials+(token*8+rank)*hidden+column);
            if constexpr(Fp32) {
              using Global=__attribute__((address_space(1))) float const;
              weights[rank]=*((Global*)p.weights+token*8+rank);
            } else {
              using Global=__attribute__((address_space(1))) unsigned short const;
              unsigned const bits=*((Global*)p.weights+token*8+rank);
              weights[rank]=__uint_as_float(bits<<16);
            }
          }
          __builtin_mxc_arrive_gvmcnt(0);
          alignas(16) Bf16 result[8],residual[8];
          *reinterpret_cast<uint4*>(residual)=old;
          float sum[8]={};
#pragma unroll
          for(unsigned rank=0;rank<8;++rank) {
            alignas(16) Bf16 values[8];
            *reinterpret_cast<uint4*>(values)=partial[rank];
#pragma unroll
            for(unsigned e=0;e<8;++e) {
              float product=__fmul_rn(weights[rank],float(values[e]));
              if constexpr(!Fp32)product=float(Bf16(product));
              sum[e]=__fadd_rn(sum[e],product);
            }
          }
#pragma unroll
          for(unsigned e=0;e<8;++e)
            result[e]=Bf16(__fadd_rn(float(Bf16(sum[e])),float(residual[e])));
          MxStore16V2(p.output+token*hidden+column,*reinterpret_cast<uint4 const*>(result));
        }
      }
    });
  }
};
template<int B,int Hidden,int TN,bool Fp32,int Threads>
struct MxMoeCombineV2 {
  static_assert(B>0&&B<=16&&Hidden%TN==0);
  __device__ __forceinline__ static bool Arrive(MxMoeCombineOperandsV2 const& p,
      unsigned tile,unsigned active,unsigned* shared_last) {
    return MxMoeCombineTileV2<TN,Fp32,Threads>::Arrive(p,tile,active,B,Hidden,shared_last);
  }
};
}
