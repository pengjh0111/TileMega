// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Mx/DecodeGemmV2.h>
#include <tilemega/Codegen/tasks/mx/StageSyncV2.cuh>
#include <cmath>
namespace tilemega::backend::mx {
struct MxMoeDispatchOperands {
  Bf16 const* logits;
  unsigned* n_active;
  unsigned* active;
  unsigned* row_count;
  uint2* rows;
  void* weights;
  unsigned* u_gate_up;
  unsigned* u_down;
#if TILEMEGA_MX_DIAG
  unsigned* route_debug;
  unsigned debug_step,layer,layers;
#endif
};
template <class T>
__device__ __forceinline__ void MxMoeStoreGlobal(T* target,T value) {
  using Global=__attribute__((address_space(1))) T;
  *(Global*)target=value;
}
__device__ __forceinline__ void MxMoeStoreGlobal(uint2* target,uint2 value) {
  MxStore8V2(reinterpret_cast<Bf16*>(target),value);
}
__device__ __forceinline__ void MxMoeStoreGlobal(uint4* target,uint4 value) {
  MxStore16V2(reinterpret_cast<Bf16*>(target),value);
}
template <int B,int Hidden,int Intermediate,int TnGate,int TnDown,bool Fp32,int Threads,
          int Experts=128,bool NormalizeTopK=true>
struct MxMoeDispatchV2 {
  static_assert(B>0&&B<=16&&Hidden%TnDown==0&&(2*Intermediate)%TnGate==0);
  static_assert(TnGate%32==0 && (Threads==512||Threads==576));
  static constexpr int E=Experts,K=8;
  static_assert(E==64||E==128);
  static constexpr int kSharedBytes=(B*E+B*K+B*K+E)*4;
  using Sync=codegen::mx::StageSyncV2<Threads>;
  // Caller is the last completed router tile and has acquired all BF16 logits.
  __device__ __forceinline__ static void Run(MxMoeDispatchOperands const& p,char* storage) {
    unsigned const t=StageThread();
    __builtin_assume(t<unsigned(Threads));
    float* const logits=reinterpret_cast<float*>(storage);
    float* const selected=logits+B*E;
    unsigned* const indices=reinterpret_cast<unsigned*>(selected+B*K);
    unsigned* const counts=indices+B*K;
    if(t<512 && t*8<B*E) {
      uint4 const packet=LoadGlobal16Gvm(p.logits+t*8);
      __builtin_mxc_arrive_gvmcnt(0);
      alignas(16) Bf16 values[8];*reinterpret_cast<uint4*>(values)=packet;
#pragma unroll
      for(int j=0;j<8;++j)logits[t*8+j]=float(values[j]);
    }
    Sync::CtaSync();
    if(t<512) {
      for(unsigned item=t;item<B*E;item+=512) {
        unsigned const token=item/E,expert=item%E;
        float const value=logits[item];
        if(!isfinite(value))maca::Trap();
        unsigned rank=0;
#pragma unroll 4
        for(unsigned base=0;base<E;base+=4) {
#pragma unroll
          for(unsigned offset=0;offset<4;++offset) {
            unsigned const j=base+offset;
            float const other=logits[token*E+j];
            rank+=unsigned(other>value)|(unsigned(other==value)&unsigned(j<expert));
          }
        }
        if(rank<K) {selected[token*K+rank]=value;indices[token*K+rank]=expert;}
      }
    }
    Sync::CtaSync();
    if(t<B) {
      float probabilities[K],sum=0;
#pragma unroll
      for(unsigned rank=0;rank<K;++rank) {
        probabilities[rank]=expf(selected[t*K+rank]-selected[t*K]);
        if constexpr(NormalizeTopK)sum+=probabilities[rank];
      }
      if constexpr(!NormalizeTopK) {
#pragma unroll 4
        for(unsigned e=0;e<E;++e)sum+=expf(logits[t*E+e]-selected[t*K]);
      }
      if constexpr(Fp32) {
        alignas(16) float weights[K];
#pragma unroll
        for(unsigned rank=0;rank<K;++rank)weights[rank]=probabilities[rank]/sum;
        auto* target=reinterpret_cast<uint4*>(p.weights)+t*2;
        MxMoeStoreGlobal(target,*reinterpret_cast<uint4 const*>(weights));
        MxMoeStoreGlobal(target+1,*reinterpret_cast<uint4 const*>(weights+4));
      } else {
        alignas(16) Bf16 weights[K];
#pragma unroll
        for(unsigned rank=0;rank<K;++rank)weights[rank]=Bf16(probabilities[rank]/sum);
        MxMoeStoreGlobal(reinterpret_cast<uint4*>(p.weights)+t,*reinterpret_cast<uint4 const*>(weights));
      }
    }
    if(t<E) {
      unsigned count=0;
#pragma unroll 4
      for(unsigned item=0;item<B*K;++item)count+=unsigned(indices[item]==t);
      counts[t]=count;
    }
    Sync::CtaSync();
    if(t<E && counts[t]) {
      unsigned ordinal=0;
#pragma unroll 4
      for(unsigned e=0;e<t;++e)ordinal+=unsigned(counts[e]!=0);
      MxMoeStoreGlobal(p.active+ordinal,t);
      MxMoeStoreGlobal(p.row_count+ordinal,counts[t]);
      unsigned row=0;
#pragma unroll 4
      for(unsigned item=0;item<B*K;++item) {
        if(indices[item]==t) {
          MxMoeStoreGlobal(p.rows+ordinal*B+row,uint2{item/K,item%K});++row;
        }
      }
    }
    if(t==0) {
      unsigned total=0;
#pragma unroll 4
      for(unsigned e=0;e<E;++e)total+=unsigned(counts[e]!=0);
      MxMoeStoreGlobal(p.n_active,total);
      MxMoeStoreGlobal(p.u_gate_up,total*(2*Intermediate/TnGate));
      MxMoeStoreGlobal(p.u_down,total*(Hidden/TnDown));
    }
#if TILEMEGA_MX_DIAG
    if(t<512 && p.route_debug && p.debug_step<64) {
      for(unsigned item=t;item<B*K;item+=512)
        MxMoeStoreGlobal(p.route_debug+((p.debug_step*p.layers+p.layer)*B*K+item),indices[item]);
    }
#endif
    // The router body performs its final CTA publication before its grid arrival.
  }
};
}
