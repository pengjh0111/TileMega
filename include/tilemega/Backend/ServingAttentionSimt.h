// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingAttentionWarp.h>

namespace tilemega::backend {
// Groups of D/8 lanes own one token and eight dimensions per lane. Two
// tokens per group are loaded before consuming either dependency chain.
template<class Arch,int D,int Q,class QueryLayout,class KeyLayout>
struct ServingAttentionSimt {
  static_assert(arch::Caps<Arch>::kCpAsync);
  static_assert((D==64 || D==128) && Q>0 && Q<=4);
  using Element=cutlass::bfloat16_t;
  static constexpr int Lanes=D/8,Groups=32/Lanes;
  float query[Q][8],acc[Q][8],maximum[Q],sum[Q];
  __device__ explicit ServingAttentionSimt(Element const* source) {
    int column=(ComputeThread()%Lanes)*8;
    #pragma unroll
    for(int q=0;q<Q;++q) {
      uint4 raw=*reinterpret_cast<uint4 const*>(source+QueryLayout{}(q,column));
      auto const* values=reinterpret_cast<Element const*>(&raw);
      #pragma unroll
      for(int i=0;i<8;++i){query[q][i]=float(values[i])*(1.4426950408889634f/sqrtf(float(D)));acc[q][i]=0;}
      maximum[q]=-INFINITY;sum[q]=0;
    }
  }
  template<class Valid>
  __device__ void Page(Element const* keys,Element const* values,
      int row_offset,int rows,Valid const& valid) {
    int lane=ComputeThread()%32,group=lane/Lanes,column=(lane%Lanes)*8;
    for(int begin=0;begin<rows;begin+=2*Groups) {
      uint4 key[2],value[2];bool active[2];
      #pragma unroll
      for(int u=0;u<2;++u) {
        int local=begin+u*Groups+group,row=row_offset+local;
        active[u]=local<rows && valid(local);
        key[u]={};value[u]={};
        if(active[u]) {
          int offset=(row/16)*16*D+KeyLayout{}(row%16,column);
          key[u]=*reinterpret_cast<uint4 const*>(keys+offset);
          value[u]=*reinterpret_cast<uint4 const*>(values+offset);
        }
      }
      #pragma unroll
      for(int u=0;u<2;++u) {
        auto const* k=reinterpret_cast<Element const*>(&key[u]);
        auto const* v=reinterpret_cast<Element const*>(&value[u]);
        #pragma unroll
        for(int q=0;q<Q;++q) {
          float score=0;
          #pragma unroll
          for(int i=0;i<8;++i)score+=query[q][i]*float(k[i]);
          #pragma unroll
          for(int delta=Lanes/2;delta>0;delta/=2)
            score+=__shfl_xor_sync(0xffffffffu,score,delta);
          score=active[u]?score:-INFINITY;
          float next=fmaxf(maximum[q],score);
          float alpha=isfinite(maximum[q])?exp2f(maximum[q]-next):0;
          float weight=active[u]?exp2f(score-next):0;
          sum[q]=sum[q]*alpha+weight;maximum[q]=next;
          #pragma unroll
          for(int i=0;i<8;++i)acc[q][i]=acc[q][i]*alpha+weight*float(v[i]);
        }
      }
    }
  }
  __device__ void Store(float* partial,float* lse,int stride) const {
    int lane=ComputeThread()%32,column=(lane%Lanes)*8;
    #pragma unroll
    for(int q=0;q<Q;++q) {
      float m=maximum[q];
      #pragma unroll
      for(int delta=Lanes;delta<32;delta*=2)
        m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,delta));
      float weight=isfinite(maximum[q])?exp2f(maximum[q]-m):0;
      float normalizer=weight*sum[q];
      #pragma unroll
      for(int delta=Lanes;delta<32;delta*=2)
        normalizer+=__shfl_xor_sync(0xffffffffu,normalizer,delta);
      #pragma unroll
      for(int i=0;i<8;++i) {
        float value=weight*acc[q][i];
        #pragma unroll
        for(int delta=Lanes;delta<32;delta*=2)
          value+=__shfl_xor_sync(0xffffffffu,value,delta);
        if(lane<Lanes)partial[q*stride+column+i]=normalizer>0?value/normalizer:0;
      }
      if(lane==0)lse[q]=normalizer>0?m+log2f(normalizer):-INFINITY;
    }
  }
};
} // namespace tilemega::backend
