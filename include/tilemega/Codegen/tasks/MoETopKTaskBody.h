// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/ArchDispatch.h>
#include <cutlass/bfloat16.h>
#include <cstdint>
#include <cmath>

#ifndef TILEMEGA_MOE_TOPK_WARP
#define TILEMEGA_MOE_TOPK_WARP 0
#endif

namespace tilemega::codegen {
struct MoeTopKOperands {
  // Router tiles publish top-K *rounded BF16* logits, with global indices.
  // Invalid lanes contain (-inf, INT32_MAX), including TN/K tails.
  float const* partial_logits=nullptr;
  std::int32_t const* partial_indices=nullptr;
  std::int32_t* indices=nullptr;
  void* weights=nullptr;
  unsigned tokens=0,experts=0,parts=0;
  std::uint64_t row_stride=0,part_stride=0,rank_stride=1;
  bool weights_fp32=false;
};

template<class Arch,int TopK=8,int TokensPerTask=128>
struct MoETopKTaskBody {
  static_assert(arch::Caps<Arch>::kBf16TensorCore);
  static_assert(TopK>0 && TopK<=32 && TokensPerTask>0);
  static constexpr int kThreads=128,kSharedBytes=0;
  __host__ __device__ static unsigned Count(MoeTopKOperands const& p) {
    return (p.tokens+TokensPerTask-1)/TokensPerTask;
  }
  __device__ static void RunWarp(MoeTopKOperands const& p,unsigned task) {
    unsigned lane=executor::ComputeThread()%32,warp=executor::ComputeThread()/32;
    for(unsigned local=warp;local<TokensPerTask;local+=4) {
      unsigned token=task*TokensPerTask+local;
      if(token>=p.tokens)continue;
      float candidates[4];std::int32_t indices[4];
      #pragma unroll
      for(unsigned slot=0;slot<4;++slot) {
        unsigned candidate=lane+slot*32;
        candidates[slot]=-INFINITY;indices[slot]=INT32_MAX;
        if(candidate<p.parts*TopK) {
          auto offset=token*p.row_stride+(candidate/TopK)*p.part_stride+
              (candidate%TopK)*p.rank_stride;
          candidates[slot]=p.partial_logits[offset];indices[slot]=p.partial_indices[offset];
          if(indices[slot]!=INT32_MAX && (indices[slot]<0 ||
              unsigned(indices[slot])>=p.experts || !isfinite(candidates[slot]))) {
            asm volatile("trap;");return;
          }
        }
      }
      float logits[TopK];std::int32_t selected[TopK];
      #pragma unroll
      for(unsigned rank=0;rank<TopK;++rank) {
        float value=-INFINITY;std::int32_t index=INT32_MAX;
        #pragma unroll
        for(unsigned slot=0;slot<4;++slot)
          if(indices[slot]!=INT32_MAX && (candidates[slot]>value ||
              (candidates[slot]==value && indices[slot]<index))) {
            value=candidates[slot];index=indices[slot];
          }
        #pragma unroll
        for(unsigned shift=16;shift;shift/=2) {
          float other=__shfl_xor_sync(0xffffffff,value,shift);
          auto other_index=__shfl_xor_sync(0xffffffff,index,shift);
          if(other>value || (other==value && other_index<index)) {
            value=other;index=other_index;
          }
        }
        if(index==INT32_MAX) {asm volatile("trap;");return;}
        unsigned matches=0;
        #pragma unroll
        for(unsigned slot=0;slot<4;++slot)matches+=indices[slot]==index;
        if(__reduce_add_sync(0xffffffff,matches)!=1) {asm volatile("trap;");return;}
        if(!lane) {logits[rank]=value;selected[rank]=index;}
        #pragma unroll
        for(unsigned slot=0;slot<4;++slot)
          if(indices[slot]==index)indices[slot]=INT32_MAX;
      }
      if(!lane) {
        // Keep the scalar path's rank order and FP32 softmax rounding.
        float probabilities[TopK],sum=0;
        #pragma unroll
        for(unsigned rank=0;rank<TopK;++rank) {
          probabilities[rank]=expf(logits[rank]-logits[0]);sum+=probabilities[rank];
        }
        #pragma unroll
        for(unsigned rank=0;rank<TopK;++rank) {
          p.indices[token*TopK+rank]=selected[rank];
          float weight=probabilities[rank]/sum;
          if(p.weights_fp32)static_cast<float*>(p.weights)[token*TopK+rank]=weight;
          else static_cast<cutlass::bfloat16_t*>(p.weights)[token*TopK+rank]=cutlass::bfloat16_t(weight);
        }
      }
    }
  }
  __device__ static void Run(MoeTopKOperands const& p,unsigned task) {
    if(!p.partial_logits || !p.partial_indices || !p.indices || !p.weights ||
       !p.parts || p.experts<TopK || !p.row_stride || !p.part_stride || !p.rank_stride) {
      asm volatile("trap;");return;
    }
#if TILEMEGA_MOE_TOPK_WARP
    if(p.parts*TopK<=128) {RunWarp(p,task);return;}
#endif
    for(unsigned local=executor::ComputeThread();local<TokensPerTask;local+=kThreads) {
      unsigned token=task*TokensPerTask+local;if(token>=p.tokens)continue;
      float logits[TopK];std::int32_t selected[TopK];
      #pragma unroll
      for(unsigned k=0;k<TopK;++k) {logits[k]=-INFINITY;selected[k]=INT32_MAX;}
      for(unsigned part=0;part<p.parts;++part)for(unsigned k=0;k<TopK;++k) {
        auto offset=token*p.row_stride+part*p.part_stride+k*p.rank_stride;
        float value=p.partial_logits[offset];auto index=p.partial_indices[offset];
        if(index==INT32_MAX)continue;
        if(index<0 || unsigned(index)>=p.experts || !isfinite(value)) {
          asm volatile("trap;");return;
        }
        #pragma unroll
        for(unsigned rank=0;rank<TopK;++rank) {
          if(value>logits[rank] || (value==logits[rank] && index<selected[rank])) {
            float old_value=logits[rank];auto old_index=selected[rank];
            logits[rank]=value;selected[rank]=index;value=old_value;index=old_index;
          }
        }
      }
      if(selected[TopK-1]==INT32_MAX) {asm volatile("trap;");return;}
      float probabilities[TopK],sum=0;
      #pragma unroll
      for(unsigned rank=0;rank<TopK;++rank) {
        probabilities[rank]=expf(logits[rank]-logits[0]);sum+=probabilities[rank];
        #pragma unroll
        for(unsigned prior=0;prior<rank;++prior)
          if(selected[prior]==selected[rank]) {asm volatile("trap;");return;}
      }
      #pragma unroll
      for(unsigned rank=0;rank<TopK;++rank) {
        p.indices[token*TopK+rank]=selected[rank];
        float weight=probabilities[rank]/sum;
        if(p.weights_fp32)static_cast<float*>(p.weights)[token*TopK+rank]=weight;
        else static_cast<cutlass::bfloat16_t*>(p.weights)[token*TopK+rank]=cutlass::bfloat16_t(weight);
      }
    }
  }
};
} // namespace tilemega::codegen
