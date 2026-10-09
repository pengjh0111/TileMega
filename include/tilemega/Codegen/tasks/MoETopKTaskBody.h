// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/ArchDispatch.h>
#include <cutlass/bfloat16.h>
#include <cstdint>
#include <cmath>

namespace tilemega::codegen {
struct MoeTopKOperands {
  // Router tiles publish top-K *rounded BF16* logits, with global indices.
  // Invalid lanes contain (-inf, INT32_MAX), including TN/K tails.
  float const* partial_logits=nullptr;
  std::int32_t const* partial_indices=nullptr;
  std::int32_t* indices=nullptr;
  cutlass::bfloat16_t* weights=nullptr;
  unsigned tokens=0,experts=0,parts=0;
  std::uint64_t row_stride=0,part_stride=0,rank_stride=1;
};

template<class Arch,int TopK=8,int TokensPerTask=128>
struct MoETopKTaskBody {
  static_assert(arch::Caps<Arch>::kBf16TensorCore);
  static_assert(TopK>0 && TopK<=32 && TokensPerTask>0);
  static constexpr int kThreads=128,kSharedBytes=0;
  __host__ __device__ static unsigned Count(MoeTopKOperands const& p) {
    return (p.tokens+TokensPerTask-1)/TokensPerTask;
  }
  __device__ static void Run(MoeTopKOperands const& p,unsigned task) {
    if(!p.partial_logits || !p.partial_indices || !p.indices || !p.weights ||
       !p.parts || p.experts<TopK || !p.row_stride || !p.part_stride || !p.rank_stride) {
      asm volatile("trap;");return;
    }
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
        // The normalized selected softmax has one BF16 rounding, after FP32
        // exp/sum/div. No routing weight is folded into an expert GEMM.
        p.weights[token*TopK+rank]=cutlass::bfloat16_t(probabilities[rank]/sum);
      }
    }
  }
};
} // namespace tilemega::codegen
