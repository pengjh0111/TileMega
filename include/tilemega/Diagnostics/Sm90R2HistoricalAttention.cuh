// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#ifndef TILEMEGA_SM90R2_HISTORICAL_ABLATION
#error "Historical attention is a reconstruction control, never a native plan option"
#endif
#include <tilemega/Codegen/tasks/Sm90DenseProgram.cuh>
#include <tilemega/Codegen/tasks/IndependentAttentionTaskBody.h>
#include <tilemega/Codegen/tasks/AttentionMergeTaskBody.h>

namespace tilemega::codegen::native_dense {

// Only the offline ablation builder includes this header. Production native
// lowering has no call site or coordinate that can select these stages.
template<int Batch,int D,int Q,bool Norm,int Extent,int Capacity,
         unsigned Qkv,unsigned Key,unsigned Value,unsigned Output,unsigned Cosine,
         unsigned Sine,unsigned QNorm,unsigned KNorm,unsigned Partial,unsigned Lse>
struct HistoricalAttention {
  using Body=IndependentAttentionTaskBody<arch::Sm90,D,Q,Norm>;
  static constexpr int kActivationTileK=0,kActivationSlots=0,kRingSlots=0,kMaxPreissue=0;
  static constexpr bool kArgmax=false;
  static constexpr int kBlocks=(Capacity+Extent-1)/Extent,kTasks=Batch*8*kBlocks;
  static constexpr int kSharedBytes=1024+sizeof(typename Body::SharedStorage);
  static constexpr backend::Sm90DenseLayout kLayout{0,0,1024,kSharedBytes};
  template<int Bytes>static constexpr backend::Sm90DensePayload Payload(){return {};}
  template<int Prefix,class B,class Tail,bool Initialize=true>
  __device__ static __forceinline__ void Run(B const& b,int past,char* shared,float epsilon,
      backend::Sm90DenseRingState&,Tail) {
    static_assert(Prefix==0);
    if(threadIdx.x<128) {
      ServingAttentionOperands p{b.template Get<Qkv>(),b.template Get<Key>(),b.template Get<Value>(),
          b.template Get<Cosine>(),b.template Get<Sine>(),b.template Get<QNorm>(),b.template Get<KNorm>(),
          b.template Get<Output>(),reinterpret_cast<float*>(b.template Get<Partial>()),
          reinterpret_cast<float*>(b.template Get<Lse>()),Batch,8,Capacity,past,Extent,epsilon};
      for(int task=blockIdx.x;task<kTasks;task+=gridDim.x) {
        int bg=task/kBlocks;
        Body::Run(p,*reinterpret_cast<typename Body::SharedStorage*>(shared+1024),
            bg%Batch,bg/Batch,0,task%kBlocks);
        executor::ComputeSync();
      }
    }
    __syncthreads();
  }
};

template<int Batch,int D,int Q,int Extent,int Capacity,unsigned Partial,unsigned Lse,unsigned Output>
struct HistoricalMerge {
  static constexpr int kActivationTileK=0,kActivationSlots=0,kRingSlots=0,kMaxPreissue=0;
  static constexpr bool kArgmax=false;
  static constexpr int kTasks=Batch*8,kSharedBytes=1040;
  static constexpr backend::Sm90DenseLayout kLayout{0,0,1024,kSharedBytes};
  template<int Bytes>static constexpr backend::Sm90DensePayload Payload(){return {};}
  template<int Prefix,class B,class Tail,bool Initialize=true>
  __device__ static __forceinline__ void Run(B const& b,int past,char*,float,
      backend::Sm90DenseRingState&,Tail) {
    static_assert(Prefix==0);
    if(threadIdx.x<128)for(int task=blockIdx.x;task<kTasks;task+=gridDim.x)
      AttentionMergeTaskBody<D,Q,1>::Run(reinterpret_cast<float const*>(b.template Get<Partial>()),
          reinterpret_cast<float const*>(b.template Get<Lse>()),b.template Get<Output>(),
          task%Batch,task/Batch,8,Capacity,Extent,past);
    __syncthreads();
  }
};

}  // namespace tilemega::codegen::native_dense
