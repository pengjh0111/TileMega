// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <tilemega/Backend/Sm90DenseAttentionPartition.h>
#include <cutlass/bfloat16.h>

namespace tilemega::backend {
template<bool Stream,int D,int Q,int Chunks,int Keys,int Slots,int RowWork,int Capacity>
struct Sm90DenseKvPreissue {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  using Partition=Sm90DenseAttentionPartition<Chunks>;
  static constexpr auto kLayout=Sm90DenseLayout::Attention(
      Stream?8:1,D,Q,Chunks,Keys,Slots,RowWork,Capacity);
  template<int Bytes>
  __device__ static __forceinline__ void Prepare(Element const* key,Element const* value,
      int past,char* shared,Sm90DenseRingState& ring) {
    auto range=Partition::Range(past,int(blockIdx.x)%Chunks,RowWork);
    int bg=int(blockIdx.x)/Chunks,remaining=Bytes;
    constexpr int rows_per_slot=Stream?Keys:Partition::MaxKeys(Capacity,RowWork);
    constexpr int count=Stream?Slots:1;
    Async::ProxyAsyncGlobalFence();
    #pragma unroll
    for(int slot=0;slot<count;++slot) {
      int begin=range.begin+slot*rows_per_slot;
      int history=max(0,min(begin+rows_per_slot,min(range.end,past))-begin);
      int rows=min(history,remaining/(4*D));remaining-=rows*4*D;
      if(rows) {
        Async::Wait(ring.empty+slot,ring.producer_generations[slot]);
        Async::ExpectTx(ring.full+slot,4*history*D);
        std::size_t offset=(std::size_t(bg)*Capacity+begin)*D;
        Async::Bulk(shared+kLayout.key+slot*2*rows_per_slot*D,key+offset,2*rows*D,ring.full+slot);
        Async::Bulk(shared+kLayout.value+slot*2*rows_per_slot*D,value+offset,2*rows*D,ring.full+slot);
        ring.prefix_bytes[slot]=4*rows*D;
      }
    }
  }
  template<int Bytes,int Prefix>
  __device__ static __forceinline__ void Prefetch(Element const* key,Element const* value,int past) {
    auto range=Partition::Range(past,int(blockIdx.x)%Chunks,RowWork);
    int history=min(range.end,past)-range.begin,done=min(history,Prefix/(4*D));
    int bytes=min(Bytes/2,2*D*(history-done))&~15;
    if(bytes) {
      Async::ProxyAsyncGlobalFence();
      std::size_t offset=(std::size_t(int(blockIdx.x)/Chunks)*Capacity+range.begin+done)*D;
      Async::Prefetch(key+offset,bytes);Async::Prefetch(value+offset,bytes);
    }
  }
};
}  // namespace tilemega::backend
