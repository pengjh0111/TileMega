// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/tasks/ClusterSync.cuh>

namespace tilemega::codegen::executor {

// All ranks execute the same ordered episodes. The ready handshake publishes
// expect_tx before any peer push; the acknowledgement protects source reuse.
template<class Arch,unsigned Ranks,unsigned Capacity>
struct ClusterMailbox {
  static_assert(Ranks==2 || Ranks==4,"mailbox requires two or four ranks");
  static_assert(arch::Caps<Arch>::kDsmem && arch::Caps<Arch>::kBulkCopy,
                "mailbox requires DSMEM bulk copies");
  using A=Async<Arch>;
  using CS=ClusterSync<Arch>;
  struct alignas(1024) Storage {
    std::uint64_t full[2][Ranks-1];
    std::uint64_t ready[2],ack[2];
    unsigned episode;
    alignas(16) float partial[2][Ranks-1][Capacity];
  };
  __device__ static __forceinline__ void Init(Storage& s) {
    if(threadIdx.x==0) {
      s.episode=0;
      for(unsigned slot=0;slot<2;++slot) {
        A::Init(&s.ready[slot],1);A::Init(&s.ack[slot],1);
        for(unsigned rank=1;rank<Ranks;++rank)A::Init(&s.full[slot][rank-1],1);
      }
      A::InitFence();
    }
    CS::Sync();
  }
  template<unsigned Elements,class Compute,class Finish>
  __device__ static __forceinline__ bool Run(Storage& s,Compute compute,Finish finish) {
    static_assert(Elements<=Capacity && Elements%4==0,"invalid mailbox tile");
    unsigned const rank=CS::Rank(),episode=s.episode;
    unsigned const slot=episode&1,phase=(episode/2)&1;
    if(rank==0 && threadIdx.x==0) {
      for(unsigned peer=1;peer<Ranks;++peer) {
        A::ExpectTx(&s.full[slot][peer-1],Elements*sizeof(float));
        A::RemoteArrive(A::Mapa(&s.ready[slot],peer));
      }
    }
    if(rank!=0)A::Wait(&s.ready[slot],phase);
    __syncthreads();
    float* tile=compute();
    A::FenceProxyAsyncShared();
    __syncthreads();
    if(rank!=0 && threadIdx.x==0)
      A::BulkSmemToPeer(A::Mapa(s.partial[slot][rank-1],0),tile,
          Elements*sizeof(float),A::Mapa(&s.full[slot][rank-1],0));
    if(rank==0) {
      for(unsigned peer=1;peer<Ranks;++peer)A::Wait(&s.full[slot][peer-1],phase);
      __syncthreads();
      for(unsigned i=threadIdx.x;i<Elements;i+=blockDim.x) {
        float sum=tile[i];
        #pragma unroll
        for(unsigned peer=1;peer<Ranks;++peer)sum+=s.partial[slot][peer-1][i];
        tile[i]=sum;
      }
      __syncthreads();
      finish(tile);
      __syncthreads();
      if(threadIdx.x==0)for(unsigned peer=1;peer<Ranks;++peer)
        A::RemoteArrive(A::Mapa(&s.ack[slot],peer));
    } else A::Wait(&s.ack[slot],phase);
    __syncthreads();
    if(threadIdx.x==0)++s.episode;
    __syncthreads();
    return rank==0;
  }
  // Exchange retains each rank's numerical state for reductions such as softmax.
  template<unsigned Elements,class Compute,class Finish>
  __device__ static __forceinline__ bool Exchange(Storage& s,Compute compute,Finish finish) {
    static_assert(Elements<=Capacity && Elements%4==0,"invalid mailbox exchange tile");
    unsigned const rank=CS::Rank(),episode=s.episode;
    unsigned const slot=episode&1,phase=(episode/2)&1;
    if(rank==0 && threadIdx.x==0) {
      for(unsigned peer=1;peer<Ranks;++peer) {
        A::ExpectTx(&s.full[slot][peer-1],Elements*sizeof(float));
        A::RemoteArrive(A::Mapa(&s.ready[slot],peer));
      }
    }
    if(rank!=0)A::Wait(&s.ready[slot],phase);
    __syncthreads();
    float* tile=compute();
    A::FenceProxyAsyncShared();
    __syncthreads();
    if(rank!=0 && threadIdx.x==0)
      A::BulkSmemToPeer(A::Mapa(s.partial[slot][rank-1],0),tile,
          Elements*sizeof(float),A::Mapa(&s.full[slot][rank-1],0));
    if(rank==0) {
      for(unsigned peer=1;peer<Ranks;++peer)A::Wait(&s.full[slot][peer-1],phase);
      __syncthreads();
      finish(tile,s.partial[slot]);
      __syncthreads();
      if(threadIdx.x==0)for(unsigned peer=1;peer<Ranks;++peer)
        A::RemoteArrive(A::Mapa(&s.ack[slot],peer));
    }else A::Wait(&s.ack[slot],phase);
    __syncthreads();
    if(threadIdx.x==0)++s.episode;
    __syncthreads();
    return rank==0;
  }
  // A DSMEM target must remain alive until every peer stops using it.
  __device__ static __forceinline__ void Close() {CS::Sync();}
};
} // namespace tilemega::codegen::executor
