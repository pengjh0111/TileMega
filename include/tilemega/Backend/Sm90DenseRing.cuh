// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <tilemega/Codegen/executor/Async.cuh>

namespace tilemega::backend {

// This header survives all stages in one launch. Operand storage starts at
// byte 1024; epilogues never alias either an in-flight slot or these barriers.
struct alignas(16) Sm90DenseRingState {
  std::uint64_t full[8],empty[8];
  // The phase bits keep the 256-byte ring ABI with a third residual barrier.
  std::uint64_t side[kSm90DenseResidualTiles];
  unsigned producer_generations[8],consumer_generations[8];
  // D10c (clusters only): the multicast issuer of slot s (rank s % 2 of its
  // CTA pair) waits on peer[s / 2] for the peer's 128 consumer releases.
  std::uint64_t peer[4];
  unsigned side_phase;
#if TILEMEGA_SM80_WIDE_PRODUCER || TILEMEGA_DENSE_SERVICE_PRODUCERS == 64
  template<bool CpAsync=false,unsigned Producers=32>
#else
  template<bool CpAsync=false>
#endif
  __device__ void Initialize() {
    using Async=codegen::executor::Async<std::conditional_t<CpAsync,arch::Sm80,arch::Sm90>>;
#if TILEMEGA_SM80_WIDE_PRODUCER || TILEMEGA_DENSE_SERVICE_PRODUCERS == 64
    static_assert(Producers==32 || (CpAsync && Producers==64));
    constexpr unsigned producers=CpAsync?Producers:1;
#else
    constexpr unsigned producers=CpAsync?32:1;
#endif
    if(threadIdx.x==0) {
      for(int i=0;i<8;++i) {
        Async::Init(full+i,producers);Async::Init(empty+i,128);
        producer_generations[i]=0;consumer_generations[i]=0;
      }
      for(int i=0;i<kSm90DenseResidualTiles;++i)Async::Init(side+i,producers);
      side_phase=0;
      Async::InitFence();
    }
    __syncthreads();
    if(threadIdx.x<128)for(int i=0;i<8;++i)Async::Arrive(empty+i);
    __syncthreads();
  }
  template<bool CpAsync=false>
  __device__ void Invalidate() {
    using Async=codegen::executor::Async<std::conditional_t<CpAsync,arch::Sm80,arch::Sm90>>;
    __syncthreads();
    if(threadIdx.x==0) {
      for(int i=0;i<8;++i) {Async::Invalidate(full+i);Async::Invalidate(empty+i);}
      for(int i=0;i<kSm90DenseResidualTiles;++i)Async::Invalidate(side+i);
    }
    __syncthreads();
  }
};
static_assert(2*sizeof(Sm90DenseRingState)<=1024);

struct Sm90DenseNoTail {
  __device__ void operator()() const {}
};


}  // namespace tilemega::backend
