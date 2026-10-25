// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/Async.cuh>

namespace tilemega::backend {

// This header survives all stages in one launch. Operand storage starts at
// byte 1024; epilogues never alias either an in-flight slot or these barriers.
struct alignas(16) Sm90DenseRingState {
  std::uint64_t full[8],empty[8];
  // Residual tiles of a stage's (at most two) local tasks arrive here.
  std::uint64_t side[2];
  unsigned producer_generations[8],consumer_generations[8];
  // D10c (clusters only): the multicast issuer of slot s (rank s % 2 of its
  // CTA pair) waits on peer[s / 2] for the peer's 128 consumer releases.
  std::uint64_t peer[4];
  unsigned side_generations[2];
  using Async=codegen::executor::Async<arch::Sm90>;
  __device__ void Initialize() {
    if(threadIdx.x==0) {
      for(int i=0;i<8;++i) {
        // Every operand of a slot arrives by bulk/TMA under one expect_tx.
        Async::Init(full+i,1);Async::Init(empty+i,128);
        producer_generations[i]=0;consumer_generations[i]=0;
      }
      for(int i=0;i<2;++i) {Async::Init(side+i,1);side_generations[i]=0;}
      Async::InitFence();
    }
    __syncthreads();
    if(threadIdx.x<128)for(int i=0;i<8;++i)Async::Arrive(empty+i);
    __syncthreads();
  }
  __device__ void Invalidate() {
    __syncthreads();
    if(threadIdx.x==0) {
      for(int i=0;i<8;++i) {Async::Invalidate(full+i);Async::Invalidate(empty+i);}
      for(int i=0;i<2;++i)Async::Invalidate(side+i);
    }
    __syncthreads();
  }
};
static_assert(2*sizeof(Sm90DenseRingState)<=1024);

struct Sm90DenseNoTail {
  __device__ void operator()() const {}
};


}  // namespace tilemega::backend
