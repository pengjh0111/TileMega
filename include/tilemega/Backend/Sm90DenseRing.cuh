// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/Async.cuh>

namespace tilemega::backend {

// This header survives all stages in one launch. Operand storage starts at
// byte 1024; epilogues never alias either an in-flight slot or these barriers.
struct alignas(16) Sm90DenseRingState {
  std::uint64_t full[8],empty[8];
  unsigned producer_generations[8],consumer_generations[8];
  unsigned prefix_bytes[8];
  using Async=codegen::executor::Async<arch::Sm90>;
  __device__ void Initialize() {
    if(threadIdx.x==0) {
      for(int i=0;i<8;++i) {
        Async::Init(full+i,33);Async::Init(empty+i,128);
        producer_generations[i]=0;consumer_generations[i]=0;
        prefix_bytes[i]=0;
      }
      Async::InitFence();
    }
    __syncthreads();
    if(threadIdx.x<128)for(int i=0;i<8;++i)Async::Arrive(empty+i);
    __syncthreads();
  }
  __device__ void Invalidate() {
    __syncthreads();
    if(threadIdx.x==0)for(int i=0;i<8;++i) {
      Async::Invalidate(full+i);Async::Invalidate(empty+i);
    }
    __syncthreads();
  }
  __device__ void RetireOperands() {
    for(int i=0;i<8;++i)Async::Wait(empty+i,producer_generations[i]);
  }
};
static_assert(2*sizeof(Sm90DenseRingState)<=1024);

struct Sm90DenseNoTail {
  __device__ void operator()() const {}
};


}  // namespace tilemega::backend
