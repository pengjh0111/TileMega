// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/Async.cuh>

#ifndef TILEMEGA_NATIVE_WEIGHT_EVICT_FIRST
#define TILEMEGA_NATIVE_WEIGHT_EVICT_FIRST 0
#endif

namespace tilemega::backend {

// E1 (RC6): a MoE layer streams 75 MB (B1) to 778 MB (B16) of expert weights
// through a 50 MB L2, so by default it evicts the data the serial tails then
// re-read -- P, h, the logits and the stage code. Every stream that is read
// exactly once takes an evict_first policy, which leaves the reused data
// resident. Activations, the residual and the embedding table keep the default
// policy: they are read again within the step.
//
// D0 measured the effect this targets: attention's Prepare costs 5.34 us in
// situ against a 0.99 us probe (5.4x), and the router's last arriver 12.48 us
// against 3.39 us (3.7x), where the probe keeps its inputs in L2.
//
// Only MoE programs define the macro, so a dense program's generated source and
// its PTX are unchanged.
struct Sm90DenseStream {
  using Async=codegen::executor::Async<arch::Sm90>;

  __device__ static __forceinline__ std::uint64_t WeightPolicy() {
#if TILEMEGA_NATIVE_WEIGHT_EVICT_FIRST
    return Async::EvictFirst();
#else
    return 0;
#endif
  }

  __device__ static __forceinline__ void WeightBulk(void* destination,void const* source,
      unsigned bytes,std::uint64_t* barrier,std::uint64_t policy) {
#if TILEMEGA_NATIVE_WEIGHT_EVICT_FIRST
    Async::BulkHint(destination,source,bytes,barrier,policy);
#else
    (void)policy;Async::Bulk(destination,source,bytes,barrier);
#endif
  }

  __device__ static __forceinline__ void HistoryTensor4D(void* destination,void const* map,
      int x,int y,int z,int w,std::uint64_t* barrier,std::uint64_t policy) {
#if TILEMEGA_NATIVE_WEIGHT_EVICT_FIRST
    Async::Tensor4DHint(destination,map,x,y,z,w,barrier,policy);
#else
    (void)policy;Async::Tensor4D(destination,map,x,y,z,w,barrier);
#endif
  }
};

}  // namespace tilemega::backend
