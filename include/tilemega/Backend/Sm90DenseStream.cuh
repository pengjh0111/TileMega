// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/Async.cuh>
#include <type_traits>

#ifndef TILEMEGA_NATIVE_WEIGHT_EVICT_FIRST
#define TILEMEGA_NATIVE_WEIGHT_EVICT_FIRST 0
#endif

#ifndef TILEMEGA_SM80_WEIGHT_EVICT_FIRST
#define TILEMEGA_SM80_WEIGHT_EVICT_FIRST 0
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
template<bool CpAsync>
struct Sm90DenseStreamT {
  using Async=codegen::executor::Async<std::conditional_t<CpAsync,arch::Sm80,arch::Sm90>>;

  __device__ static __forceinline__ std::uint64_t WeightPolicy() {
    if constexpr(CpAsync && TILEMEGA_SM80_WEIGHT_EVICT_FIRST) {
      return Async::EvictFirst();
    } else if constexpr(!CpAsync && TILEMEGA_NATIVE_WEIGHT_EVICT_FIRST) {
      return Async::EvictFirst();
    }
    return 0;
  }

  __device__ static __forceinline__ void WeightCopy16(void* destination,void const* source,
      std::uint64_t policy,bool valid=true) {
    if constexpr(CpAsync && TILEMEGA_SM80_WEIGHT_EVICT_FIRST) {
      if(valid)Async::Copy16Hint(destination,source,policy);
      else Async::Copy16(destination,source,false);
    } else {
      Async::Copy16(destination,source,valid);
    }
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

using Sm90DenseStream=Sm90DenseStreamT<false>;

}  // namespace tilemega::backend
