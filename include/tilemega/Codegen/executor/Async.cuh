// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/ArchDispatch.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <tilemega/Codegen/executor/Watchdog.cuh>

namespace tilemega::codegen::executor {
// PTX target requirements: mbarrier/test_wait/cp.async start at SM80;
// try_wait, transaction counts, bulk/TMA and grid dependency control at SM90.
// https://docs.nvidia.com/cuda/parallel-thread-execution/
template<class Arch=arch::CurrentArch, bool ForceSm80=false>
struct Async {
  using Selected=std::conditional_t<ForceSm80,arch::Sm80,Arch>;
  using Caps=arch::Caps<Selected>;
  __device__ static std::uint32_t Shared(void const* pointer) {
    return static_cast<std::uint32_t>(__cvta_generic_to_shared(pointer));
  }
  __device__ static void Init(std::uint64_t* barrier,unsigned arrivals) {
    if constexpr(Caps::kMbarrier)
      asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(Shared(barrier)),"r"(arrivals):"memory");
  }
  __device__ static void InitFence() {
    if constexpr(Caps::kMbarrierTx)
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __device__ static void ProxyAsyncGlobalFence() {
    if constexpr(Caps::kBulkCopy)
      asm volatile("fence.proxy.async.global;" ::: "memory");
  }
  __device__ static void Arrive(std::uint64_t* barrier) {
    if constexpr(Caps::kMbarrier)
      asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(Shared(barrier)):"memory");
  }
  __device__ static bool Ready(std::uint64_t* barrier,unsigned phase) {
    unsigned ready=0;
    if constexpr(Caps::kMbarrierTryWait)
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
          :"=r"(ready):"r"(Shared(barrier)),"r"(phase):"memory");
    else if constexpr(Caps::kMbarrier)
      asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
          :"=r"(ready):"r"(Shared(barrier)),"r"(phase):"memory");
    return ready!=0;
  }
  __device__ static void Wait(std::uint64_t* barrier,unsigned phase) {
    while(!Ready(barrier,phase)) {}
  }
  __device__ static void Wait(std::uint64_t* barrier,unsigned phase,
                              Watch const* watch,unsigned site,unsigned long long row) {
    unsigned long long start=0,failures=0;
    while(!Ready(barrier,phase)) {
      if(watch) {
        Watch here=*watch;here.site=site;here.row=row;
        WatchExpired(&here,start,++failures,phase,0);
      }
    }
  }
  __device__ static void Copy16Bytes(void* destination,void const* source,unsigned bytes) {
    if constexpr(Caps::kCpAsync)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
          "r"(Shared(destination)),"l"(source),"r"(bytes):"memory");
  }
  __device__ static void Copy16(void* destination,void const* source,bool valid=true) {
    Copy16Bytes(destination,source,valid?16:0);
  }
  __device__ static std::uint64_t EvictFirst() {
    std::uint64_t policy=0;
    if constexpr(Caps::kCpAsync)
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    return policy;
  }
  __device__ static void Copy16Hint(void* destination,void const* source,
                                    std::uint64_t policy) {
    if constexpr(Caps::kCpAsync)
      asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::
          "r"(Shared(destination)),"l"(source),"l"(policy):"memory");
  }
  __device__ static void CompleteCopies(std::uint64_t* barrier) {
    if constexpr(Caps::kCpAsync && Caps::kMbarrier)
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(Shared(barrier)):"memory");
  }
  __device__ static void ExpectTx(std::uint64_t* barrier,unsigned bytes) {
    if constexpr(Caps::kMbarrierTx)
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
          "r"(Shared(barrier)),"r"(bytes):"memory");
  }
  __device__ static void Bulk(void* destination,void const* source,unsigned bytes,std::uint64_t* barrier) {
    if constexpr(Caps::kBulkCopy)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(Shared(destination)),"l"(source),"r"(bytes),"r"(Shared(barrier)):"memory");
  }
  __device__ static void BulkHint(void* destination,void const* source,unsigned bytes,
                                  std::uint64_t* barrier,std::uint64_t policy) {
    if constexpr(Caps::kBulkCopy)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::
          "r"(Shared(destination)),"l"(source),"r"(bytes),
          "r"(Shared(barrier)),"l"(policy):"memory");
  }
  __device__ static void Tensor2D(void* destination,void const* map,int x,int y,std::uint64_t* barrier) {
    if constexpr(Caps::kTma)
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::
          "r"(Shared(destination)),"l"(map),"r"(x),"r"(y),"r"(Shared(barrier)):"memory");
  }
  __device__ static void Prefetch(void const* source,unsigned bytes) {
    if constexpr(Caps::kBulkPrefetch)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(source),"r"(bytes):"memory");
    else if constexpr(Caps::kCpAsync)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(source):"memory");
  }
  __device__ static void PrefetchEvictLast(void const* source,unsigned bytes) {
    if constexpr(Caps::kBulkPrefetch) {
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::
          "l"(source),"r"(bytes):"memory");
    } else if constexpr(Caps::kCpAsync) {
      asm volatile("prefetch.global.L2::evict_last [%0];" ::
          "l"(source):"memory");
    }
  }
  __device__ static void WaitPreviousGrid() {
    if constexpr(Caps::kPdl)asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  __device__ static void LaunchDependents() {
    if constexpr(Caps::kPdl)asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
};
} // namespace tilemega::codegen::executor
