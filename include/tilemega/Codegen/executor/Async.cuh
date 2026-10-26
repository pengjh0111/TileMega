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
  // All users must rendezvous before invalidation and before repurposing storage.
  __device__ static void Invalidate(std::uint64_t* barrier) {
    if constexpr(Caps::kMbarrier)
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(Shared(barrier)):"memory");
  }
  __device__ static void InitFence() {
    if constexpr(Caps::kMbarrierTx)
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __device__ static void ProxyAsyncGlobalFence() {
    if constexpr(Caps::kBulkCopy)
      asm volatile("fence.proxy.async.global;" ::: "memory");
  }
  __device__ static void FenceProxyAsyncShared() {
    if constexpr(Caps::kBulkCopy)
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __device__ static bool ElectOne() {
    unsigned elected=0;
    if constexpr(Caps::kElectSync)
      asm volatile("{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 %0, 1, 0, p; }"
          :"=r"(elected)::"memory");
    else elected=(threadIdx.x&31)==0;
    return elected!=0;
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
    Watch here=watch?*watch:Watch{};here.site=site;here.row=row;
    while(!Ready(barrier,phase)) {
      if(watch)WatchExpired(&here,start,++failures,phase,0);
    }
  }
  __device__ static void Copy16Bytes(void* destination,void const* source,unsigned bytes) {
    if constexpr(Caps::kCpAsync)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
          "r"(Shared(destination)),"l"(source),"r"(bytes):"memory");
  }
  __device__ static void Copy8Bytes(void* destination,void const* source,unsigned bytes) {
    if constexpr(Caps::kCpAsync)
      asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::
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
  __device__ static void Tensor3D(void* destination,void const* map,
      int channel,int x,int y,std::uint64_t* barrier) {
    static_assert(Caps::kTma,"3D halo copies require TMA support");
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4}], [%5];" :: "r"(Shared(destination)),"l"(map),
        "r"(channel),"r"(x),"r"(y),"r"(Shared(barrier)):"memory");
  }
  __device__ static void Tensor4D(void* destination,void const* map,
      int x,int y,int z,int w,std::uint64_t* barrier) {
    static_assert(Caps::kTma,"4D tensor copies require TMA support");
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4, %5}], [%6];" :: "r"(Shared(destination)),"l"(map),
        "r"(x),"r"(y),"r"(z),"r"(w),"r"(Shared(barrier)):"memory");
  }
  __device__ static void Tensor2DMulticast(void* destination,void const* map,
      int x,int y,std::uint64_t* barrier,std::uint16_t mask) {
    static_assert(Caps::kTmaMulticast,"TMA multicast requires cluster multicast support");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%2, %3}], [%4], %5;" ::
        "r"(Shared(destination)),"l"(map),"r"(x),"r"(y),
        "r"(Shared(barrier)),"h"(mask):"memory");
  }
  // D10c: one CTA's TMA box lands at the same offset in every CTA of mask,
  // completing each destination's barrier at the same offset.
  __device__ static void Tensor3DMulticast(void* destination,void const* map,
      int channel,int x,int y,std::uint64_t* barrier,std::uint16_t mask) {
    static_assert(Caps::kTmaMulticast,"TMA multicast requires cluster multicast support");
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%2, %3, %4}], [%5], %6;" ::
        "r"(Shared(destination)),"l"(map),"r"(channel),"r"(x),"r"(y),
        "r"(Shared(barrier)),"h"(mask):"memory");
  }
  __device__ static unsigned ClusterRank() {
    unsigned rank;asm volatile("mov.u32 %0, %%cluster_ctarank;":"=r"(rank));return rank;
  }
  // Every thread of every CTA of the cluster: earlier shared-memory writes
  // (local or DSMEM) become visible to the cluster.
  __device__ static void ClusterSync() {
    static_assert(Caps::kDsmem,"cluster barriers require DSMEM support");
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  }
  __device__ static void StoreCluster(std::uint32_t address,float value) {
    static_assert(Caps::kDsmem,"peer shared stores require DSMEM support");
    asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(address),"f"(value):"memory");
  }
  // A phase completed by peer CTAs' release.cluster arrivals.
  __device__ static void WaitCluster(std::uint64_t* barrier,unsigned phase) {
    static_assert(Caps::kDsmem,"cluster-scope barrier waits require DSMEM support");
    unsigned ready=0;
    do asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; "
        "selp.u32 %0, 1, 0, p; }" :"=r"(ready):"r"(Shared(barrier)),"r"(phase):"memory");
    while(!ready);
  }
  __device__ static std::uint32_t Mapa(void const* local,unsigned rank) {
    static_assert(Caps::kDsmem,"peer shared addresses require DSMEM support");
    std::uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" :"=r"(remote):
        "r"(Shared(local)),"r"(rank));
    return remote;
  }
  // The target must be a peer CTA; this instruction excludes local copies.
  __device__ static void BulkSmemToPeer(std::uint32_t destination,
      void const* source,unsigned bytes,std::uint32_t barrier) {
    static_assert(Caps::kDsmem && Caps::kBulkCopy,"peer bulk copies require DSMEM and bulk support");
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];" ::"r"(destination),"r"(Shared(source)),
        "r"(bytes),"r"(barrier):"memory");
  }
  __device__ static void RemoteArrive(std::uint32_t barrier) {
    static_assert(Caps::kDsmem,"remote barrier arrival requires DSMEM support");
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::
        "r"(barrier):"memory");
  }
  // Floating reductions have no fixed inter-CTA order and are opt-in only.
  __device__ static void BulkReduceAddF32(float* destination,
      float const* source,unsigned bytes) {
    static_assert(Caps::kBulkReduce,"bulk reduction requires reduction support");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
        "[%0], [%1], %2;" ::"l"(destination),"r"(Shared(source)),"r"(bytes):"memory");
  }
  __device__ static void BulkCommit() {
    static_assert(Caps::kBulkCopy,"bulk groups require bulk copy support");
    asm volatile("cp.async.bulk.commit_group;" :::"memory");
  }
  template<int Pending=0,bool ReadOnly=false>
  __device__ static void BulkWait() {
    static_assert(Caps::kBulkCopy,"bulk groups require bulk copy support");
    static_assert(Pending>=0 && Pending<=7,"bulk wait group count must be in [0,7]");
    if constexpr(ReadOnly)
      asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(Pending):"memory");
    else asm volatile("cp.async.bulk.wait_group %0;" ::"n"(Pending):"memory");
  }
  __device__ static void PrefetchTensorMap(void const* map) {
    static_assert(Caps::kTma,"tensor map prefetch requires TMA support");
    asm volatile("prefetch.tensormap [%0];" ::"l"(map):"memory");
  }
  __device__ static void TensorIm2col4D(void* destination,void const* map,
      int c,int w,int h,int n,std::uint16_t offset_w,std::uint16_t offset_h,
      std::uint64_t* barrier) {
    static_assert(Caps::kTmaIm2col,"TMA im2col requires an architecture with im2col support");
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4, %5}], [%6], {%7, %8};" ::
        "r"(Shared(destination)),"l"(map),"r"(c),"r"(w),"r"(h),"r"(n),
        "r"(Shared(barrier)),"h"(offset_w),"h"(offset_h):"memory");
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
