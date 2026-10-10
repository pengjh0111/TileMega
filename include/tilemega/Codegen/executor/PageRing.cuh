// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Codegen/executor/PageTrace.cuh>
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
#include <cuda/atomic>
#endif

namespace tilemega::codegen::executor {
struct NoPageHook { __device__ void operator()(unsigned) const {} };
template<int PageBytes,int Pages,class Arch=arch::CurrentArch,bool ForceSm80=false>
struct PageRing {
  static constexpr int kPageBytes=PageBytes;
  static_assert(PageBytes==8192 || PageBytes==16384);
  static_assert(Pages>0);
  using Copy=Async<Arch,ForceSm80>;
  struct alignas(16) Slot { std::uint64_t full,empty,generation; };
  Slot* slots;
  char* data;
  PageTraceRecord* trace = nullptr;
  Watch* watch = nullptr;

  __device__ unsigned* SharedLastFlag() const {
    return reinterpret_cast<unsigned*>(slots + Pages);
  }

  __device__ void Initialize() const {
    if(ComputeThread()==0) {
      for(int p=0;p<Pages;++p) {
        slots[p].generation=~std::uint64_t(0);
        Copy::Init(&slots[p].full,kLoaderThreads);
        Copy::Init(&slots[p].empty,kComputeThreads);
      }
      Copy::InitFence();
    }
    // Initialization is the only rendezvous including the loader. Mainloop
    // barriers use ID 1 and exactly the four compute warps.
    asm volatile("bar.sync 0, 160;" ::: "memory");
    if(IsCompute())for(int p=0;p<Pages;++p)Copy::Arrive(&slots[p].empty);
  }
  __device__ static int SlotIndex(std::uint64_t sequence) { return sequence%Pages; }
  __device__ static unsigned Phase(std::uint64_t sequence) { return (sequence/Pages)&1; }
  __device__ char* Page(std::uint64_t sequence) const { return data+SlotIndex(sequence)*PageBytes; }
  __device__ void AcquireEmpty(std::uint64_t sequence) const {
    if(LoaderLane()==0)PageTraceTransition(trace,2u,true);
    Copy::Wait(&slots[SlotIndex(sequence)].empty,Phase(sequence),watch,8,sequence);
    if(LoaderLane()==0)PageTraceTransition(trace,2u,false);
#if TILEMEGA_PAGE_TRACE
    if(LoaderLane()==0 && trace)trace->loader_issue_begin_ns=PageTraceNow();
#endif
    if(LoaderLane()==0) {
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
      cuda::atomic_ref<std::uint64_t,cuda::thread_scope_block>(
          slots[SlotIndex(sequence)].generation).exchange(sequence,cuda::memory_order_release);
#else
      *reinterpret_cast<volatile std::uint64_t*>(&slots[SlotIndex(sequence)].generation)=sequence;
#endif
    }
  }
  __device__ void EndIssue() const {
#if TILEMEGA_PAGE_TRACE
    if(LoaderLane()==0 && trace)
      trace->loader_issue_ns+=PageTraceNow()-trace->loader_issue_begin_ns;
#endif
  }
  __device__ void PublishCopies(std::uint64_t sequence) const {
    // All 32 loader lanes contribute one asynchronous arrival, even when a
    // lane has no valid copy in the last partial page.
    Copy::CompleteCopies(&slots[SlotIndex(sequence)].full);EndIssue();
  }
  __device__ void PublishBulk(std::uint64_t sequence,void const* source,unsigned bytes) const {
    auto* barrier=&slots[SlotIndex(sequence)].full;
    if(LoaderLane()==0) {
      Copy::ExpectTx(barrier,bytes);
      Copy::Bulk(Page(sequence),source,bytes,barrier);
    }else Copy::Arrive(barrier);
    EndIssue();
  }
  __device__ void PublishBulkHint(std::uint64_t sequence,void const* source,
                                  unsigned bytes,std::uint64_t policy) const {
    auto* barrier=&slots[SlotIndex(sequence)].full;
    if(LoaderLane()==0) {
      Copy::ExpectTx(barrier,bytes);
      Copy::BulkHint(Page(sequence),source,bytes,barrier,policy);
    }else Copy::Arrive(barrier);
    EndIssue();
  }
  __device__ void AwaitFull(std::uint64_t sequence) const {
    // Independent attention warps can request a slot two generations ahead.
    // Parity alone would accept an old completion; the sequence tag prevents
    // that ABA. The loader writes it only after acquiring the empty slot.
    unsigned long long start=0,failures=0;
    Watch here=watch?*watch:Watch{};here.site=6;here.row=sequence;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    cuda::atomic_ref<std::uint64_t,cuda::thread_scope_block> generation(
        slots[SlotIndex(sequence)].generation);
    auto observed=generation.fetch_add(0,cuda::memory_order_acquire);
    while(observed!=sequence) {
      if(watch)WatchExpired(&here,start,++failures,sequence,observed);
      observed=generation.fetch_add(0,cuda::memory_order_acquire);
    }
#else
    auto* generation=reinterpret_cast<volatile std::uint64_t*>(&slots[SlotIndex(sequence)].generation);
    while(*generation!=sequence) {
      if(watch)WatchExpired(&here,start,++failures,sequence,*generation);
    }
#endif
    Copy::Wait(&slots[SlotIndex(sequence)].full,Phase(sequence),watch,7,sequence);
  }
  __device__ void Release(std::uint64_t sequence) const {
    Copy::Arrive(&slots[SlotIndex(sequence)].empty);
  }
};
} // namespace tilemega::codegen::executor
