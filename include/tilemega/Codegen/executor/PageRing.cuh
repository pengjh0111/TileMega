// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>

namespace tilemega::codegen::executor {
template<int PageBytes,int Pages,class Arch=arch::CurrentArch,bool ForceSm80=false>
struct PageRing {
  static_assert(PageBytes==8192 || PageBytes==16384);
  static_assert(Pages>0);
  using Copy=Async<Arch,ForceSm80>;
  struct alignas(16) Slot { std::uint64_t full,empty,generation; };
  Slot* slots;
  char* data;

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
    Copy::Wait(&slots[SlotIndex(sequence)].empty,Phase(sequence));
    if(LoaderLane()==0)
      *reinterpret_cast<volatile std::uint64_t*>(&slots[SlotIndex(sequence)].generation)=sequence;
  }
  __device__ void PublishCopies(std::uint64_t sequence) const {
    // All 32 loader lanes contribute one asynchronous arrival, even when a
    // lane has no valid copy in the last partial page.
    Copy::CompleteCopies(&slots[SlotIndex(sequence)].full);
  }
  __device__ void PublishBulk(std::uint64_t sequence,void const* source,unsigned bytes) const {
    auto* barrier=&slots[SlotIndex(sequence)].full;
    if(LoaderLane()==0) {
      Copy::ExpectTx(barrier,bytes);
      Copy::Bulk(Page(sequence),source,bytes,barrier);
    }else Copy::Arrive(barrier);
  }
  __device__ void AwaitFull(std::uint64_t sequence) const {
    // Independent attention warps can request a slot two generations ahead.
    // Parity alone would accept an old completion; the sequence tag prevents
    // that ABA. The loader writes it only after acquiring the empty slot.
    auto* generation=reinterpret_cast<volatile std::uint64_t*>(&slots[SlotIndex(sequence)].generation);
    while(*generation!=sequence) {}
    Copy::Wait(&slots[SlotIndex(sequence)].full,Phase(sequence));
  }
  __device__ void Release(std::uint64_t sequence) const {
    Copy::Arrive(&slots[SlotIndex(sequence)].empty);
  }
};
} // namespace tilemega::codegen::executor
