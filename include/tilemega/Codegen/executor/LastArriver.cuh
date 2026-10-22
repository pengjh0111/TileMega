// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
namespace tilemega::codegen::executor {
// Each producer stores its partial before arrival. The callback is the same
// fixed-order combine/merge used by a separate task, executed by the final CTA.
// Tickets are per output tile and cannot be shared between concurrent plans.
struct LastArriver {
  template<class Reduction>
  __device__ static bool RunWeighted(unsigned* ticket, unsigned total,
      unsigned contribution, unsigned* shared_last, Reduction reduce) {
    __threadfence(); ComputeSync();
    if (ComputeThread() == 0) {
      *shared_last = 0;
      if (contribution) {
        unsigned previous=maca::TicketAddAcqRel(ticket,unsigned(contribution));
        if (total == 0 || contribution > total || previous > total - contribution)
          maca::Trap();
        *shared_last = previous == total - contribution;
      }
    }
    ComputeSync();
    bool last = *shared_last != 0;
    ComputeSync();
    if (last) {
      reduce();
      __threadfence(); ComputeSync();
      if (ComputeThread() == 0) {
        maca::TicketStoreRelease(ticket,unsigned(0));
      }
      ComputeSync();
    }
    return last;
  }
  // Force-inlined: the MX-4 program build left it as a call whose closure
  // (the reduction lambda's captures) sat on the stack (measured).
  template<class Reduction>
  __device__ __forceinline__ static bool Run(unsigned* ticket,unsigned producers,unsigned* shared_last,
                            Reduction reduce) {
    __threadfence();ComputeSync();
    if(ComputeThread()==0) {
      unsigned previous=maca::TicketAddAcqRel(ticket,unsigned(1));
      if(previous>=producers || producers==0)maca::Trap();
      *shared_last=previous+1==producers;
    }
    ComputeSync();
    bool last=*shared_last!=0;
    // The reduction can overwrite its work area only after every warp has
    // consumed the independent flag.
    ComputeSync();
    if(last) {
      reduce();
      __threadfence();ComputeSync();
      if(ComputeThread()==0) {
        maca::TicketStoreRelease(ticket,unsigned(0));
      }
      ComputeSync();
    }
    return last;
  }
  /// Run for `count` items of one CTA at once (MX-4 decode attention): every
  /// thread fences, then thread i arrives for item i on `ticket(i)` (nullptr:
  /// item i has no ticket) with `producers(i)` producers; for each item this
  /// CTA arrived last on, the whole CTA calls `merge(i)`, in item order; then
  /// a fence and the merged items' tickets return to 0. `shared_last` holds
  /// `count` flags; count <= the block size.
  template<class Ticket,class Producers,class Merge>
  __device__ __forceinline__ static void RunMany(int count,unsigned* shared_last,Ticket ticket,Producers producers,
                                 Merge merge) {
    __threadfence();ComputeSync();
    if(int(ComputeThread())<count) {
      int const i=ComputeThread();
      unsigned* const t=ticket(i);
      unsigned last=0;
      if(t) {
        unsigned const total=producers(i);
        unsigned const previous=maca::TicketAddAcqRel(t,unsigned(1));
        if(previous>=total || total==0)maca::Trap();
        last=previous+1==total;
      }
      shared_last[i]=last;
    }
    ComputeSync();
    bool any=false;
    for(int i=0;i<count;++i)
      if(shared_last[i]) {
        merge(i);
        any=true;
      }
    if(any) {
      __threadfence();ComputeSync();
      if(int(ComputeThread())<count && shared_last[ComputeThread()])
        maca::TicketStoreRelease(ticket(ComputeThread()),unsigned(0));
    }
    // The flags may be overwritten by the next use only after every thread
    // has read them.
    ComputeSync();
  }
};
} // namespace tilemega::codegen::executor
