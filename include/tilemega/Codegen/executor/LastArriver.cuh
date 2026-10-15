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
  template<class Reduction>
  __device__ static bool Run(unsigned* ticket,unsigned producers,unsigned* shared_last,
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
};
} // namespace tilemega::codegen::executor
