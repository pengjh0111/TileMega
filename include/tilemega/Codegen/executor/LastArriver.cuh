// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cuda/atomic>
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
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> counter(*ticket);
        unsigned previous = counter.fetch_add(contribution, cuda::memory_order_acq_rel);
        if (total == 0 || contribution > total || previous > total - contribution)
          asm volatile("trap;");
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
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> counter(*ticket);
        counter.store(0, cuda::memory_order_release);
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
      cuda::atomic_ref<unsigned,cuda::thread_scope_device> counter(*ticket);
      unsigned previous=counter.fetch_add(1,cuda::memory_order_acq_rel);
      if(previous>=producers || producers==0)asm volatile("trap;");
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
        cuda::atomic_ref<unsigned,cuda::thread_scope_device> counter(*ticket);
        counter.store(0,cuda::memory_order_release);
      }
      ComputeSync();
    }
    return last;
  }
};
} // namespace tilemega::codegen::executor
