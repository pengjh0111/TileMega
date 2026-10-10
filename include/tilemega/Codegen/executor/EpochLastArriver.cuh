// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/RuntimeDependencies.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cuda/atomic>

namespace tilemega::codegen::executor {
// Nonpaged handoffs retain a monotonic ticket across launches. The runtime
// must protect partial storage across epochs and use separate L1/L2 banks.
struct EpochLastArriver {
  template<class Reduce>
  __device__ static bool RunWeighted(unsigned long long* ticket,unsigned total,
      unsigned long long iteration,unsigned contribution,unsigned* shared_last,
      Reduce const& reduce) {
    std::uint64_t target=0;
    if(!ticket || !shared_last || contribution>total ||
       !CountedDependencyTarget(total,iteration,&target)) {
      asm volatile("trap;");return false;
    }
    __threadfence();
    ComputeSync();
    if(ComputeThread()==0) {
      *shared_last=0;
      if(contribution) {
        cuda::atomic_ref<unsigned long long,cuda::thread_scope_device> value(*ticket);
        auto previous=value.fetch_add(contribution,cuda::memory_order_acq_rel);
        if(previous>target || contribution>target-previous)asm volatile("trap;");
        *shared_last=previous+contribution==target;
      }
    }
    ComputeSync();
    if(*shared_last)reduce();
    ComputeSync();
    return *shared_last!=0;
  }
  template<class Reduce>
  __device__ static bool Run(unsigned long long* ticket,unsigned producers,
      unsigned long long iteration,unsigned* shared_last,Reduce const& reduce) {
    return RunWeighted(ticket,producers,iteration,1,shared_last,reduce);
  }
};
}  // namespace tilemega::codegen::executor
