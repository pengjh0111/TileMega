// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cuda/atomic>
namespace tilemega::codegen::executor {
// Fixed producer cardinality makes each iteration occupy a disjoint interval.
// Inactive attention chunks still arrive, but merge reads only live partials.
struct MonotonicLastArriver {
  template<class Reduction>
  __device__ static bool Run(unsigned long long* ticket,unsigned producers,
      unsigned long long iteration,unsigned* shared_last,Reduction reduce) {
    __threadfence();ComputeSync();
    if(ComputeThread()==0) {
      cuda::atomic_ref<unsigned long long,cuda::thread_scope_device> counter(*ticket);
      auto previous=counter.fetch_add(1,cuda::memory_order_acq_rel);
      auto begin=iteration*producers,end=begin+producers;
      if(!producers || previous<begin || previous>=end)asm volatile("trap;");
      *shared_last=previous+1==end;
    }
    ComputeSync();
    bool last=*shared_last!=0;
    ComputeSync();
    if(last) {reduce();ComputeSync();}
    return last;
  }
};
}
