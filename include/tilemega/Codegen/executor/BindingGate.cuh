// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Codegen/executor/Watchdog.cuh>
#include <cuda/atomic>

namespace tilemega::codegen::executor {
struct BindingGate {
  unsigned long long* arrivals = nullptr;
  unsigned long long target = 0;
  Watch const* watch = nullptr;

  __device__ unsigned long long Value() const {
    if (!arrivals || !target) { asm volatile("trap;"); return 0; }
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> value(*arrivals);
    return value.load(cuda::memory_order_acquire);
  }
  // Lookahead must not spin: dispatch may be downstream of the page whose
  // loader is making this query. All loader lanes retain the same cursor.
  __device__ bool LoaderReady() const {
    unsigned ready=LoaderLane()==0 ? Value()>=target : 0;
    ready=__shfl_sync(0xffffffffu,ready,0);
    __syncwarp();
    return ready;
  }
  __device__ void WaitLoader() const {
    if (LoaderLane()==0) {
      unsigned long long start=0,failures=0,value;
      Watch here=watch?*watch:Watch{};here.site=13;
      while ((value=Value())<target) {
        if(watch)WatchExpired(&here,start,++failures,target,value);
        __nanosleep(64);
      }
    }
    __syncwarp();
  }
};
}  // namespace tilemega::codegen::executor
