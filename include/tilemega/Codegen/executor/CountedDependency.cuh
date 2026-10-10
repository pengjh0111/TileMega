// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cuda/atomic>

namespace tilemega::codegen::executor {
struct CountedDependency {
  // Every contributing warp publishes its stores before the leader's release
  // RMW. The acquire reads the release sequence of all counted producers.
  __device__ static void Publish(unsigned long long* value, unsigned contribution) {
    __threadfence(); ComputeSync();
    if (ComputeThread() == 0 && contribution) {
      cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> counter(*value);
      counter.fetch_add(contribution, cuda::memory_order_acq_rel);
    }
    ComputeSync();
  }
  __device__ static void Wait(unsigned long long* value, unsigned long long target) {
    if (!target) asm volatile("trap;");
    if (ComputeThread() == 0) {
      cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> counter(*value);
      while (counter.load(cuda::memory_order_acquire) < target) __nanosleep(64);
    }
    ComputeSync();
  }
};
}  // namespace tilemega::codegen::executor
