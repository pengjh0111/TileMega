// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Target/Maca/DevicePrimitives.h>

namespace tilemega::codegen::mx {

// MX-5 DT-W/R/P: 576 threads, fence + atomicAdd, P-B. This policy is an
// explicit client choice; it does not change the MX-4 executor's defaults.
template <int Threads>
struct StageSyncV2 {
  static_assert(Threads == 512 || Threads == 576);
  static constexpr int kThreads = Threads;
  static constexpr unsigned kRole = Threads == 576 ? 512 : 0;

  __device__ __forceinline__ static bool IsRole() { return threadIdx.x == kRole; }
  __device__ __forceinline__ static bool IsCompute() { return threadIdx.x < 512; }
  __device__ __forceinline__ static void CtaSync() { __syncthreads(); }
  __device__ __forceinline__ static void LdsSync() { CtaSync(); }
  __device__ __forceinline__ static void Acquire() { __threadfence(); }

  template <class T>
  __device__ __forceinline__ static T ReleaseAdd(T* counter, T value) {
    __threadfence();
    return atomicAdd(counter, value);
  }

  __device__ __forceinline__ static void Publish(unsigned* generation, unsigned value) {
    // Broadcast stores must reach LDS before the generation becomes visible.
    asm volatile("" ::: "memory");
    __builtin_mxc_arrive_bsmcnt(0);
    *reinterpret_cast<volatile unsigned*>(generation) = value;
  }

  __device__ __forceinline__ static void Await(unsigned const* generation, unsigned value) {
    while (*reinterpret_cast<volatile unsigned const*>(generation) < value) {}
    asm volatile("" ::: "memory");
  }

  // The caller's final CTA sync precedes its next-stage prefetch (P-B).
  // There can be no CTA sync here while that prefetch is still in flight.
  template <class Broadcast>
  __device__ __forceinline__ static void Barrier(unsigned long long* counter, unsigned long long target,
                                                 unsigned* generation, unsigned value, Broadcast broadcast) {
    if (IsRole()) {
      ReleaseAdd(counter, 1ull);
      using GlobalCounter = __attribute__((address_space(1))) volatile unsigned long long const;
      while (*(GlobalCounter*)counter < target) {}
      Acquire();
      broadcast();
      Publish(generation, value);
    }
    Await(generation, value);
  }
};

}  // namespace tilemega::codegen::mx
