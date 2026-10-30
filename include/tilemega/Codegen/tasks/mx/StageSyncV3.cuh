// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/tasks/mx/StageProgram.cuh>

namespace tilemega::codegen::mx {

template <int Threads>
struct StageSyncV3 {
  static_assert(Threads == 512 || Threads == 576);
  static constexpr int kThreads = Threads;
  static constexpr int kDepthVgprLimit = Threads == 512 ? 192 : 160;
  static constexpr unsigned kRole = Threads == 576 ? 512 : 0;

  __device__ __forceinline__ static bool IsRole() { return threadIdx.x == kRole; }
  __device__ __forceinline__ static bool IsCompute() { return threadIdx.x < 512; }
  __device__ __forceinline__ static void CtaSync() { __syncthreads(); }
  __device__ __forceinline__ static void LdsSync() {
    asm volatile("" ::: "memory");
    __builtin_mxc_arrive_bsmcnt(0);
    __builtin_mxc_barrier_inst();
    asm volatile("" ::: "memory");
  }
  __device__ __forceinline__ static void Acquire() { __threadfence(); }

  // Tile completion keeps the existing last-arriver protocol. Stage arrival
  // below deliberately discards the atomic result before issuing prefetches.
  template <class T>
  __device__ __forceinline__ static T ReleaseAdd(T* counter, T value) {
    __threadfence();
    return atomicAdd(counter, value);
  }
  __device__ __forceinline__ static void Arrive(unsigned long long* counter) {
    if (IsRole()) {
      __threadfence();
      atomicAdd(counter, 1ull);
    }
  }
  __device__ __forceinline__ static void Publish(unsigned* generation, unsigned value) {
    asm volatile("" ::: "memory");
    __builtin_mxc_arrive_bsmcnt(0);
    *reinterpret_cast<volatile unsigned*>(generation) = value;
  }
  __device__ __forceinline__ static void Await(unsigned const* generation, unsigned value) {
    while (*reinterpret_cast<volatile unsigned const*>(generation) < value) {}
    asm volatile("" ::: "memory");
  }
  template <class Broadcast>
  __device__ __forceinline__ static void Wait(unsigned long long* counter, unsigned long long target,
      unsigned* generation, unsigned value, Broadcast broadcast, StageWatchV2 watch) {
    if (IsRole()) {
      using GlobalCounter = __attribute__((address_space(1))) volatile unsigned long long const;
#if TILEMEGA_WATCHDOG
      unsigned long long start = 0, failures = 0;
#endif
      for (;;) {
        unsigned long long const observed = *(GlobalCounter*)counter;
        if (observed >= target) break;
#if TILEMEGA_WATCHDOG
        if (watch.record && watch.limit_ns && !(++failures & 4095ull)) {
          unsigned long long const now = maca::GlobalTimerNs();
          if (!start) start = now;
          if (now - start > watch.limit_ns && atomicCAS(&watch.record->fired, 0ull, 1ull) == 0ull) {
            watch.record->site = 11;
            watch.record->block = blockIdx.x;
            watch.record->thread = threadIdx.x;
            watch.record->waiter_stage = watch.stage;
            watch.record->need = target;
            watch.record->value = observed;
            watch.record->iteration = watch.iteration;
            __threadfence_system();
            maca::Trap();
          }
        }
#endif
      }
      Acquire();
      broadcast();
      Publish(generation, value);
    }
#if !TILEMEGA_WATCHDOG
    (void)watch;
#endif
    Await(generation, value);
  }
};

}  // namespace tilemega::codegen::mx
