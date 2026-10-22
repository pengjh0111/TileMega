// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Codegen/tasks/mx/DecodeBarrier.cuh
//
// The MX-4 decode program's stage barrier (MX4 prompt §3.3), shared by the
// generated program and experiments/mx4/stage_bench.cu so both time the
// same thing:
//
//   __syncthreads(); thread 0 arrives (fence + atomicAdd);
//   every thread issues the next stage's prefetch (global-space loads of
//   data with no producer: weights, the KV history) into registers;
//   thread 0 waits (volatile poll, then fence); __syncthreads().
//
// The prefetch is issued after the arrival, so it is not ordered by the
// arrival's fence. TILEMEGA_MX_BARRIER_PREFETCH=0 (§6 D4) issues it after the
// wait instead. Polling never sleeps; the watchdog is compiled in only when
// TILEMEGA_WATCHDOG=1 (unit tests, C-1/C-2), not in timed builds.
#pragma once

#include <tilemega/Codegen/executor/Watchdog.cuh>
#include <tilemega/Codegen/tasks/EventSync.cuh>

#ifndef TILEMEGA_MX_BARRIER_PREFETCH
#define TILEMEGA_MX_BARRIER_PREFETCH 1
#endif

namespace tilemega::codegen::mx {

struct NoPrefetch {};

/// The watchdog of a barrier wait, as scalars: a struct passed by reference
/// to the shared (non-inlined) watched wait put a stack frame in the program.
struct MxWatch {
  WatchdogRecord* record;
  unsigned long long limit_ns;
  unsigned stage;
  unsigned long long iteration;
};

/// WaitAtLeast with the watchdog of Watchdog.cuh (site 11, grid barrier):
/// every 4096 polls past the limit it records the wait and traps.
__device__ __forceinline__ void WaitWatched(unsigned long long const* counter, unsigned long long target,
                                            MxWatch const watch) {
  unsigned long long start = 0, failures = 0;
  for (;;) {
    auto const value = maca::EventPeek(counter);
    if (value >= target) break;
    if (watch.record && watch.limit_ns && !(++failures & 4095ull)) {
      auto const now = maca::GlobalTimerNs();
      if (!start) start = now;
      if (now - start > watch.limit_ns && atomicCAS(&watch.record->fired, 0ull, 1ull) == 0ull) {
        watch.record->site = 11;
        watch.record->block = blockIdx.x;
        watch.record->thread = threadIdx.x;
        watch.record->waiter_stage = watch.stage;
        watch.record->need = target;
        watch.record->value = value;
        watch.record->iteration = watch.iteration;
        __threadfence_system();
        maca::Trap();
      }
    }
  }
  maca::FenceDevice();
}

/// `counter`: this stage's 64-bit arrival counter (its own 64 B line, never
/// reset between steps); the wait ends at `target` = G * (iteration + 1).
template <class Prefetch>
__device__ __forceinline__ auto Barrier(unsigned long long* counter, unsigned long long target, Prefetch prefetch,
                                        MxWatch const watch = MxWatch{nullptr, 0, 0, 0}) {
  __syncthreads();
  if (threadIdx.x == 0) RedRelease(counter, 1ull);
#if TILEMEGA_MX_BARRIER_PREFETCH
  auto pf = prefetch();
#endif
  if (threadIdx.x == 0) {
#if TILEMEGA_WATCHDOG
    WaitWatched(counter, target, watch);
#else
    (void)watch;
    WaitAtLeast(counter, target);
#endif
  }
  __syncthreads();
#if !TILEMEGA_MX_BARRIER_PREFETCH
  auto pf = prefetch();
#endif
  return pf;
}

}  // namespace tilemega::codegen::mx
