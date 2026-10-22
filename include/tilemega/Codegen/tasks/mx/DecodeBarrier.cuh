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

#include <tilemega/Codegen/tasks/EventSync.cuh>

#ifndef TILEMEGA_MX_BARRIER_PREFETCH
#define TILEMEGA_MX_BARRIER_PREFETCH 1
#endif

namespace tilemega::codegen::mx {

struct NoPrefetch {};

/// `counter`: this stage's 64-bit arrival counter (its own 64 B line, never
/// reset between steps); the wait ends at `target` = G * (iteration + 1).
template <class Prefetch>
__device__ __forceinline__ auto Barrier(unsigned long long* counter, unsigned long long target, Prefetch prefetch,
                                        Watch const* watch = nullptr) {
  __syncthreads();
  if (threadIdx.x == 0) RedRelease(counter, 1ull);
#if TILEMEGA_MX_BARRIER_PREFETCH
  auto pf = prefetch();
#endif
  if (threadIdx.x == 0) {
#if TILEMEGA_WATCHDOG
    if (watch) WaitAtLeast(counter, target, *watch);
    else WaitAtLeast(counter, target);
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
