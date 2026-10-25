// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/executor/Watchdog.cuh>
#include <tilemega/Codegen/tasks/mx/StageSyncV2.cuh>

#ifndef TILEMEGA_MX_DIAG
#define TILEMEGA_MX_DIAG 0
#endif
#ifndef TILEMEGA_TRACE_STAGE
#define TILEMEGA_TRACE_STAGE 0
#endif

namespace tilemega::codegen::mx {

constexpr int StageAlign16V2(int bytes) { return (bytes + 15) / 16 * 16; }

struct StageWatchV2 {
  WatchdogRecord* record;
  unsigned long long limit_ns;
  unsigned stage;
  unsigned long long iteration;
};

// P-B has already executed its final CTA sync inside Body::Run. A sync in
// this wrapper would drain the next body's weight/KV prefetch.
template <class Sync, class Broadcast>
__device__ __forceinline__ void StageBarrierV2(unsigned long long* counter, unsigned long long target,
                                               unsigned* generation, unsigned value, Broadcast broadcast,
                                               StageWatchV2 watch) {
  if (Sync::IsRole()) {
    Sync::ReleaseAdd(counter, 1ull);
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
    Sync::Acquire();
    broadcast();
    Sync::Publish(generation, value);
  }
#if !TILEMEGA_WATCHDOG
  (void)watch;
#endif
  Sync::Await(generation, value);
}

template <class Body, class Prefetched, class Operands, class Shape, class Next>
__device__ __forceinline__ auto RunStageV2(bool enabled, Prefetched pf, Operands const& operands,
                                          Shape const& shape, char* smem, Next next) {
#if TILEMEGA_MX_DIAG
  if (!enabled) {
    using Sync = StageSyncV2<Shape::kThreads>;
    // A skipped consumer must retire its incoming prefetch before the CTA
    // sync; the next prefetch still follows the selected P-B issue point.
    __builtin_mxc_arrive_gvmcnt(0);
    Sync::CtaSync();
    using Result = decltype(next());
    if (Sync::IsCompute()) return next();
    return Result{};
  }
#else
  (void)enabled;
#endif
  return Body::Run(pf, operands, shape, smem, next);
}

}  // namespace tilemega::codegen::mx
