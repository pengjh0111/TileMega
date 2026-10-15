// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <tilemega/Codegen/executor/Watchdog.cuh>

#ifndef TILEMEGA_SYNC_V3
#define TILEMEGA_SYNC_V3 0
#endif
#ifndef TILEMEGA_V3_POLL_NS
#define TILEMEGA_V3_POLL_NS 0
#endif

#ifndef TILEMEGA_EVENT_LOAD_POLL
#define TILEMEGA_EVENT_LOAD_POLL 0
#endif

// EX-E3 step 0/2: the wait policy, one switch (H2) and four calibrated values.
// With the switch off these hold the wait the generator has always emitted, so
// `Pause()` folds to the `__nanosleep(64)` it replaces and the default build's
// SASS was identical in the archived SYNC_V2 check. The values
// belong to the target, not to a runner: they come from `TargetSpec`'s `wait_*`
// fields through `tilemega-wait-policy`, and sm_89 and sm_120 do not agree on
// the answer (F-145 measured 910 of sm_89's 1235 ns hop to be this backoff,
// against ~32 ns of sm_120's 448).
#ifndef TILEMEGA_WAIT_POLICY
#define TILEMEGA_WAIT_POLICY 0
#endif
#ifndef TILEMEGA_WAIT_SPIN_ITERS
#define TILEMEGA_WAIT_SPIN_ITERS 0
#endif
#ifndef TILEMEGA_WAIT_BACKOFF_NS
#define TILEMEGA_WAIT_BACKOFF_NS 64
#endif
#ifndef TILEMEGA_WAIT_BACKOFF_GROW
#define TILEMEGA_WAIT_BACKOFF_GROW 1
#endif
#ifndef TILEMEGA_WAIT_BACKOFF_CAP_NS
#define TILEMEGA_WAIT_BACKOFF_CAP_NS 64
#endif

namespace tilemega::codegen {
// Release arrival / acquire wait in the §2.6 xcore1000 protocol (P-7):
// producers converge before the release add; consumers poll with a volatile
// load, fence once the value suffices, and converge afterwards.
__device__ inline unsigned long long LoadAcquire(unsigned long long const* event) {
  return maca::EventLoadAcquire(event);
}
__device__ inline void RedRelease(unsigned long long* event,unsigned long long value) {
  maca::EventAddRelease(event,value);
}
__device__ inline void WaitAtLeast(unsigned long long const* event,
                                   unsigned long long need) {
  while(maca::EventPeek(event)<need) {
#if TILEMEGA_V3_POLL_NS > 0
    maca::Nanosleep(TILEMEGA_V3_POLL_NS);
#endif
  }
  maca::FenceDevice();
}
__device__ inline void WaitAtLeast(unsigned long long const* event,
                                   unsigned long long need,Watch const& watch) {
  unsigned long long start=0,failures=0;
  for(;;) {
    auto value=maca::EventPeek(event);
    if(value>=need)break;
    WatchExpired(&watch,start,++failures,need,value);
#if TILEMEGA_V3_POLL_NS > 0
    maca::Nanosleep(TILEMEGA_V3_POLL_NS);
#endif
  }
  maca::FenceDevice();
}
// T1.1: the following acquire fence remains at the caller.
__device__ inline unsigned long long EventPoll(unsigned long long* event) {
#if TILEMEGA_SYNC_V3 || TILEMEGA_EVENT_LOAD_POLL
  return maca::EventPeek(event);
#else
  return atomicAdd(event, 0ull);
#endif
}

/// One step of the wait policy. It is an object rather than a loop because the
/// two poll loops that share the policy do not share the load: the harness
/// polls an event through `EventPoll`, ClusterSync reads a peer's epoch through
/// a volatile pointer. Every bound is a compile-time constant, so the spin
/// counter and the growth are dead code at the default policy.
///
/// §8.3: this is a performance rule, not a correctness one -- removing the
/// backoff outright left 0/150 mismatches. Spinning does not measurably slow a
/// co-resident computing worker (1.0000 against an issue-bound one, 1.0034
/// against a memory-bound one, inside the idle arm's own 1.0303 spread), so
/// what the sleep buys is not protection of a neighbour.
struct WaitBackoff {
  int spun = 0;
  int nap = TILEMEGA_WAIT_BACKOFF_NS;

  __device__ inline void Pause() {
    if (spun < TILEMEGA_WAIT_SPIN_ITERS) {
      ++spun;
      return;
    }
    if (nap > 0) {
      __nanosleep(nap);
      if (TILEMEGA_WAIT_BACKOFF_GROW > 1) {
        nap *= TILEMEGA_WAIT_BACKOFF_GROW;
        if (nap > TILEMEGA_WAIT_BACKOFF_CAP_NS) nap = TILEMEGA_WAIT_BACKOFF_CAP_NS;
      }
    }
  }
};

/// The policy applied to an event epoch: what `TILEMEGA_GENERATED_WAIT_*`
/// expands to once the switch is on.
__device__ inline void GradedWait(unsigned long long* event,
                                  unsigned long long need) {
#if TILEMEGA_SYNC_V3
  WaitAtLeast(event,need);
#elif TILEMEGA_WAIT_POLICY
  WaitBackoff backoff;
  while (EventPoll(event) < need) backoff.Pause();
#else
  while (EventPoll(event) < need) __nanosleep(64);
#endif
}
}  // namespace tilemega::codegen
