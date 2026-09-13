// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda/atomic>
#include <cuda_runtime.h>

#ifndef TILEMEGA_EVENT_LOAD_POLL
#define TILEMEGA_EVENT_LOAD_POLL 0
#endif

// EX-E3 step 0/2: the wait policy, one switch (H2) and four calibrated values.
// With the switch off these hold the wait the generator has always emitted, so
// `Pause()` folds to the `__nanosleep(64)` it replaces and the default build's
// SASS does not move (docs/experiments/SYNC_V2/sass_identity/).  The values
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
// T1.1: the following acquire fence remains at the caller. A relaxed atomic
// read participates in the fence synchronization without an RMW transaction.
__device__ inline unsigned long long EventPoll(unsigned long long* event) {
#if TILEMEGA_EVENT_LOAD_POLL
  return cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(*event)
      .load(cuda::memory_order_relaxed);
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
  WaitBackoff backoff;
  while (EventPoll(event) < need) backoff.Pause();
}
}  // namespace tilemega::codegen
