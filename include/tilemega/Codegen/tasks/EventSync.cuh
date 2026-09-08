// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda/atomic>
#include <cuda_runtime.h>

#ifndef TILEMEGA_EVENT_LOAD_POLL
#define TILEMEGA_EVENT_LOAD_POLL 0
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
}  // namespace tilemega::codegen
