// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#ifndef TILEMEGA_TRACE_TASK
#define TILEMEGA_TRACE_TASK 0
#endif
namespace tilemega::codegen {
struct ServingTaskProfile {
  unsigned long long iteration=0,past=0,run_begin=0,run_end=0,stage=0,task=0,worker=0,
      bytes=0,first_ready=0,query_ns=0,first_page_wait_ns=0,later_page_wait_ns=0,
      wave_compute_ns=0,la_ns=0,epilogue_ns=0,
      // Task start to the body's first instruction: the descriptor chain
      // (Params -> stage -> invocation -> buffers) a combine walks first.
      decode_ns=0;
};
#ifdef __CUDACC__
__device__ inline unsigned long long TaskProfileNow(ServingTaskProfile* p) {
#if TILEMEGA_TRACE_TASK
  if(p)return static_cast<unsigned long long>(__builtin_mxc_get_realtime())*10ull;
#endif
  return 0;
}
#if TILEMEGA_TRACE_TASK
// A body several calls below RunStage (the split-K combine) stamps its own
// phases through this per-CTA slot rather than a new argument on every call
// in between; RunStage sets it around the task. Trace builds only.
static __device__ ServingTaskProfile* g_serving_profile_slot[4096];
__device__ inline ServingTaskProfile* ProfileSlot() {
  return threadIdx.x == 0 ? g_serving_profile_slot[blockIdx.x] : nullptr;
}
#endif
struct FirstTileReadyProfile {
  ServingTaskProfile* profile;
  __device__ void operator()() const {
#if TILEMEGA_TRACE_TASK
    if(profile && (threadIdx.x & 127)==0)profile->first_ready=TaskProfileNow(profile);
#endif
  }
};
#endif
}
