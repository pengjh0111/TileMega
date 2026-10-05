// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#ifndef TILEMEGA_TRACE_TASK
#define TILEMEGA_TRACE_TASK 0
#endif
namespace tilemega::codegen {
struct ServingTaskProfile {
  unsigned long long iteration=0,past=0,run_begin=0,run_end=0,stage=0,task=0,worker=0,
      bytes=0,first_ready=0,query_ns=0,first_page_wait_ns=0,later_page_wait_ns=0,
      wave_compute_ns=0,la_ns=0,epilogue_ns=0;
};
#ifdef __CUDACC__
__device__ inline unsigned long long TaskProfileNow(ServingTaskProfile* p) {
#if TILEMEGA_TRACE_TASK
  if(p){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
#endif
  return 0;
}
#endif
}
