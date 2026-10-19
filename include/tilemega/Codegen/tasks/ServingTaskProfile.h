// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#ifndef TILEMEGA_TRACE_TASK
#define TILEMEGA_TRACE_TASK 0
#endif
#ifndef TILEMEGA_TRACE_LOCAL_CLOCK
#define TILEMEGA_TRACE_LOCAL_CLOCK 0
#endif
namespace tilemega::codegen {
struct ServingTaskProfile {
  unsigned long long iteration=0,past=0,run_begin=0,run_end=0,stage=0,task=0,worker=0,
      bytes=0,first_ready=0,query_ns=0,first_page_wait_ns=0,later_page_wait_ns=0,
      wave_compute_ns=0,la_ns=0,epilogue_ns=0;
#if TILEMEGA_TRACE_LOCAL_CLOCK
  unsigned long long local_begin=0,local_end=0,la_total=0,new_row=0,partial_store=0;
  unsigned long long new_row_warp0=0,new_row_warp1=0,new_row_warp2=0,new_row_warp3=0;
#endif
};
#ifdef __CUDACC__
__device__ inline unsigned long long TaskProfileNow(ServingTaskProfile* p) {
#if TILEMEGA_TRACE_TASK
  if(p){
#if TILEMEGA_TRACE_LOCAL_CLOCK
    return clock64();
#else
    unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;
#endif
  }
#endif
  return 0;
}
struct FirstTileReadyProfile {
  ServingTaskProfile* profile;
  __device__ void operator()() const {
#if TILEMEGA_TRACE_TASK
    if(profile && (threadIdx.x & 127)==0)profile->first_ready=TaskProfileNow(profile);
#endif
  }
};
struct DmEpilogueProfile {
  ServingTaskProfile* profile;
  unsigned long long begin;
  __device__ explicit DmEpilogueProfile(ServingTaskProfile* p)
      :profile(p),begin(TaskProfileNow(p)) {}
  __device__ ~DmEpilogueProfile() {
#if TILEMEGA_TRACE_TASK
    if(profile)profile->epilogue_ns+=TaskProfileNow(profile)-begin;
#endif
  }
};
#endif
}
