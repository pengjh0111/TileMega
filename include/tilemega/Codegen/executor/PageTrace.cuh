// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#ifndef TILEMEGA_PAGE_TRACE
#define TILEMEGA_PAGE_TRACE 0
#endif
namespace tilemega::codegen::executor {
__device__ inline unsigned long long PageTraceNow() {
  unsigned long long now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
  return now;
}
// The two warp leaders may change state concurrently. The short lock makes
// each interval belong to precisely the state observed before its transition.
__device__ inline void PageTraceTransition(PageTraceRecord* record,
    unsigned bit,bool entering) {
#if TILEMEGA_PAGE_TRACE
  if(!record)return;
  while(atomicCAS(&record->lock,0u,1u)!=0u) {}
  unsigned long long now=PageTraceNow();
  unsigned long long delta=record->last_ns && now>=record->last_ns
      ? now-record->last_ns:0ull;
  if(record->flags&1u)record->dependency_wait_ns+=delta;
  if(record->flags&2u)record->page_full_ns+=delta;
  if(record->flags==3u)record->full_and_wait_ns+=delta;
  if(entering) {
    if(bit==1u)++record->dependency_episodes;
    else ++record->page_full_episodes;
    record->flags|=bit;
  }else record->flags&=~bit;
  record->last_ns=now;
  __threadfence();
  atomicExch(&record->lock,0u);
#else
  (void)record;(void)bit;(void)entering;
#endif
}
} // namespace tilemega::codegen::executor
