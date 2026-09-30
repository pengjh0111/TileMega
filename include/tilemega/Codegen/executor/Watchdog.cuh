// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#ifndef TILEMEGA_WATCHDOG
#define TILEMEGA_WATCHDOG 1
#endif
#include <tilemega/Codegen/tasks/ModelRuntime.h>

namespace tilemega::codegen {
// Sites: 1 paged dependency, 2 phase, 3 token lag, 4 KV lag,
// 5 step timestamp, 6 page generation, 7 page full, 8 page empty,
// 9 paged stage barrier, 10 ordinary dependency, 11 grid barrier,
// 12 prefetch stage barrier.
struct Watch {
  WatchdogRecord* record=nullptr;
  unsigned long long limit_ns=0;
  unsigned site=0,waiter_stage=0,waiter_task=0,producer_stage=0,group=0;
  unsigned long long row=0,iteration=0;
  __device__ void Fire(unsigned long long need,unsigned long long value) const {
    if(!record || !limit_ns)return;
    if(atomicCAS(&record->fired,0ull,1ull)!=0ull)return;
    record->site=site;record->block=blockIdx.x;record->thread=threadIdx.x;
    record->waiter_stage=waiter_stage;record->waiter_task=waiter_task;
    record->producer_stage=producer_stage;record->group=group;record->row=row;
    record->need=need;record->value=value;record->iteration=iteration;
    __threadfence_system();asm volatile("trap;");
  }
};
__device__ inline unsigned long long WatchNow() {
  unsigned long long now;
  asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));
  return now;
}
__device__ inline bool WatchExpired(Watch const* watch,unsigned long long& start,
                                    unsigned long long failures,unsigned long long need,
                                    unsigned long long value) {
#if !TILEMEGA_WATCHDOG
  return false;
#else
  if(!watch || !watch->record || !watch->limit_ns || (failures&4095ull))return false;
  auto now=WatchNow();if(!start)start=now;
  if(now-start>watch->limit_ns){watch->Fire(need,value);return true;}
  return false;
#endif
}
} // namespace tilemega::codegen
