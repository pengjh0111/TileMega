// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>

namespace tilemega::codegen::executor {
// All fields and instrumentation disappear in an ordinary build. No extra
// synchronization is introduced: completion stamps use existing barriers.
__device__ inline unsigned long long ServingTraceNow() {
  unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;
}
__device__ inline void StageBegin(Params const& p,unsigned stage,
    unsigned long long iteration,unsigned tasks) {
#if TILEMEGA_TRACE_STAGE
  if(threadIdx.x==0 && p.serving_stage_trace && p.serving_trace_launches) {
    auto& r=p.serving_stage_trace[(iteration%p.serving_trace_launches)*p.stage_count*gridDim.x+
                                 stage*gridDim.x+blockIdx.x];
    unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
    r={iteration,static_cast<unsigned long long>(p.dims.past),ServingTraceNow(),0,0,tasks,sm,stage};
  }
#endif
}
__device__ inline void StageTasksEnd(Params const* p,unsigned stage,
    unsigned long long iteration) {
#if TILEMEGA_TRACE_STAGE
  if(p && threadIdx.x==0 && p->serving_stage_trace && p->serving_trace_launches)
    p->serving_stage_trace[(iteration%p->serving_trace_launches)*p->stage_count*gridDim.x+
                          stage*gridDim.x+blockIdx.x].t_tasks_end=ServingTraceNow();
#endif
}
__device__ inline void StageRelease(Params const& p,unsigned stage,
    unsigned long long iteration) {
#if TILEMEGA_TRACE_STAGE
  if(threadIdx.x==0 && p.serving_stage_trace && p.serving_trace_launches)
    p.serving_stage_trace[(iteration%p.serving_trace_launches)*p.stage_count*gridDim.x+
                         stage*gridDim.x+blockIdx.x].t_release=ServingTraceNow();
#endif
}
} // namespace tilemega::codegen::executor
