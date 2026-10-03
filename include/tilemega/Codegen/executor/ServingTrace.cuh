// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>

namespace tilemega::codegen::executor {
// All fields and instrumentation disappear in an ordinary build. No extra
// synchronization is introduced: completion stamps use existing barriers.
__device__ inline unsigned long long ServingTraceNow() {
  unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;
}

__device__ inline StepTraceRecord* StepRow(Params const& p,unsigned long long iteration) {
#if TILEMEGA_TRACE_STEP
  return p.serving_step_trace && p.serving_trace_launches ?
      p.serving_step_trace+(iteration%p.serving_trace_launches)*gridDim.x+blockIdx.x:nullptr;
#else
  return nullptr;
#endif
}
__device__ inline void StepBegin(Params const& p,unsigned long long iteration) {
#if TILEMEGA_TRACE_STEP
  if(threadIdx.x==0)if(auto* r=StepRow(p,iteration)) {
    r->iteration=iteration;r->past=p.dims.past;if(!r->kernel_begin)r->kernel_begin=ServingTraceNow();
  }
#endif
}
__device__ inline void StepEnd(Params const& p,unsigned long long iteration) {
#if TILEMEGA_TRACE_STEP
  if(threadIdx.x==0)if(auto* r=StepRow(p,iteration))atomicMax(&r->kernel_end,ServingTraceNow());
#endif
}
__device__ inline void TaskBegin(Params const& p,unsigned long long iteration) {
#if TILEMEGA_TRACE_STEP
  if(threadIdx.x==0)if(auto* r=StepRow(p,iteration))
    if(!r->first_task)r->first_task=ServingTraceNow();
#endif
}
__device__ inline void TaskEnd(Params const& p,unsigned long long iteration) {
#if TILEMEGA_TRACE_STEP
  if(threadIdx.x==0)if(auto* r=StepRow(p,iteration))r->last_task=ServingTraceNow();
#endif
}
__device__ inline void StepDelay(Params const& p,unsigned long long iteration,
    unsigned kind,unsigned long long begin) {
#if TILEMEGA_TRACE_STEP
  if(auto* r=StepRow(p,iteration)) {
    auto dt=ServingTraceNow()-begin;
    // Exactly one writer per field: compute leader for token/barrier,
    // loader leader for KV. No atomic operation is needed.
    if(kind==0)r->token_lag+=dt;
    else if(kind==1)r->kv_lag+=dt;
    else r->last_barrier_wait+=dt;
  }
#endif
}
__device__ inline void StageBegin(Params const& p,unsigned stage,
    unsigned long long iteration,unsigned tasks) {
#if TILEMEGA_TRACE_STEP
  if(threadIdx.x==0)if(auto* r=StepRow(p,iteration)) {
    r->stage_has_tasks=tasks;
    if(tasks && !r->first_task)r->first_task=ServingTraceNow();
  }
#endif
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
#if TILEMEGA_TRACE_STEP
  if(p && threadIdx.x==0)if(auto* r=StepRow(*p,iteration))
    if(r->stage_has_tasks)r->last_task=ServingTraceNow();
#endif
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
