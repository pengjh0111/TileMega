// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

#ifndef TILEMEGA_TRACE_NATIVE_PHASES
#define TILEMEGA_TRACE_NATIVE_PHASES 0
#endif

#if TILEMEGA_TRACE_NATIVE_PHASES
#define TILEMEGA_NATIVE_PHASES(...) __VA_ARGS__
#define TILEMEGA_NATIVE_PHASE_SCOPE_PARAM , Sm90DenseDiagnosticScope& diagnostic
#define TILEMEGA_NATIVE_PHASE_SCOPE_ARG , diagnostic
#define TILEMEGA_NATIVE_PHASE_TASK_PARAM , int task
#define TILEMEGA_NATIVE_PHASE_TASK_ARG , task
namespace tilemega::backend {

struct Sm90DenseDiagnosticRecord {
  unsigned long long iteration,past,stage,task,worker,smid;
  unsigned long long global_begin,global_end,previous_release_global;
  unsigned long long clock_begin,clock_end,previous_release_clock[4],first_mma_clock[4];
  unsigned long long query_cycles[4],new_row_cycles,wait_cycles,compute_cycles;
  unsigned long long partial_cycles,publish_cycles,merge_cycles;
};
struct Sm90DenseDiagnosticContext {
  Sm90DenseDiagnosticRecord* records;
  unsigned long long iteration,past,stage,previous_release_global,previous_release_clock[4];
};

struct Sm90DenseDiagnostic {
  __device__ static __forceinline__ unsigned long long Clock() {
#if TILEMEGA_TRACE_NATIVE_PHASES
    return clock64();
#else
    return 0;
#endif
  }
  __device__ static __forceinline__ unsigned long long Global() {
    unsigned long long result;
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(result));return result;
  }
  __device__ static Sm90DenseDiagnosticContext& Context(char* shared) {
    // Both persistent ring headers end before byte 512. Operand arenas start
    // at byte 1024; diagnostics never add a barrier or alias live operands.
    static_assert(512+sizeof(Sm90DenseDiagnosticContext)<=1024);
    return *reinterpret_cast<Sm90DenseDiagnosticContext*>(shared+512);
  }
  template<class Ring>
  __device__ static Sm90DenseDiagnosticContext& ContextFromRing(Ring& ring) {
    static_assert(2*sizeof(Ring)<=512);
    auto base=reinterpret_cast<std::uintptr_t>(&ring)&~std::uintptr_t(1023);
    return Context(reinterpret_cast<char*>(base));
  }
  template<class P>
  __device__ static __forceinline__ void Query(P const& p,int task,int warp,
                                               unsigned long long begin) {
#if TILEMEGA_TRACE_NATIVE_PHASES
    if((threadIdx.x&31)==0 && p.diagnostic.records)
      p.diagnostic.records[task].query_cycles[warp]=Clock()-begin;
#endif
  }
  template<class P>
  __device__ static __forceinline__ void NewRow(P const& p,int task,
                                                unsigned long long begin) {
#if TILEMEGA_TRACE_NATIVE_PHASES
    if((threadIdx.x&31)==0 && p.diagnostic.records)
      p.diagnostic.records[task].new_row_cycles=Clock()-begin;
#endif
  }
};

struct Sm90DenseDiagnosticScope {
#if TILEMEGA_TRACE_NATIVE_PHASES
  Sm90DenseDiagnosticRecord* record=nullptr;
  unsigned long long wait=0,compute=0;
  bool recorded_mma=false;
#endif
  template<class P>
  __device__ __forceinline__ void Begin(P const& p,int task) {
#if TILEMEGA_TRACE_NATIVE_PHASES
    record=p.diagnostic.records?p.diagnostic.records+task:nullptr;
    if((threadIdx.x&31)==0 && record) {
      int warp=threadIdx.x/32;
      record->first_mma_clock[warp]=0;
      record->previous_release_clock[warp]=p.diagnostic.previous_release_clock[warp];
    }
    if(threadIdx.x==0 && record) {
      auto const& c=p.diagnostic;unsigned sm;
      asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
      // Other warp leaders own query/new-row fields. Do not zero them here:
      // doing so would race their writes without an extra CTA rendezvous.
      record->iteration=c.iteration;record->past=c.past;record->stage=c.stage;
      record->task=task;record->worker=blockIdx.x;record->smid=sm;
      record->previous_release_global=c.previous_release_global;
      record->global_begin=Sm90DenseDiagnostic::Global();
      record->clock_begin=Sm90DenseDiagnostic::Clock();
      record->partial_cycles=record->publish_cycles=record->merge_cycles=0;
    }
#endif
  }
  __device__ __forceinline__ void FirstMma() {
#if TILEMEGA_TRACE_NATIVE_PHASES
    if((threadIdx.x&31)==0 && record && !recorded_mma) {
      record->first_mma_clock[threadIdx.x/32]=Sm90DenseDiagnostic::Clock();recorded_mma=true;
    }
#endif
  }
  __device__ __forceinline__ void Wait(unsigned long long begin) {
#if TILEMEGA_TRACE_NATIVE_PHASES
    if(threadIdx.x==0)wait+=Sm90DenseDiagnostic::Clock()-begin;
#endif
  }
  __device__ __forceinline__ void Compute(unsigned long long begin) {
#if TILEMEGA_TRACE_NATIVE_PHASES
    if(threadIdx.x==0)compute+=Sm90DenseDiagnostic::Clock()-begin;
#endif
  }
  __device__ __forceinline__ void Partial(unsigned long long begin) {
#if TILEMEGA_TRACE_NATIVE_PHASES
    if(threadIdx.x==0 && record)record->partial_cycles=Sm90DenseDiagnostic::Clock()-begin;
#endif
  }
  __device__ __forceinline__ void Publish(unsigned long long begin) {
#if TILEMEGA_TRACE_NATIVE_PHASES
    if(threadIdx.x==0 && record)record->publish_cycles=Sm90DenseDiagnostic::Clock()-begin;
#endif
  }
  __device__ __forceinline__ void Merge(unsigned long long begin) {
#if TILEMEGA_TRACE_NATIVE_PHASES
    if(threadIdx.x==0 && record)record->merge_cycles=Sm90DenseDiagnostic::Clock()-begin;
#endif
  }
  __device__ __forceinline__ void End() {
#if TILEMEGA_TRACE_NATIVE_PHASES
    if(threadIdx.x==0 && record) {
      record->wait_cycles=wait;record->compute_cycles=compute;
      record->clock_end=Sm90DenseDiagnostic::Clock();
      record->global_end=Sm90DenseDiagnostic::Global();
    }
#endif
  }
};

}  // namespace tilemega::backend
#else
#define TILEMEGA_NATIVE_PHASES(...)
#define TILEMEGA_NATIVE_PHASE_SCOPE_PARAM
#define TILEMEGA_NATIVE_PHASE_SCOPE_ARG
#define TILEMEGA_NATIVE_PHASE_TASK_PARAM
#define TILEMEGA_NATIVE_PHASE_TASK_ARG
#endif
