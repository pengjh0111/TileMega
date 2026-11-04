// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#ifndef TILEMEGA_TRACE_STAGE
#define TILEMEGA_TRACE_STAGE 0
#endif

#if TILEMEGA_TRACE_STAGE
#include <cstdint>
#ifndef TILEMEGA_FORWARD_TRACE_STAGES
#error "forward tracing needs the dimensions emitted by ForwardCodegen"
#endif
#ifndef TILEMEGA_FORWARD_TRACE_GRID
#error "forward tracing needs the grid emitted by ForwardCodegen"
#endif

namespace tilemega::backend {

static_assert(TILEMEGA_FORWARD_TRACE_STAGES>0 && TILEMEGA_FORWARD_TRACE_GRID>0);

enum class Sm90ForwardTraceSegment : unsigned {
  kSlotWait,kMainLoop,kPartialWrite,kSplitKReduce,kEpilogue,
  kHaloWait,kDepthwiseCompute,kFusedDepthwise,kFusedPointwise,
  kKvWait,kAttentionQk,kAttentionSoftmax,kAttentionPv,
  kSmallLoad,kSmallCompute,kSmallStore,kCount
};
inline constexpr unsigned kSm90ForwardTraceSegments=
    unsigned(Sm90ForwardTraceSegment::kCount);
inline constexpr char const* kSm90ForwardTraceNames[]={
    "slot_wait","main_loop","partial_write","splitk_reduce","epilogue",
    "halo_wait","depthwise_compute","fused_depthwise","fused_pointwise",
    "kv_wait","attention_qk","attention_softmax","attention_pv",
    "small_load","small_compute","small_store"};
static_assert(sizeof(kSm90ForwardTraceNames)/sizeof(*kSm90ForwardTraceNames)==
    kSm90ForwardTraceSegments);

// All fields are uint64 nanoseconds from %globaltimer. Each CTA has one writer
// (consumer thread zero); body spans accumulate in registers across tasks and
// are flushed once. Nested wait spans are diagnostics, not additive phases.
struct Sm90ForwardTraceSpan {
  unsigned long long total_ns,count,first_begin,last_end;
};
struct Sm90ForwardTraceRecord {
  unsigned long long enter,arrive,leave,first_ready;
  Sm90ForwardTraceSpan segments[kSm90ForwardTraceSegments];
};
static_assert(sizeof(Sm90ForwardTraceRecord)==8*(4+4*kSm90ForwardTraceSegments));
static __device__ Sm90ForwardTraceRecord sm90_forward_trace_records[
    TILEMEGA_FORWARD_TRACE_STAGES*TILEMEGA_FORWARD_TRACE_GRID];
static __device__ unsigned sm90_forward_trace_stage[TILEMEGA_FORWARD_TRACE_GRID];

__device__ __forceinline__ unsigned long long Sm90ForwardTraceNow() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory");
  return value;
}
__device__ __forceinline__ Sm90ForwardTraceRecord& Sm90ForwardTraceCurrent() {
  return sm90_forward_trace_records[
      sm90_forward_trace_stage[blockIdx.x]*TILEMEGA_FORWARD_TRACE_GRID+blockIdx.x];
}
__device__ __forceinline__ void Sm90ForwardTraceEnter(unsigned stage) {
  if(threadIdx.x==0) {
    sm90_forward_trace_stage[blockIdx.x]=stage;
    Sm90ForwardTraceCurrent().enter=Sm90ForwardTraceNow();
  }
}
__device__ __forceinline__ void Sm90ForwardTraceArrive() {
  if(threadIdx.x==0)Sm90ForwardTraceCurrent().arrive=Sm90ForwardTraceNow();
}
__device__ __forceinline__ void Sm90ForwardTraceLeave() {
  if(threadIdx.x==0)Sm90ForwardTraceCurrent().leave=Sm90ForwardTraceNow();
}
__device__ __forceinline__ void Sm90ForwardTraceFirstReady() {
  if(threadIdx.x==0)Sm90ForwardTraceCurrent().first_ready=Sm90ForwardTraceNow();
}

struct Sm90ForwardTraceAccumulator {
  unsigned long long begin=0,total=0,count=0,first=0,last=0;
  __device__ __forceinline__ void Begin() {
    if(threadIdx.x==0)begin=Sm90ForwardTraceNow();
  }
  __device__ __forceinline__ void End() {
    if(threadIdx.x==0) {
      auto end=Sm90ForwardTraceNow();
      if(!count)first=begin;
      total+=end-begin;last=end;++count;
    }
  }
  __device__ __forceinline__ void Point() {
    if(threadIdx.x==0) {
      auto now=Sm90ForwardTraceNow();
      if(!count)first=now;
      last=now;++count;
    }
  }
  __device__ __forceinline__ void Flush(Sm90ForwardTraceSegment segment) const {
    if(threadIdx.x==0)
      Sm90ForwardTraceCurrent().segments[unsigned(segment)]={total,count,first,last};
  }
};

}  // namespace tilemega::backend
#define TM_FWD_TRACE_ACC(name) ::tilemega::backend::Sm90ForwardTraceAccumulator name
#define TM_FWD_TRACE_BEGIN(name) name.Begin()
#define TM_FWD_TRACE_END(name) name.End()
#define TM_FWD_TRACE_POINT(name) name.Point()
#define TM_FWD_TRACE_FLUSH(name,kind) name.Flush(::tilemega::backend::Sm90ForwardTraceSegment::kind)
#define TM_FWD_TRACE_FIRST_READY() ::tilemega::backend::Sm90ForwardTraceFirstReady()
#else
#define TM_FWD_TRACE_ACC(name)
#define TM_FWD_TRACE_BEGIN(name)
#define TM_FWD_TRACE_END(name)
#define TM_FWD_TRACE_POINT(name)
#define TM_FWD_TRACE_FLUSH(name,kind)
#define TM_FWD_TRACE_FIRST_READY()
#endif
