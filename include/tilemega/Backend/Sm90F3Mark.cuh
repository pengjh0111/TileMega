// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#ifndef TILEMEGA_TRACE_STAGE
#define TILEMEGA_TRACE_STAGE 0
#endif

#if TILEMEGA_TRACE_STAGE
#include <cstdint>
#include <tilemega/Backend/Sm90F3Async.cuh>
#ifndef TILEMEGA_FORWARD_TRACE_STAGES
#error "forward tracing needs the dimensions emitted by ForwardCodegen"
#endif
#ifndef TILEMEGA_FORWARD_TRACE_GRID
#error "forward tracing needs the grid emitted by ForwardCodegen"
#endif

namespace tilemega::backend {

static_assert(TILEMEGA_FORWARD_TRACE_STAGES>0 && TILEMEGA_FORWARD_TRACE_GRID>0);

enum class Sm90F3TraceSegment : unsigned {
  kSlotWait,kMainLoop,kSideWait,kEpilogue,kStore,kSplitK,
  kAttentionQk,kAttentionSoftmax,kAttentionPv,kSmallCompute,kCount
};
inline constexpr unsigned kSm90F3TraceSegments=unsigned(Sm90F3TraceSegment::kCount);
inline constexpr char const* kSm90F3TraceNames[]={
  "slot_wait","main_loop","side_wait","epilogue","store","splitk",
  "attention_qk","attention_softmax","attention_pv","small_compute"};
static_assert(sizeof(kSm90F3TraceNames)/sizeof(*kSm90F3TraceNames)==kSm90F3TraceSegments);

// All fields are uint64 nanoseconds from %globaltimer. Each CTA has one writer
// (consumer thread zero); body spans update the global record per task.
// Nested wait spans are diagnostics, not additive phases.
struct Sm90F3TraceSpan {
  unsigned long long total_ns,count,first_begin,last_end;
};
struct Sm90F3TraceRecord {
  unsigned long long enter,arrive,leave,first_ready;
  Sm90F3TraceSpan segments[kSm90F3TraceSegments];
};
static_assert(sizeof(Sm90F3TraceRecord)==8*(4+4*kSm90F3TraceSegments));
static __device__ Sm90F3TraceRecord sm90_f3_trace_records[
    TILEMEGA_FORWARD_TRACE_STAGES*TILEMEGA_FORWARD_TRACE_GRID];
static __device__ unsigned sm90_f3_trace_stage[TILEMEGA_FORWARD_TRACE_GRID];

__device__ __forceinline__ unsigned long long Sm90F3TraceNow() {
  return Sm90F3Async::GlobalTimer();
}
__device__ __forceinline__ Sm90F3TraceRecord& Sm90F3TraceCurrent() {
  return sm90_f3_trace_records[
      sm90_f3_trace_stage[blockIdx.x]*TILEMEGA_FORWARD_TRACE_GRID+blockIdx.x];
}
__device__ __forceinline__ void Sm90F3TraceEnter(unsigned stage) {
  if(threadIdx.x==0) {
    sm90_f3_trace_stage[blockIdx.x]=stage;
    Sm90F3TraceCurrent()={};
    Sm90F3TraceCurrent().enter=Sm90F3TraceNow();
  }
}
__device__ __forceinline__ void Sm90F3TraceArrive() {
  if(threadIdx.x==0)Sm90F3TraceCurrent().arrive=Sm90F3TraceNow();
}
__device__ __forceinline__ void Sm90F3TraceLeave() {
  if(threadIdx.x==0)Sm90F3TraceCurrent().leave=Sm90F3TraceNow();
}
__device__ __forceinline__ void Sm90F3TraceFirstReady() {
  if(threadIdx.x==0 && !Sm90F3TraceCurrent().first_ready)Sm90F3TraceCurrent().first_ready=Sm90F3TraceNow();
}

// Keep attribution counters in their CTA record, outside the compute register set.
__device__ __forceinline__ void Sm90F3TraceBegin(Sm90F3TraceSegment kind) {
  if(threadIdx.x==0) {
    auto& span=Sm90F3TraceCurrent().segments[unsigned(kind)];
    auto now=Sm90F3TraceNow();
    if(!span.count)span.first_begin=now;
    span.last_end=now;
  }
}
__device__ __forceinline__ void Sm90F3TraceEnd(Sm90F3TraceSegment kind) {
  if(threadIdx.x==0) {
    auto now=Sm90F3TraceNow();
    auto& span=Sm90F3TraceCurrent().segments[unsigned(kind)];
    span.total_ns+=now-span.last_end;span.last_end=now;++span.count;
  }
}
}  // namespace tilemega::backend
#define TM_F3_TRACE_ENTER(i) ::tilemega::backend::Sm90F3TraceEnter(i)
#define TM_F3_TRACE_ARRIVE() ::tilemega::backend::Sm90F3TraceArrive()
#define TM_F3_TRACE_LEAVE() ::tilemega::backend::Sm90F3TraceLeave()
#define TM_F3_TRACE_FIRST_READY() ::tilemega::backend::Sm90F3TraceFirstReady()
#define TM_F3_TRACE_BEGIN(kind) ::tilemega::backend::Sm90F3TraceBegin(::tilemega::backend::Sm90F3TraceSegment::kind)
#define TM_F3_TRACE_END(kind) ::tilemega::backend::Sm90F3TraceEnd(::tilemega::backend::Sm90F3TraceSegment::kind)
#else
#define TM_F3_TRACE_ENTER(i)
#define TM_F3_TRACE_ARRIVE()
#define TM_F3_TRACE_LEAVE()
#define TM_F3_TRACE_FIRST_READY()
#define TM_F3_TRACE_BEGIN(kind)
#define TM_F3_TRACE_END(kind)
#endif
