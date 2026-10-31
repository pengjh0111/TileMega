// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseOperands.cuh>

#ifndef TILEMEGA_TRACE_STAGE
#define TILEMEGA_TRACE_STAGE 0
#endif

namespace tilemega::backend {

// D0 (prompt §4 A0): timestamps inside a stage, so a serial tail can be
// separated from the stage's streaming part. The boundary publishes the current
// stage trace's mark array in the shared header before Body::Run; a body then
// stamps %globaltimer at the fixed points of appendix D without carrying a
// trace pointer through its signatures.
//
// The slot holds the mark base rather than the record, so this header needs no
// knowledge of StageTrace's layout and stays independent of the program header
// that defines it.
struct Sm90DenseMarks {
  static constexpr int kCount=6;
  __device__ static __forceinline__ unsigned long long* Base() {
    extern __shared__ __align__(1024) char shared[];
    return *reinterpret_cast<unsigned long long**>(shared+Sm90DenseOperandHeader::kTraceSlot);
  }
  __device__ static __forceinline__ unsigned long long Now() {
    unsigned long long time;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(time));return time;
  }
};

// Overwritten by every task that reaches it: the mark reports the last one.
__device__ __forceinline__ void Sm90DenseMark(int index) {
#if TILEMEGA_TRACE_STAGE == 2
  auto* marks=Sm90DenseMarks::Base();
  if(marks)marks[index]=Sm90DenseMarks::Now();
#else
  (void)index;
#endif
}

// Kept from the first task that reaches it; zero means "not reached", which the
// analysis treats as absent (a stage whose geometry has no such task).
__device__ __forceinline__ void Sm90DenseMarkFirst(int index) {
#if TILEMEGA_TRACE_STAGE == 2
  auto* marks=Sm90DenseMarks::Base();
  if(marks && !marks[index])marks[index]=Sm90DenseMarks::Now();
#else
  (void)index;
#endif
}

}  // namespace tilemega::backend
