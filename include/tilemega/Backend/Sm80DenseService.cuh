// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseOperands.cuh>

namespace tilemega::backend {

// Diagnostic only. The two representative lanes publish to separate fields.
struct Sm80DenseService {
  using Tick=unsigned long long;
  Tick producer_wait=0,producer_issue=0,consumer_wait=0,consumer_mma=0,consumer_epilogue=0;

  __device__ static Tick Now() {
    Tick value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory");return value;
  }
#ifdef TILEMEGA_SM80_DENSE_EPILOGUE_CLOCK
  __device__ static Tick Cycles() {
    Tick value;asm volatile("mov.u64 %0, %%clock64;":"=l"(value)::"memory");return value;
  }
#endif
  __device__ static Sm80DenseService* Output() {
    extern __shared__ __align__(1024) char shared[];
    return *reinterpret_cast<Sm80DenseService**>(shared+Sm90DenseOperandHeader::kTraceSlot);
  }
};

}  // namespace tilemega::backend
