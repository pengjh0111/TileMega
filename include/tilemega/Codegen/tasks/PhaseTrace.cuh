// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>

#if TILEMEGA_TRACE_PHASE
namespace tilemega::codegen {
#if TILEMEGA_TRACE_KLOOP || TILEMEGA_TRACE_SIMT
__device__ inline unsigned long long PhaseLoopClock(TaskPhase* phase) {
  unsigned long long cycles=0;
  if (phase != nullptr && threadIdx.x == 0)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles) :: "memory");
  return cycles;
}
#endif
__device__ inline void PhaseStamp(TaskPhase* phase, int boundary) {
  if (phase != nullptr && threadIdx.x == 0) {
    unsigned long long time, cycles;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time) :: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles) :: "memory");
    phase->ns[boundary] = time;
    phase->cycles[boundary] = cycles;
  }
}
// SIMT bodies without a separate operand-only prologue expose zero load_wait.
// Their intertwined reads/arithmetic remain in mainloop, not inferred away.
__device__ inline void PhaseSimtSetup(TaskPhase* phase) {
  PhaseStamp(phase, 1);
  if (phase != nullptr && threadIdx.x == 0) {
    phase->ns[2] = phase->ns[1];
    phase->cycles[2] = phase->cycles[1];
  }
}
#if TILEMEGA_TRACE_SIMT
/// Accumulate one already-executed barrier's wait into the slot (B0).  The
/// caller brackets a `__syncthreads()` the body executes anyway; nothing new
/// is issued, so the measured body keeps its original barrier count.
__device__ inline void PhaseSimtAddWait(TaskPhase* phase,
                                        unsigned long long cycles) {
  if (phase != nullptr && threadIdx.x == 0) {
    phase->simt_wait_cycles += cycles;
    phase->simt_barriers += 1;
  }
}
#endif
}
#define TILEMEGA_PHASE_ARG , TaskPhase* phase = nullptr
#define TILEMEGA_PHASE_PASS , phase
#define TILEMEGA_PHASE_STAMP(i) PhaseStamp(phase, i)
#define TILEMEGA_PHASE_SIMT_SETUP() PhaseSimtSetup(phase)
#if TILEMEGA_TRACE_SIMT
// Times a barrier the body already contains.  `__syncthreads()` is issued
// exactly once per expansion, on every thread, as it was before.
#define TILEMEGA_PHASE_SIMT_BARRIER()                                    \
  do {                                                                   \
    unsigned long long tilemega_bar_begin = PhaseLoopClock(phase);       \
    __syncthreads();                                                     \
    PhaseSimtAddWait(phase, PhaseLoopClock(phase) - tilemega_bar_begin); \
  } while (0)
#else
#define TILEMEGA_PHASE_SIMT_BARRIER() __syncthreads()
#endif
#else
#define TILEMEGA_PHASE_ARG
#define TILEMEGA_PHASE_PASS
#define TILEMEGA_PHASE_STAMP(i) ((void)0)
#define TILEMEGA_PHASE_SIMT_SETUP() ((void)0)
#define TILEMEGA_PHASE_SIMT_BARRIER() __syncthreads()
#endif
