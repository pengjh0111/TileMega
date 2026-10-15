// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Target/Maca/DevicePrimitives.h
//
// Every device primitive the executor, harness, epilogues and TaskBodies use
// on MetaX xcore1000 (C550). This replaces the PTX inline assembly of the
// NVIDIA build; nothing outside this header names a __builtin_mxc_* function
// for synchronization, timing or async copies. Choices and evidence are in
// docs/results/MX1_probe.md (MX-P).
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace tilemega::maca {

/// Hardware wave width. TileMega keeps 32-lane logical warps; a wave holds two.
inline constexpr int kWaveSize = 64;
/// Shuffle masks are ignored by the MACA intrinsics (P-5); the named value
/// documents intent at each call site.
inline constexpr unsigned long long kFullWaveMask = ~0ull;

#ifndef TILEMEGA_TRAP_TRACE
#define TILEMEGA_TRAP_TRACE 0
#endif
/// With TILEMEGA_TRAP_TRACE the call site is printed before the trap (bring-up
/// diagnostics; MACA reports only the kernel name).
__device__ __forceinline__ void Trap(char const* file = __builtin_FILE(), int line = __builtin_LINE()) {
#if TILEMEGA_TRAP_TRACE
  printf("TILEMEGA_TRAP %s:%d block=%d thread=%d\n", file, line, int(blockIdx.x), int(threadIdx.x));
#else
  (void)file;
  (void)line;
#endif
  __trap();
}

/// 100 MHz device-wide realtime counter, identical across APs (P-11).
__device__ __forceinline__ unsigned long long GlobalTimerNs() {
  return static_cast<unsigned long long>(__builtin_mxc_get_realtime()) * 10ull;
}

__device__ __forceinline__ long long Clock64() { return clock64(); }

/// No AP identifier is readable (hwreg scan, P-11). Persistent kernels run one
/// CTA per AP, so the block index identifies the execution slot instead.
__device__ __forceinline__ unsigned SlotId() { return blockIdx.x; }

__device__ __forceinline__ void Nanosleep(unsigned ns) {
  if (ns) __nanosleep(ns);
}

/// Device-scope sequentially consistent fence. Also the compiler barrier that
/// keeps loads after an observed flag from reusing earlier values (P-7).
__device__ __forceinline__ void FenceDevice() { __threadfence(); }

// ---- events (§2.6 protocol, P-7) -------------------------------------------
// Publish: data stores -> CTA barrier (caller) -> FenceDevice -> atomic add.
// Wait: volatile poll -> FenceDevice -> CTA barrier (caller).

__device__ __forceinline__ unsigned long long EventPeek(unsigned long long const* event) {
  return *reinterpret_cast<volatile unsigned long long const*>(event);
}
__device__ __forceinline__ unsigned EventPeek(unsigned const* event) {
  return *reinterpret_cast<volatile unsigned const*>(event);
}

/// A single acquiring read: the value, then a fence ordering later reads.
__device__ __forceinline__ unsigned long long EventLoadAcquire(unsigned long long const* event) {
  unsigned long long value = EventPeek(event);
  FenceDevice();
  return value;
}

__device__ __forceinline__ void EventAddRelease(unsigned long long* event, unsigned long long value) {
  FenceDevice();
  atomicAdd(event, value);
}
__device__ __forceinline__ void EventAddRelease(unsigned* event, unsigned value) {
  FenceDevice();
  atomicAdd(event, value);
}

// ---- tickets (grid barriers, last-arriver; §2.6 item 3) ---------------------
// Fences on both sides of the RMW give it acquire-release order at device
// scope; the trailing fence also keeps later loads after the observed ticket.

template <class T>
__device__ __forceinline__ T TicketAddAcqRel(T* ticket, T value) {
  FenceDevice();
  T previous = atomicAdd(ticket, value);
  FenceDevice();
  return previous;
}

template <class T>
__device__ __forceinline__ void TicketStoreRelease(T* ticket, T value) {
  FenceDevice();
  atomicExch(ticket, value);
}

// ---- asynchronous global -> shared copies (P-10) ----------------------------
// One 16-byte load per call; the copy is complete after WaitAsyncCopies<N>()
// leaves at most N of this thread's async copies outstanding (in-order
// retirement; there is no commit/group step).

__device__ __forceinline__ void AsyncCopy16(void* shared_destination, void const* global_source) {
  __builtin_mxc_ldg_b128_bsm(shared_destination, const_cast<void*>(global_source), 0, -1, false, true, false,
                             false);
}

template <int kOutstanding>
__device__ __forceinline__ void WaitAsyncCopies() {
  static_assert(kOutstanding >= 0 && kOutstanding <= 63, "arrive_gvmcnt counts 0..63");
  __builtin_mxc_arrive_gvmcnt(kOutstanding);
}

}  // namespace tilemega::maca
