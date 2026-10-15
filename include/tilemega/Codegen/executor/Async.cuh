// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/ArchDispatch.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <tilemega/Codegen/executor/Watchdog.cuh>

namespace tilemega::codegen::executor {
// The NVIDIA asynchronous transport (cp.async, mbarrier, bulk/TMA, PDL) has no
// xcore1000 counterpart; MetaX async copies use Target/Maca/DevicePrimitives.h.
// Each operation is gated on its capability: a device-pass call of one the
// target lacks is a compile error, never a silent no-op. L2 prefetch is the
// exception the caller opts into (`nonpaged_la` is not a candidate here).
template<class Arch=arch::CurrentArch, bool ForceSm80=false>
struct Async {
  static_assert(!ForceSm80, "xcore1000 has no SM80 transport subset");
  using Caps=arch::Caps<Arch>;
  template<bool kAvailable>
  __device__ static void Require() {
    static_assert(!arch::kDevicePass || kAvailable,
                  "NVIDIA async transport primitive used on a target without it");
  }
  __device__ static void Init(std::uint64_t*,unsigned) { Require<Caps::kMbarrier>(); }
  __device__ static void InitFence() { Require<Caps::kMbarrierTx>(); }
  __device__ static void ProxyAsyncGlobalFence() { Require<Caps::kBulkCopy>(); }
  __device__ static void Arrive(std::uint64_t*) { Require<Caps::kMbarrier>(); }
  __device__ static bool Ready(std::uint64_t*,unsigned) { Require<Caps::kMbarrier>(); return true; }
  __device__ static void Wait(std::uint64_t*,unsigned) { Require<Caps::kMbarrier>(); }
  __device__ static void Wait(std::uint64_t*,unsigned,Watch const*,unsigned,unsigned long long) {
    Require<Caps::kMbarrier>();
  }
  __device__ static void Copy16Bytes(void*,void const*,unsigned) { Require<Caps::kCpAsync>(); }
  __device__ static void Copy8Bytes(void*,void const*,unsigned) { Require<Caps::kCpAsync>(); }
  __device__ static void Copy16(void*,void const*,bool=true) { Require<Caps::kCpAsync>(); }
  __device__ static std::uint64_t EvictFirst() { return 0; }
  __device__ static void Copy16Hint(void*,void const*,std::uint64_t) { Require<Caps::kCpAsync>(); }
  __device__ static void CompleteCopies(std::uint64_t*) { Require<Caps::kCpAsync>(); }
  __device__ static void ExpectTx(std::uint64_t*,unsigned) { Require<Caps::kMbarrierTx>(); }
  __device__ static void Bulk(void*,void const*,unsigned,std::uint64_t*) { Require<Caps::kBulkCopy>(); }
  __device__ static void BulkHint(void*,void const*,unsigned,std::uint64_t*,std::uint64_t) {
    Require<Caps::kBulkCopy>();
  }
  __device__ static void Tensor2D(void*,void const*,int,int,std::uint64_t*) { Require<Caps::kTma>(); }
  // No L2 prefetch instruction exists on xcore1000 (MX-P); these are no-ops.
  __device__ static void Prefetch(void const*,unsigned) {}
  __device__ static void PrefetchEvictLast(void const*,unsigned) {}
  __device__ static void WaitPreviousGrid() { Require<Caps::kPdl>(); }
  __device__ static void LaunchDependents() { Require<Caps::kPdl>(); }
};
} // namespace tilemega::codegen::executor
