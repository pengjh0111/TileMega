// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cstdint>
#ifndef TILEMEGA_DIAGNOSTIC_PREFETCH_ADDRESS_ONLY
#define TILEMEGA_DIAGNOSTIC_PREFETCH_ADDRESS_ONLY 0
#endif
#ifndef TILEMEGA_DIAGNOSTIC_PREFETCH_WARP1
#define TILEMEGA_DIAGNOSTIC_PREFETCH_WARP1 0
#endif
namespace tilemega::codegen::executor {
struct PrefetchRange { void const* address; unsigned bytes; };
// Each lane sees the same range list and budget. No dependency event is read.
template<class Arch,bool ForceSm80=false>
struct L2Prefetch {
  using Copy=Async<Arch,ForceSm80>;
  __device__ static void Issue(PrefetchRange range,unsigned& remaining,unsigned stride) {
    unsigned bytes=min(range.bytes,remaining);remaining-=bytes;
    if constexpr(Copy::Caps::kBulkPrefetch) {
      bytes &= ~15u;
      #if TILEMEGA_DIAGNOSTIC_PREFETCH_ADDRESS_ONLY
      asm volatile("" :: "l"(range.address),"r"(bytes));
#else
      if(ComputeThread()==(TILEMEGA_DIAGNOSTIC_PREFETCH_WARP1?32:0) && bytes)
        Copy::Prefetch(range.address,bytes);
#endif
    }else {
      for(unsigned offset=ComputeThread()*stride;offset<bytes;offset+=kComputeThreads*stride)
        Copy::Prefetch(static_cast<char const*>(range.address)+offset,stride);
    }
  }
};
} // namespace tilemega::codegen::executor
