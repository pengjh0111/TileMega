// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cstdint>
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
      if(ComputeThread()==0 && bytes)Copy::Prefetch(range.address,bytes);
    }else {
      for(unsigned offset=ComputeThread()*stride;offset<bytes;offset+=kComputeThreads*stride)
        Copy::Prefetch(static_cast<char const*>(range.address)+offset,stride);
    }
  }
};
} // namespace tilemega::codegen::executor
