// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/RuntimeDependencies.h>

namespace tilemega::codegen {
struct RuntimeCountedThresholdView {
  std::uint32_t const* values = nullptr;
  std::uint32_t size = 0;
};
#if defined(__CUDACC__)
#define TILEMEGA_COUNTED_THRESHOLD_HD __host__ __device__
#else
#define TILEMEGA_COUNTED_THRESHOLD_HD
#endif
// A missing table keeps the old uniform threshold. Binding-derived tail
// counts use one immutable entry per consumer, shared by every executor.
TILEMEGA_COUNTED_THRESHOLD_HD inline bool ReadCountedThreshold(
    RuntimeCountedThresholdView table,std::uint32_t offset,std::uint32_t rows,
    std::uint32_t consumer,std::uint32_t uniform,std::uint32_t* count) {
  if(!count)return false;
  if(offset==std::numeric_limits<std::uint32_t>::max()) {
    if(!uniform)return false;
    *count=uniform;return true;
  }
  if(!table.values || !rows || consumer>=rows ||
     std::uint64_t(offset)+rows>table.size || !table.values[offset+consumer])return false;
  *count=table.values[offset+consumer];return true;
}
TILEMEGA_COUNTED_THRESHOLD_HD inline bool CountedThresholdTarget(
    RuntimeCountedThresholdView table,std::uint32_t offset,std::uint32_t rows,
    std::uint32_t consumer,std::uint32_t uniform,std::uint64_t iteration,
    std::uint64_t* target) {
  std::uint32_t count;
  return ReadCountedThreshold(table,offset,rows,consumer,uniform,&count) &&
      CountedDependencyTarget(count,iteration,target);
}
#undef TILEMEGA_COUNTED_THRESHOLD_HD
} // namespace tilemega::codegen
