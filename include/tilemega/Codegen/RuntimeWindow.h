// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>

namespace tilemega::codegen {

#if defined(__CUDACC__)
#define TILEMEGA_RUNTIME_WINDOW_HD __host__ __device__
#else
#define TILEMEGA_RUNTIME_WINDOW_HD
#endif

struct RuntimeWindowBounds {
  int first = 0;
  int past = 0;
  TILEMEGA_RUNTIME_WINDOW_HD int last() const {
    return first < past ? past - 1 : -1;
  }
};

// The host queue builder and the solver must agree on the producer task a
// runtime window can still require.  The exclusive upper bound is preserved
// because that is the form used by event-group materialization.
TILEMEGA_RUNTIME_WINDOW_HD inline RuntimeWindowBounds RuntimeDependencyBounds(
    int consumer_task, int producer_count, bool all,
    std::int64_t div, std::int64_t scale, std::int64_t offset,
    std::int64_t count) {
  if (producer_count <= 0) return {};
  if (all) return {0, producer_count};
  if (div <= 0 || count <= 0) return {};
  std::int64_t at = (consumer_task / div) * scale + offset;
  std::int64_t begin = at < 0 ? 0 : at;
  std::int64_t end = at + count < producer_count ? at + count : producer_count;
  if (begin >= end) return {};
  return {static_cast<int>(begin), static_cast<int>(end)};
}

#undef TILEMEGA_RUNTIME_WINDOW_HD

}  // namespace tilemega::codegen
