// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace tilemega::codegen {

struct ServingAttentionTaskIndex {
  int batch;
  int query_block;
  int group;
  int cache_block;
};

// The CG tiles the prefill m axis before the group axis. Its row-major task
// order is (batch, query_block, group, cache_block). Runtime dependencies and
// the task body must decode exactly this order.
#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr ServingAttentionTaskIndex DecodeServingAttentionTask(
    int task, int query_blocks, int groups, int cache_blocks) {
  ServingAttentionTaskIndex result{};
  result.cache_block = task % cache_blocks;
  task /= cache_blocks;
  result.group = task % groups;
  task /= groups;
  result.query_block = task % query_blocks;
  result.batch = task / query_blocks;
  return result;
}

}  // namespace tilemega::codegen
