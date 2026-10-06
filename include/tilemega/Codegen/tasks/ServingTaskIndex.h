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

#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr ServingAttentionTaskIndex DecodeServingAttentionTaskGMajor(
    int task, int batch, int cache_blocks) {
  ServingAttentionTaskIndex result{};
  result.cache_block=task%cache_blocks;
  task/=cache_blocks;
  result.batch=task%batch;
  result.group=task/batch;
  result.query_block=0;
  return result;
}

// L1 execution ordinal only: logical event/CG numbering remains g-major.
// Small chunks otherwise put inactive capacity tasks between active prefixes,
// concentrating useful work on a subset of grid-stride workers.
#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr int ServingAttentionL1Task(int ordinal,int batch,int groups,
                                     int cache_blocks,int chunk,int seq=1) {
  return seq==1 && chunk<=64
      ? (ordinal%(batch*groups))*cache_blocks+ordinal/(batch*groups)
      : ordinal;
}

// A coprime lane permutation spreads a short final lm_head wave over the
// grid instead of concentrating it on the first CTAs. Invalid padded slots
// are skipped; every real task still executes exactly once.
#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr int ServingSpreadTailTask(int ordinal,int grid) {
  int stride=grid*3/8+1;
  for(;;++stride) {
    int a=stride,b=grid;
    while(b){int r=a%b;a=b;b=r;}
    if(a==1)break;
  }
  return ordinal/grid*grid+(ordinal%grid)*stride%grid;
}

}  // namespace tilemega::codegen
