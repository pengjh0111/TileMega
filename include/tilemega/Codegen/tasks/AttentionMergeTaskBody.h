// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cutlass/bfloat16.h>

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace tilemega::codegen {

template <int HeadDim, int QPerKV, int Tokens>
struct AttentionMergeTaskBody {
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(QPerKV == 2 || QPerKV == 4);
  static constexpr int kThreads = 128;
  struct SharedStorage { alignas(16) unsigned char byte[16]; };

  __device__ static void Run(float const* partial, float const* lse,
                             cutlass::bfloat16_t* context,
                             int batch, int group, int heads_kv,
                             int capacity, int block_extent, int past) {
    int blocks_max = (capacity + block_extent - 1) / block_extent;
    int blocks_live = (past + Tokens + block_extent - 1) / block_extent;
    constexpr int rows = QPerKV * Tokens;
    for (int index = int(threadIdx.x); index < rows * HeadDim;
         index += int(blockDim.x)) {
      int row = index / HeadDim;
      int dim = index % HeadDim;
      float maximum = -INFINITY;
      for (int c = 0; c < blocks_live; ++c) {
        std::size_t base = (((std::size_t(batch) * heads_kv + group) *
                             blocks_max + c) * rows + row);
        maximum = fmaxf(maximum, lse[base]);
      }
      float normalizer = 0.0f;
      float numerator = 0.0f;
      for (int c = 0; c < blocks_live; ++c) {
        std::size_t base = (((std::size_t(batch) * heads_kv + group) *
                             blocks_max + c) * rows + row);
        float weight = exp2f(lse[base] - maximum);
        normalizer += weight;
        numerator += weight * partial[base * HeadDim + dim];
      }
      int token = row / QPerKV;
      int head = row % QPerKV;
      std::size_t output = (std::size_t(batch) * Tokens + token) *
          heads_kv * QPerKV * HeadDim + (group * QPerKV + head) * HeadDim + dim;
      context[output] = cutlass::bfloat16_t(
          normalizer > 0.0f ? numerator / normalizer : 0.0f);
    }
  }
};

}  // namespace tilemega::codegen
