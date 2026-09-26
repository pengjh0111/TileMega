// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cutlass/bfloat16.h>
#include <tilemega/Backend/ServingVectorIO.h>

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
    // Each thread owns eight adjacent BF16 outputs. Both FP32 inputs are
    // aligned 16-byte vectors because every row has HeadDim % 8 == 0.
    for (int vector = int(threadIdx.x); vector < rows * HeadDim / 8;
         vector += int(blockDim.x)) {
      int row = vector / (HeadDim / 8);
      int dim = (vector % (HeadDim / 8)) * 8;
      float maximum = -INFINITY;
      for (int c = 0; c < blocks_live; ++c) {
        std::size_t base = (((std::size_t(batch) * heads_kv + group) *
                             blocks_max + c) * rows + row);
        maximum = fmaxf(maximum, lse[base]);
      }
      float normalizer = 0.0f;
      float numerator[8] = {};
      for (int c = 0; c < blocks_live; ++c) {
        std::size_t base = (((std::size_t(batch) * heads_kv + group) *
                             blocks_max + c) * rows + row);
        float weight = exp2f(lse[base] - maximum);
        normalizer += weight;
        float4 first = *reinterpret_cast<float4 const*>(partial + base * HeadDim + dim);
        float4 second = *reinterpret_cast<float4 const*>(partial + base * HeadDim + dim + 4);
        float values[8] = {first.x, first.y, first.z, first.w,
                           second.x, second.y, second.z, second.w};
        #pragma unroll
        for (int lane = 0; lane < 8; ++lane)
          numerator[lane] += weight * values[lane];
      }
      int token = row / QPerKV;
      int head = row % QPerKV;
      std::size_t output = (std::size_t(batch) * Tokens + token) *
          heads_kv * QPerKV * HeadDim + (group * QPerKV + head) * HeadDim + dim;
      alignas(16) cutlass::bfloat16_t result[8];
      #pragma unroll
      for (int lane = 0; lane < 8; ++lane)
        result[lane] = cutlass::bfloat16_t(
            normalizer > 0.0f ? numerator[lane] / normalizer : 0.0f);
      backend::StoreGlobal16(context + output,*reinterpret_cast<uint4 const*>(result));
    }
  }
};

}  // namespace tilemega::codegen
