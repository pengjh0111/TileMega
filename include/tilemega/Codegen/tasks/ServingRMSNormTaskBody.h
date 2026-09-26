// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cutlass/bfloat16.h>
#include <tilemega/Backend/ServingVectorIO.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace tilemega::codegen {

/// One selected row per CTA.  `row_stride` and `row_offset` describe the
/// source row, so the final normalization can select b*S+S-1 without a copy.
struct ServingRMSNormTaskBody {
  static constexpr int kThreads = 128;
  static constexpr int kSharedBytes = 4 * sizeof(float);

  __device__ static void RunRow(cutlass::bfloat16_t const* input,
                                 cutlass::bfloat16_t const* weight,
                                 cutlass::bfloat16_t* output,
                                 int row, int row_stride, int row_offset,
                                 int width, float epsilon, float* shared) {
    int source_row = row * row_stride + row_offset;
    auto const* src = input + source_row * width;
    auto* dst = output + row * width;
    float sum = 0.0f;
    for (int base = int(threadIdx.x) * 8; base < width;
         base += int(blockDim.x) * 8) {
      if (base + 8 <= width &&
          (reinterpret_cast<std::uintptr_t>(src + base) & 15) == 0) {
        alignas(16) cutlass::bfloat16_t values[8];
        *reinterpret_cast<uint4*>(values) =
            backend::LoadGlobal16(src + base);
        for (int j = 0; j < 8; ++j) {
          float x = float(values[j]);
          sum += x * x;
        }
      } else {
        int limit = base + 8 < width ? base + 8 : width;
        for (int j = base; j < limit; ++j) {
          float x = float(src[j]);
          sum += x * x;
        }
      }
    }
    for (int delta = 16; delta > 0; delta >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, delta);
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 4) {
      float value = shared[threadIdx.x];
      for (int delta = 2; delta > 0; delta >>= 1)
        value += __shfl_down_sync(0xf, value, delta, 4);
      if (threadIdx.x == 0) shared[0] = rsqrtf(value / width + epsilon);
    }
    __syncthreads();
    float scale = shared[0];
    for (int base = int(threadIdx.x) * 8; base < width;
         base += int(blockDim.x) * 8) {
      alignas(16) cutlass::bfloat16_t values[8];
      alignas(16) cutlass::bfloat16_t source[8];
      alignas(16) cutlass::bfloat16_t scales[8];
      int count = width - base < 8 ? width - base : 8;
      bool vector_input = count == 8 &&
                          (reinterpret_cast<std::uintptr_t>(src + base) & 15) == 0 &&
                          (reinterpret_cast<std::uintptr_t>(weight + base) & 15) == 0;
      if (vector_input) {
        *reinterpret_cast<uint4*>(source) =
            backend::LoadGlobal16(src + base);
        *reinterpret_cast<uint4*>(scales) =
            backend::LoadGlobal16(weight + base);
      }
      for (int j = 0; j < count; ++j) {
        float x = float(vector_input ? source[j] : src[base + j]);
        float w = float(vector_input ? scales[j] : weight[base + j]);
        auto normalized = cutlass::bfloat16_t(x * scale);
        values[j] = cutlass::bfloat16_t(
            float(normalized) * w);
      }
      if (count == 8 &&
          (reinterpret_cast<std::uintptr_t>(dst + base) & 15) == 0)
        backend::StoreGlobal16(dst + base,*reinterpret_cast<uint4 const*>(values));
      else
        for (int j = 0; j < count; ++j) dst[base + j] = values[j];
    }
  }
};

}  // namespace tilemega::codegen
