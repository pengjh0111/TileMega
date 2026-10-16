// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/executor/ComputeGroup.cuh>

#include <tilemega/Target/Maca/Bf16.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <tilemega/Backend/ServingVectorIO.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace tilemega::codegen {

using codegen::executor::ComputeThread;
using codegen::executor::ComputeSync;
using codegen::executor::kComputeThreads;

/// One selected row per CTA.  `row_stride` and `row_offset` describe the
/// source row, so the final normalization can select b*S+S-1 without a copy.
struct ServingRMSNormTaskBody {
  static constexpr int kThreads = 128;
  static constexpr int kSharedBytes = 4 * sizeof(float);

  __device__ static tilemega::Bf16 Transform(
      float x, float inverse_rms, float weight) {
    return tilemega::Bf16(
        float(tilemega::Bf16(x * inverse_rms)) * weight);
  }

  __device__ static float RowInvRms(tilemega::Bf16 const* src,
                                    int width, float epsilon, float* shared) {
    float sum = 0.0f;
    for (int base = ComputeThread() * 8; base < width;
         base += kComputeThreads * 8) {
      if (base + 8 <= width &&
          (reinterpret_cast<std::uintptr_t>(src + base) & 15) == 0) {
        alignas(16) tilemega::Bf16 values[8];
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
    if ((ComputeThread() & 31) == 0) shared[ComputeThread() >> 5] = sum;
    ComputeSync();
    if (ComputeThread() < 4) {
      float value = shared[ComputeThread()];
      for (int delta = 2; delta > 0; delta >>= 1)
        value += __shfl_down_sync(0xf, value, delta, 4);
      if (ComputeThread() == 0) shared[0] = maca::Rsqrt(value / width + epsilon);
    }
    ComputeSync();
    return shared[0];
  }

  __device__ static void RunRow(tilemega::Bf16 const* input,
                                tilemega::Bf16 const* weight,
                                tilemega::Bf16* output,
                                int row, int row_stride, int row_offset,
                                int width, float epsilon, float* shared) {
    int source_row = row * row_stride + row_offset;
    auto const* src = input + source_row * width;
    auto* dst = output + row * width;
    float scale = RowInvRms(src, width, epsilon, shared);
    for (int base = ComputeThread() * 8; base < width;
         base += kComputeThreads * 8) {
      alignas(16) tilemega::Bf16 values[8];
      alignas(16) tilemega::Bf16 source[8];
      alignas(16) tilemega::Bf16 scales[8];
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
        values[j] = Transform(x, scale, w);
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
