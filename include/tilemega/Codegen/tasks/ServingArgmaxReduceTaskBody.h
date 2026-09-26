// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

namespace tilemega::codegen {

/// One request per CTA; the partials are produced by the lm_head epilogue.
/// The tie rule matches torch.argmax: choose the smallest vocabulary index.
struct ServingArgmaxReduceTaskBody {
  static constexpr int kThreads = 128;
  struct SharedStorage {
    float value[4];
    int index[4];
  };
  static constexpr int kSharedBytes = sizeof(SharedStorage);

  __device__ static void RunRow(float const* partial_value,
                                int const* partial_index, int32_t* tokens,
                                int batch, int partial_count, int capacity,
                                int output_position, SharedStorage* shared) {
    float best = -INFINITY;
    int index = INT_MAX;
    // Align by the physical row address, including rows whose partial count
    // is not divisible by four. Only the two edge vectors use scalar loads.
    for (int part = int(threadIdx.x) * 4 - (batch * partial_count % 4);
         part < partial_count; part += int(blockDim.x) * 4) {
      float values[4]; int indices[4];
      if (part >= 0 && part + 4 <= partial_count) {
        int offset = batch * partial_count + part;
        float4 v = *reinterpret_cast<float4 const*>(partial_value + offset);
        int4 i = *reinterpret_cast<int4 const*>(partial_index + offset);
        values[0]=v.x;values[1]=v.y;values[2]=v.z;values[3]=v.w;
        indices[0]=i.x;indices[1]=i.y;indices[2]=i.z;indices[3]=i.w;
      } else {
        #pragma unroll
        for (int lane=0;lane<4;++lane) {
          bool valid=part+lane>=0 && part+lane<partial_count;
          int offset=batch*partial_count+part+lane;
          values[lane]=valid?partial_value[offset]:-INFINITY;
          indices[lane]=valid?partial_index[offset]:INT_MAX;
        }
      }
      #pragma unroll
      for(int lane=0;lane<4;++lane)
        if(values[lane]>best || (values[lane]==best && indices[lane]<index)) {
          best=values[lane];index=indices[lane];
        }
    }
    for (int delta = 16; delta > 0; delta >>= 1) {
      float other = __shfl_down_sync(0xffffffff, best, delta);
      int other_index = __shfl_down_sync(0xffffffff, index, delta);
      if (other > best || (other == best && other_index < index)) {
        best = other;
        index = other_index;
      }
    }
    if ((threadIdx.x & 31) == 0) {
      shared->value[threadIdx.x >> 5] = best;
      shared->index[threadIdx.x >> 5] = index;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      best = shared->value[0];
      index = shared->index[0];
      for (int warp = 1; warp < 4; ++warp) {
        float other = shared->value[warp];
        int other_index = shared->index[warp];
        if (other > best || (other == best && other_index < index)) {
          best = other;
          index = other_index;
        }
      }
      if (output_position >= 0 && output_position < capacity)
        tokens[batch * capacity + output_position] = index;
      else
        asm volatile("trap;");
    }
  }
};

}  // namespace tilemega::codegen
