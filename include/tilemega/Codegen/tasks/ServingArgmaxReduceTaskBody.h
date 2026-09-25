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
    for (int part = int(threadIdx.x); part < partial_count;
         part += int(blockDim.x)) {
      int offset = batch * partial_count + part;
      float other = partial_value[offset];
      int other_index = partial_index[offset];
      if (other > best || (other == best && other_index < index)) {
        best = other;
        index = other_index;
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
