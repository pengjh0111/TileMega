// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cutlass/bfloat16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace tilemega::codegen {

/// A token row is owned by one CTA.  The token state is persistent across
/// launches; each launch reads only the position selected by its Params.
struct ServingEmbeddingTaskBody {
  static constexpr int kThreads = 128;
  static constexpr int kSharedBytes = 0;

  __device__ static void RunRow(int32_t const* tokens,
                                cutlass::bfloat16_t const* table,
                                cutlass::bfloat16_t* output,
                                int token_row, int seq, int past,
                                int capacity, int width, int vocab) {
    int batch = token_row / seq;
    int position = past + token_row % seq;
    if (position < 0 || position >= capacity) {
      asm volatile("trap;");
      return;
    }
    int id = tokens[batch * capacity + position];
    if (id < 0 || id >= vocab) {
      asm volatile("trap;");
      return;
    }
    auto const* source = table + std::int64_t(id) * width;
    auto* target = output + std::int64_t(token_row) * width;
    for (int column = int(threadIdx.x) * 8; column < width;
         column += int(blockDim.x) * 8) {
      int tail = width - column;
      if (tail >= 8 &&
          (reinterpret_cast<std::uintptr_t>(source + column) & 15) == 0 &&
          (reinterpret_cast<std::uintptr_t>(target + column) & 15) == 0) {
        *reinterpret_cast<uint4*>(target + column) =
            *reinterpret_cast<uint4 const*>(source + column);
      } else {
        int count = tail < 8 ? tail : 8;
        for (int lane = 0; lane < count; ++lane)
          target[column + lane] = source[column + lane];
      }
    }
  }
};

}  // namespace tilemega::codegen
