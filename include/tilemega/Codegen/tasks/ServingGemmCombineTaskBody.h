// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/ServingEpilogue.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace tilemega::codegen {

/// The reduction order is the split index order for every output element.
/// The final operation is exactly the same ServingEpilogue used by an unsplit
/// collective, so split-K does not add a BF16 materialization boundary.
template <int TileM, int TileN, backend::ServingEpilogueOp Op>
struct ServingGemmCombineTaskBody {
  static constexpr int kThreads = 128;
  static constexpr int kSharedBytes = 4 * TileM * TileN;

  __device__ static void Run(float const* partial, int split_count,
                             int tile_m, int tile_n, int M, int N,
                             int partial_row_stride, int output_stride,
                             cutlass::bfloat16_t* output,
                             cutlass::bfloat16_t const* residual,
                             float* argmax_value, int* argmax_index,
                             float* shared) {
    if (split_count < 1 || partial_row_stride < N) {
      asm volatile("trap;");
      return;
    }
    std::int64_t split_stride = std::int64_t(M) * partial_row_stride;
    for (int base = int(threadIdx.x) * 4; base < TileM * TileN;
         base += int(blockDim.x) * 4) {
      int row = tile_m * TileM + base / TileN;
      int col = tile_n * TileN + base % TileN;
      float value[4] = {0, 0, 0, 0};
      if (row < M && col + 4 <= N && partial_row_stride % 4 == 0) {
        for (int split = 0; split < split_count; ++split) {
          auto const* src = partial + std::int64_t(split) * split_stride +
                            std::int64_t(row) * partial_row_stride + col;
          float4 incoming = *reinterpret_cast<float4 const*>(src);
          value[0] += incoming.x;
          value[1] += incoming.y;
          value[2] += incoming.z;
          value[3] += incoming.w;
        }
      } else if (row < M) {
        for (int split = 0; split < split_count; ++split)
          for (int lane = 0; lane < 4; ++lane)
            if (col + lane < N)
              value[lane] += partial[std::int64_t(split) * split_stride +
                                     std::int64_t(row) * partial_row_stride +
                                     col + lane];
      }
      *reinterpret_cast<float4*>(shared + backend::ServingEpilogue<Op, TileM, TileN>::SharedIndex(base / TileN, base % TileN)) =
          make_float4(value[0], value[1], value[2], value[3]);
    }
    __syncthreads();
    backend::ServingEpilogue<Op, TileM, TileN>::template RunFromTile<true>(
        shared, tile_m, tile_n, M, N, output_stride, output, residual,
        nullptr, argmax_value, argmax_index);
  }
};

}  // namespace tilemega::codegen
