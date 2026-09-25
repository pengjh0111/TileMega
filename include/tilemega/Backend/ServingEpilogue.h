// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

namespace tilemega::backend {

enum class ServingEpilogueOp {
  kStore, kResidual, kSwiGLU, kArgmaxPartial, kPartial
};

/// The same scalar operation is called by an unsplit GEMM and by its combine
/// task.  Only the input accumulation differs; the BF16 materialization point
/// is therefore independent of split-K.
template <ServingEpilogueOp Op>
struct ServingEpilogueValue {
  __device__ static cutlass::bfloat16_t Apply(float acc,
                                               cutlass::bfloat16_t residual = {}) {
    if constexpr (Op == ServingEpilogueOp::kResidual) {
      auto rounded = cutlass::bfloat16_t(acc);
      return cutlass::bfloat16_t(float(rounded) + float(residual));
    } else {
      return cutlass::bfloat16_t(acc);
    }
  }
};

template <>
struct ServingEpilogueValue<ServingEpilogueOp::kSwiGLU> {
  __device__ static cutlass::bfloat16_t Apply(float gate, float up) {
    float g = float(cutlass::bfloat16_t(gate));
    float v = float(cutlass::bfloat16_t(up));
    auto activated = cutlass::bfloat16_t(g / (1.0f + expf(-g)));
    return cutlass::bfloat16_t(float(activated) * v);
  }
};

template <ServingEpilogueOp Op, int TileM, int TileN, int U = 16>
struct ServingEpilogue {
  static_assert(TileN % (2 * U) == 0 || Op != ServingEpilogueOp::kSwiGLU,
                "SwiGLU interleave must end on a gate/up pair");

  template <class Accumulator, class TiledMma>
  __device__ static void Run(Accumulator const& accum, TiledMma const& mma,
                             char* shared, int tile_m, int tile_n,
                             int M, int N, int output_stride,
                             cutlass::bfloat16_t* output,
                             cutlass::bfloat16_t const* residual = nullptr,
                             float* partial = nullptr,
                             float* argmax_value = nullptr,
                             int* argmax_index = nullptr) {
    // The cp.async mainloop has vacated this storage.  The 4-byte accumulator
    // tile fits in the same union as the (possibly larger) staged operands.
    cute::cp_async_wait<0>();
    __syncthreads();
    float* tile = reinterpret_cast<float*>(shared);
    auto coordinates = cute::make_identity_tensor(
        cute::Shape<cute::Int<TileM>, cute::Int<TileN>>{});
    auto owned = mma.get_thread_slice(int(threadIdx.x)).partition_C(coordinates);
    CUTE_STATIC_ASSERT_V(cute::size(owned) == cute::size(accum));
    for (int i = 0; i < cute::size(accum); ++i) {
      int row = cute::get<0>(owned(i));
      int column = cute::get<1>(owned(i));
      tile[row * TileN + column] = accum(i);
    }
    __syncthreads();
    RunFromTile(tile, tile_m, tile_n, M, N, output_stride, output,
                residual, partial, argmax_value, argmax_index);
  }

  __device__ static void RunFromTile(
      float* tile, int tile_m, int tile_n, int M, int N, int output_stride,
      cutlass::bfloat16_t* output,
      cutlass::bfloat16_t const* residual = nullptr,
      float* partial = nullptr, float* argmax_value = nullptr,
      int* argmax_index = nullptr) {

    if constexpr (Op == ServingEpilogueOp::kArgmaxPartial) {
      // Descending rows leave the last row free as four-warp reduction scratch.
      // This avoids adding storage beyond ServingBF16SmemBytes.
      float* scratch_values = tile + (TileM - 1) * TileN;
      int* scratch_indices = reinterpret_cast<int*>(scratch_values + 4);
      for (int row = TileM - 1; row >= 0; --row) {
        int global_row = tile_m * TileM + row;
        if (global_row >= M) continue;
        float best = -INFINITY;
        int best_index = INT32_MAX;
        for (int column = int(threadIdx.x); column < TileN; column += 128) {
          int global_column = tile_n * TileN + column;
          if (global_column >= N) continue;
          float value = float(cutlass::bfloat16_t(tile[row * TileN + column]));
          if (value > best || (value == best && global_column < best_index)) {
            best = value;
            best_index = global_column;
          }
        }
        for (int offset = 16; offset != 0; offset >>= 1) {
          float other = __shfl_down_sync(0xffffffff, best, offset);
          int other_index = __shfl_down_sync(0xffffffff, best_index, offset);
          if (other > best || (other == best && other_index < best_index)) {
            best = other;
            best_index = other_index;
          }
        }
        __syncthreads();
        if ((threadIdx.x & 31) == 0) {
          scratch_values[threadIdx.x >> 5] = best;
          scratch_indices[threadIdx.x >> 5] = best_index;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          best = scratch_values[0];
          best_index = scratch_indices[0];
          for (int warp = 1; warp < 4; ++warp) {
            float other = scratch_values[warp];
            int other_index = scratch_indices[warp];
            if (other > best || (other == best && other_index < best_index)) {
              best = other;
              best_index = other_index;
            }
          }
          argmax_value[global_row * output_stride + tile_n] = best;
          argmax_index[global_row * output_stride + tile_n] = best_index;
        }
        __syncthreads();
      }
    } else if constexpr (Op == ServingEpilogueOp::kPartial) {
      for (int index = int(threadIdx.x); index < TileM * TileN; index += 128) {
        int row = index / TileN, col = index % TileN;
        int global_row = tile_m * TileM + row;
        int global_col = tile_n * TileN + col;
        if (global_row < M && global_col < N)
          partial[global_row * output_stride + global_col] = tile[index];
      }
    } else {
      constexpr int kOutputColumns =
          Op == ServingEpilogueOp::kSwiGLU ? TileN / 2 : TileN;
      for (int vector = int(threadIdx.x); vector < TileM * kOutputColumns / 8;
           vector += 128) {
        int row = vector / (kOutputColumns / 8);
        int out_col = (vector % (kOutputColumns / 8)) * 8;
        int global_row = tile_m * TileM + row;
        int global_col = tile_n * (Op == ServingEpilogueOp::kSwiGLU ? TileN / 2 : TileN)
                         + out_col;
        if (global_row >= M) continue;
        alignas(16) cutlass::bfloat16_t values[8];
        for (int element = 0; element < 8; ++element) {
          int local_col = out_col + element;
          int col = tile_n * TileN + local_col;
          if constexpr (Op == ServingEpilogueOp::kSwiGLU) {
            int pair = local_col / U;
            int offset = local_col % U;
            int gate = 2 * U * pair + offset;
            int up = gate + U;
            values[element] = ServingEpilogueValue<Op>::Apply(
                tile[row * TileN + gate], tile[row * TileN + up]);
          } else {
            auto r = residual && col < N
                         ? residual[global_row * output_stride + col]
                         : cutlass::bfloat16_t{};
            values[element] = ServingEpilogueValue<Op>::Apply(
                tile[row * TileN + local_col], r);
          }
        }
        int output_n = Op == ServingEpilogueOp::kSwiGLU ? N / 2 : N;
        if (global_col + 8 <= output_n &&
            (reinterpret_cast<std::uintptr_t>(output + global_row * output_stride + global_col) & 15) == 0) {
          // A full vector is one coalesced 16-byte global store.
          *reinterpret_cast<uint4*>(output + global_row * output_stride + global_col) =
              *reinterpret_cast<uint4 const*>(values);
        } else {
          for (int element = 0; element < 8; ++element)
            if (global_col + element < output_n)
              output[global_row * output_stride + global_col + element] =
                  values[element];
        }
      }
    }
  }
};

}  // namespace tilemega::backend
