// SPDX-License-Identifier: BSD-3-Clause
// Numerical probe for the shared serving epilogue, including predicated rows
// and columns. Run explicitly on a CUDA device.
#include <tilemega/Backend/ServingEpilogue.h>
#include <tilemega/Backend/ServingGemm.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using tilemega::backend::ServingEpilogue;
using tilemega::backend::ServingEpilogueOp;
using Config = tilemega::backend::ServingGemmConfig<
    tilemega::arch::Sm89, 16, 128, 64, 2>;

template <ServingEpilogueOp Op>
__global__ void Probe(cutlass::bfloat16_t* output, float* partial,
                      float* argmax_value, int* argmax_index, int rows,
                      int columns, int output_stride) {
  extern __shared__ char shared[];
  Config::TiledMma mma;
  auto fragment = cute::partition_fragment_C(
      mma, cute::take<0, 2>(Config::TileShape{}));
  auto owned = mma.get_thread_slice(int(threadIdx.x)).partition_C(
      cute::make_identity_tensor(cute::Shape<cute::_16, cute::_128>{}));
  for (int i = 0; i < cute::size(fragment); ++i) {
    int row = int(blockIdx.x) * 16 + cute::get<0>(owned(i));
    int col = cute::get<1>(owned(i));
    if constexpr (Op == ServingEpilogueOp::kArgmaxPartial)
      fragment(i) = col == 5 || col == 6 ? 1.0f : -float(col) * 0.01f;
    else
      fragment(i) = 0.5f + 0.1f * float(row) + 0.01f * float(col);
  }
  ServingEpilogue<Op, 16, 128>::Run(
      fragment, mma, shared, int(blockIdx.x), 0, rows, columns,
      output_stride, output, output, partial, argmax_value, argmax_index);
}

template <class T>
T* Managed(std::size_t count) {
  T* result = nullptr;
  if (cudaMallocManaged(&result, count * sizeof(T)) != cudaSuccess) std::exit(2);
  return result;
}

template <ServingEpilogueOp Op>
void Check(int rows, int columns) {
  int stride = Op == ServingEpilogueOp::kSwiGLU ? columns / 2 : columns;
  if constexpr (Op == ServingEpilogueOp::kArgmaxPartial) stride = 1;
  auto* output = Managed<cutlass::bfloat16_t>(rows * stride);
  auto* partial = Managed<float>(rows * columns);
  auto* best = Managed<float>(rows);
  auto* index = Managed<int>(rows);
  for (int i = 0; i < rows * stride; ++i)
    output[i] = cutlass::bfloat16_t(0.5f);
  Probe<Op><<<(rows + 15) / 16, 128, Config::kSharedBytes>>>(
      output, partial, best, index, rows, columns, stride);
  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA probe failed: %s\n", cudaGetErrorString(status));
    std::exit(3);
  }
  for (int row = 0; row < rows; ++row) {
    if constexpr (Op == ServingEpilogueOp::kArgmaxPartial) {
      if (best[row] != 1.0f || index[row] != 5) {
        std::fprintf(stderr, "argmax row %d: %.9g index %d\n", row, best[row], index[row]);
        std::exit(4);
      }
    } else {
      for (int col = 0; col < stride; ++col) {
        float actual = Op == ServingEpilogueOp::kPartial
                           ? partial[row * stride + col]
                           : float(output[row * stride + col]);
        float expected;
        if constexpr (Op == ServingEpilogueOp::kSwiGLU) {
          int gate_col = 32 * (col / 16) + col % 16;
          float g = float(cutlass::bfloat16_t(
              0.5f + 0.1f * row + 0.01f * gate_col));
          float v = float(cutlass::bfloat16_t(
              0.5f + 0.1f * row + 0.01f * (gate_col + 16)));
          expected = float(cutlass::bfloat16_t(
              float(cutlass::bfloat16_t(g / (1.0f + std::exp(-g)))) * v));
        } else {
          float x = 0.5f + 0.1f * row + 0.01f * col;
          if constexpr (Op == ServingEpilogueOp::kPartial) expected = x;
          else if constexpr (Op == ServingEpilogueOp::kResidual)
            expected = float(cutlass::bfloat16_t(
                float(cutlass::bfloat16_t(x)) + 0.5f));
          else expected = float(cutlass::bfloat16_t(x));
        }
        float tolerance = expected == 0.0f ? std::ldexp(1.0f, -133)
                                            : std::ldexp(1.0f, std::ilogb(std::abs(expected)) - 7);
        if constexpr (Op == ServingEpilogueOp::kPartial)
          tolerance = 1e-6f;
        if (std::abs(actual - expected) > tolerance) {
          std::fprintf(stderr, "op %d row %d col %d: %.9g vs %.9g\n",
                       int(Op), row, col, actual, expected);
          std::exit(5);
        }
      }
    }
  }
  cudaFree(output);
  cudaFree(partial);
  cudaFree(best);
  cudaFree(index);
}

int main() {
  for (int rows : {1, 3, 17}) {
    Check<ServingEpilogueOp::kStore>(rows, 73);
    Check<ServingEpilogueOp::kResidual>(rows, 73);
    Check<ServingEpilogueOp::kSwiGLU>(rows, 128);
    Check<ServingEpilogueOp::kArgmaxPartial>(rows, 73);
    Check<ServingEpilogueOp::kPartial>(rows, 73);
  }
  std::puts("serving epilogue: pass");
  return 0;
}
