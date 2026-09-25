// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/ServingGemmCombineTaskBody.h>

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdio>

using Element = cutlass::bfloat16_t;
using tilemega::backend::ServingEpilogueOp;

template <ServingEpilogueOp Op>
__global__ void Combine(float const* partial, int split, int M, int N,
                        Element* output, Element const* residual,
                        float* argmax_value, int* argmax_index) {
  __shared__ float tile[16 * 64];
  constexpr int kStride = Op == ServingEpilogueOp::kSwiGLU ? 32 :
                          Op == ServingEpilogueOp::kArgmaxPartial ? 1 : 64;
  tilemega::codegen::ServingGemmCombineTaskBody<16, 64, Op>::Run(
      partial, split, 0, 0, M, N, N, kStride, output, residual,
      argmax_value, argmax_index, tile);
}

int main() {
  constexpr int kM = 3, kN = 64, kSplit = 4;
  float *partial = nullptr, *argmax_value = nullptr;
  int* argmax_index = nullptr;
  Element *output = nullptr, *residual = nullptr;
  cudaMallocManaged(&partial, kSplit * kM * kN * sizeof(float));
  cudaMallocManaged(&output, kM * kN * sizeof(Element));
  cudaMallocManaged(&residual, kM * kN * sizeof(Element));
  cudaMallocManaged(&argmax_value, kM * sizeof(float));
  cudaMallocManaged(&argmax_index, kM * sizeof(int));
  for (int s = 0; s < kSplit; ++s)
    for (int i = 0; i < kM * kN; ++i)
      partial[s * kM * kN + i] =
          0.01f * float((i * 7 + s * 3) % 37) - 0.07f;
  for (int i = 0; i < kM * kN; ++i) residual[i] = Element(0.125f);
  Combine<ServingEpilogueOp::kStore><<<1, 128>>>(
      partial, kSplit, kM, kN, output, residual, argmax_value, argmax_index);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  for (int i = 0; i < kM * kN; ++i) {
    float sum = 0;
    for (int s = 0; s < kSplit; ++s)
      sum += partial[s * kM * kN + i];
    assert(output[i] == Element(sum));
  }
  Combine<ServingEpilogueOp::kResidual><<<1, 128>>>(
      partial, kSplit, kM, kN, output, residual, argmax_value, argmax_index);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  for (int i = 0; i < kM * kN; ++i) {
    float sum = 0;
    for (int s = 0; s < kSplit; ++s)
      sum += partial[s * kM * kN + i];
    assert(output[i] == Element(float(Element(sum)) + 0.125f));
  }
  Combine<ServingEpilogueOp::kSwiGLU><<<1, 128>>>(
      partial, kSplit, kM, kN, output, residual, argmax_value, argmax_index);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  for (int row = 0; row < kM; ++row)
    for (int col = 0; col < kN / 2; ++col) {
      int gate_col = 32 * (col / 16) + col % 16;
      int up_col = gate_col + 16;
      float gate = 0, up = 0;
      for (int s = 0; s < kSplit; ++s) {
        gate += partial[s * kM * kN + row * kN + gate_col];
        up += partial[s * kM * kN + row * kN + up_col];
      }
      float g = float(Element(gate)), v = float(Element(up));
      Element expected = Element(float(Element(g / (1 + std::exp(-g)))) * v);
      assert(std::abs(float(output[row * (kN / 2) + col]) - float(expected)) <
             0.004f);
    }
  Combine<ServingEpilogueOp::kArgmaxPartial><<<1, 128>>>(
      partial, kSplit, kM, kN, output, residual, argmax_value, argmax_index);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  for (int row = 0; row < kM; ++row) {
    float best = -INFINITY;
    int best_col = -1;
    for (int col = 0; col < kN; ++col) {
      float value = 0;
      for (int s = 0; s < kSplit; ++s)
        value += partial[s * kM * kN + row * kN + col];
      value = float(Element(value));
      if (value > best || (value == best && col < best_col)) {
        best = value;
        best_col = col;
      }
    }
    assert(argmax_value[row] == best && argmax_index[row] == best_col);
  }
  cudaFree(partial);
  cudaFree(output);
  cudaFree(residual);
  cudaFree(argmax_value);
  cudaFree(argmax_index);
  std::puts("serving combine: pass");
}
