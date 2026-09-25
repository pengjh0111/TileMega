// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/ServingRMSNormTaskBody.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using Element = cutlass::bfloat16_t;

__global__ void Run(Element const* input, Element const* weight,
                    Element* output, int width) {
  __shared__ float warp_sums[4];
  tilemega::codegen::ServingRMSNormTaskBody::RunRow(
      input, weight, output, int(blockIdx.x), 2, 1, width, 1e-6f,
      warp_sums);
}

int main() {
  constexpr int kWidth = 2048, kRows = 3;
  Element *input = nullptr, *weight = nullptr, *output = nullptr;
  cudaMallocManaged(&input, 2 * kRows * kWidth * sizeof(Element));
  cudaMallocManaged(&weight, kWidth * sizeof(Element));
  cudaMallocManaged(&output, kRows * kWidth * sizeof(Element));
  for (int row = 0; row < 2 * kRows; ++row)
    for (int col = 0; col < kWidth; ++col)
      input[row * kWidth + col] = Element(0.01f * ((row * 17 + col) % 101) - 0.5f);
  for (int col = 0; col < kWidth; ++col)
    weight[col] = Element(1.0f + 0.001f * (col % 11));
  Run<<<kRows, 128>>>(input, weight, output, kWidth);
  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    return 2;
  }
  for (int row = 0; row < kRows; ++row) {
    double sum = 0.0;
    for (int col = 0; col < kWidth; ++col) {
      float x = float(input[(2 * row + 1) * kWidth + col]);
      sum += double(x) * x;
    }
    float scale = 1.0f / std::sqrt(float(sum / kWidth) + 1e-6f);
    for (int col = 0; col < kWidth; ++col) {
      float x = float(input[(2 * row + 1) * kWidth + col]);
      float expected = float(Element(float(Element(x * scale)) *
                                     float(weight[col])));
      float actual = float(output[row * kWidth + col]);
      float ulp = expected == 0.0f ? std::ldexp(1.0f, -133)
                                   : std::ldexp(1.0f, std::ilogb(std::abs(expected)) - 7);
      if (std::abs(actual - expected) > ulp) {
        std::fprintf(stderr, "row %d col %d: %.9g vs %.9g\n",
                     row, col, actual, expected);
        return 3;
      }
    }
  }
  cudaFree(input);
  cudaFree(weight);
  cudaFree(output);
  std::puts("serving RMSNorm: pass");
  return 0;
}
