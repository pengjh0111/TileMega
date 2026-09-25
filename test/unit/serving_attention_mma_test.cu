// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/ServingAttentionMma.h>

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdio>

using Mma = tilemega::backend::ServingAttentionMma<tilemega::arch::Sm89,
                                                   16, 64, 64>;
using PvMma = tilemega::backend::ServingAttentionMma<tilemega::arch::Sm89,
                                                     16, 64, 64, true>;
using Element = cutlass::bfloat16_t;

__global__ void Run(Element const* input_a, Element const* input_b,
                    float* output) {
  __shared__ alignas(16) Element a[Mma::kAElements];
  __shared__ alignas(16) Element b[Mma::kBElements];
  auto sA = cute::make_tensor(cute::make_smem_ptr(a), Mma::LayoutA{});
  auto sB = cute::make_tensor(cute::make_smem_ptr(b), Mma::LayoutB{});
  for (int i = int(threadIdx.x); i < 16 * 64; i += 128)
    sA(i / 64, i % 64) = input_a[i];
  for (int i = int(threadIdx.x); i < 64 * 64; i += 128)
    sB(i / 64, i % 64) = input_b[i];
  __syncthreads();
  Mma::Run(a, b, output, 64);
}

__global__ void RunPV(Element const* input_a, Element const* input_b,
                      float* output) {
  __shared__ alignas(16) Element a[PvMma::kAElements];
  __shared__ alignas(16) Element b[PvMma::kBElements];
  auto sA = cute::make_tensor(cute::make_smem_ptr(a), PvMma::LayoutA{});
  auto sB = cute::make_tensor(cute::make_smem_ptr(b), PvMma::LayoutB{});
  for (int i = int(threadIdx.x); i < 16 * 64; i += 128)
    sA(i / 64, i % 64) = input_a[i];
  for (int i = int(threadIdx.x); i < 64 * 64; i += 128)
    sB(i / 64, i % 64) = input_b[i];
  __syncthreads();
  PvMma::Run(a, b, output, 64);
}

int main() {
  Element *a = nullptr, *b = nullptr;
  float* output = nullptr;
  assert(cudaMallocManaged(&a, 16 * 64 * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&b, 64 * 64 * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&output, 16 * 64 * sizeof(float)) == cudaSuccess);
  for (int i = 0; i < 16 * 64; ++i) a[i] = Element(float((i % 13) - 6) / 32);
  for (int i = 0; i < 64 * 64; ++i) b[i] = Element(float((i % 11) - 5) / 64);
  Run<<<1, 128>>>(a, b, output);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 64; ++n) {
      float expected = 0;
      for (int k = 0; k < 64; ++k)
        expected += float(a[m * 64 + k]) * float(b[n * 64 + k]);
      if (std::abs(output[m * 64 + n] - expected) > 1e-4f) {
        std::fprintf(stderr, "m=%d n=%d output=%g reference=%g\n",
                     m, n, output[m * 64 + n], expected);
        return 1;
      }
    }
  RunPV<<<1, 128>>>(a, b, output);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 64; ++n) {
      float expected = 0;
      for (int k = 0; k < 64; ++k)
        expected += float(a[m * 64 + k]) * float(b[n * 64 + k]);
      if (std::abs(output[m * 64 + n] - expected) > 1e-4f) {
        std::fprintf(stderr, "PV m=%d n=%d output=%g reference=%g\n",
                     m, n, output[m * 64 + n], expected);
        return 2;
      }
    }
  cudaFree(a);
  cudaFree(b);
  cudaFree(output);
  std::puts("serving attention QK and PV MMA: pass");
}
