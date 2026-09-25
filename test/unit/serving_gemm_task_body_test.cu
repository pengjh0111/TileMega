// SPDX-License-Identifier: BSD-3-Clause
// CUDA numerical and memcheck probe for the vectorized SM80-class mainloop.
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using Body = tilemega::codegen::ServingGemmTaskBody<
    tilemega::arch::Sm89, 16, 128, 64, 2>;
using NarrowBody = tilemega::codegen::ServingGemmTaskBody<
    tilemega::arch::Sm89, 16, 32, 64, 2>;
using Element = cutlass::bfloat16_t;

template <class KernelBody>
__global__ void Run(tilemega::codegen::ServingGemmOperands operands) {
  extern __shared__ char storage[];
  KernelBody::Run(operands, int(blockIdx.x), int(blockIdx.y), storage);
}

template <class KernelBody, int TileM, int TileN>
void Check(int rows, int columns, int reduction) {
  int const pitch = (reduction + 7) & ~7;
  Element *a = nullptr, *b = nullptr, *out = nullptr;
  if (cudaMallocManaged(&a, std::size_t(rows) * pitch * sizeof(Element)) != cudaSuccess ||
      cudaMallocManaged(&b, std::size_t(columns) * pitch * sizeof(Element)) != cudaSuccess ||
      cudaMallocManaged(&out, std::size_t(rows) * columns * sizeof(Element)) != cudaSuccess)
    std::exit(2);
  for (int row = 0; row < rows; ++row)
    for (int k = 0; k < pitch; ++k)
      a[row * pitch + k] = Element(k < reduction ? 0.01f : 0.0f);
  for (int col = 0; col < columns; ++col)
    for (int k = 0; k < pitch; ++k)
      b[col * pitch + k] = Element(k < reduction ? 0.02f : 0.0f);
  for (int i = 0; i < rows * columns; ++i) out[i] = Element(-17.0f);
  tilemega::codegen::ServingGemmOperands operands{};
  operands.a = a;
  operands.b = b;
  operands.output = out;
  operands.m = rows;
  operands.n = columns;
  operands.k_total = reduction;
  operands.k_count = reduction;
  operands.a_row_stride = pitch;
  operands.b_row_stride = pitch;
  operands.output_stride = columns;
  Run<KernelBody><<<dim3((rows + TileM - 1) / TileM,
                           (columns + TileN - 1) / TileN),
                    128, KernelBody::kSharedBytes>>>(operands);
  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "serving GEMM failed: %s\n", cudaGetErrorString(status));
    std::exit(3);
  }
  float expected = float(Element(reduction * float(Element(0.01f)) *
                                   float(Element(0.02f))));
  float tolerance = std::ldexp(1.0f, std::ilogb(std::abs(expected)) - 7);
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < columns; ++col)
      if (std::abs(float(out[row * columns + col]) - expected) > tolerance) {
        std::fprintf(stderr, "M=%d N=%d K=%d row=%d col=%d: %.9g vs %.9g\n",
                     rows, columns, reduction, row, col,
                     float(out[row * columns + col]), expected);
        std::exit(4);
      }
  cudaFree(a);
  cudaFree(b);
  cudaFree(out);
}

int main() {
  for (int rows : {1, 3, 17})
    for (int columns : {64, 73})
      for (int reduction : {2001, 2048})
        Check<Body, 16, 128>(rows, columns, reduction);
  for (int rows : {1, 3, 17})
    for (int columns : {31, 32})
      for (int reduction : {2001, 2048})
        Check<NarrowBody, 16, 32>(rows, columns, reduction);
  Check<Body, 16, 128>(1, 3072, 2048);
  Check<Body, 16, 128>(16, 256, 6144);
  Check<Body, 16, 128>(64, 2048, 8192);
  Check<Body, 16, 128>(1024, 64, 2048);
  std::puts("serving GEMM mainloop: pass");
  return 0;
}
