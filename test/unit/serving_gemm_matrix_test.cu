// SPDX-License-Identifier: BSD-3-Clause
// Emit one complete serving GEMM case for comparison with torch.bfloat16.
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Codegen/tasks/ServingGemmCombineTaskBody.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

using Element = cutlass::bfloat16_t;
using Op = tilemega::backend::ServingEpilogueOp;

template <class Body>
__global__ void RunGemm(tilemega::codegen::ServingGemmOperands operands) {
  extern __shared__ char storage[];
  Body::Run(operands, int(blockIdx.x), int(blockIdx.y), storage);
}

template <int TileM, int TileN, Op Operation>
__global__ void RunCombine(float const* partial, int split, int rows,
                           int columns, int output_stride, Element* output,
                           Element const* residual, float* argmax_value,
                           int* argmax_index) {
  extern __shared__ float combine_storage[];
  tilemega::codegen::ServingGemmCombineTaskBody<TileM, TileN, Operation>::Run(
      partial, split, int(blockIdx.x), int(blockIdx.y), rows, columns,
      columns, output_stride, output, residual, argmax_value,
      argmax_index, combine_storage);
}

void CheckCuda(cudaError_t code, char const* where) {
  if (code != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " +
                             cudaGetErrorString(code));
}

template <class T>
T* Managed(std::size_t count) {
  T* ptr = nullptr;
  CheckCuda(cudaMallocManaged(&ptr, count * sizeof(T)), "cudaMallocManaged");
  return ptr;
}

template <int TileM, int TileN, Op Operation>
void Combine(float const* partial, int split, int rows, int columns,
             int output_stride, Element* output, Element const* residual,
             float* argmax_value, int* argmax_index) {
  auto kernel = RunCombine<TileM, TileN, Operation>;
  constexpr int bytes = 4 * TileM * TileN;
  if constexpr (bytes > 48 * 1024)
    CheckCuda(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes),
        "combine shared-memory attribute");
  kernel<<<dim3((rows + TileM - 1) / TileM,
                  (columns + TileN - 1) / TileN), 128, bytes>>>(
      partial, split, rows, columns, output_stride, output, residual,
      argmax_value, argmax_index);
}

template <int TileM, int TileN>
void DispatchCombine(int operation, float const* partial, int split,
                     int rows, int columns, int output_stride,
                     Element* output, Element const* residual,
                     float* argmax_value, int* argmax_index) {
  switch (operation) {
    case 0: Combine<TileM, TileN, Op::kStore>(partial, split, rows, columns,
        output_stride, output, residual, argmax_value, argmax_index); break;
    case 1: Combine<TileM, TileN, Op::kResidual>(partial, split, rows, columns,
        output_stride, output, residual, argmax_value, argmax_index); break;
    case 2: Combine<TileM, TileN, Op::kSwiGLU>(partial, split, rows, columns,
        output_stride, output, residual, argmax_value, argmax_index); break;
    case 3: Combine<TileM, TileN, Op::kArgmaxPartial>(partial, split, rows,
        columns, output_stride, output, residual, argmax_value,
        argmax_index); break;
    default: throw std::invalid_argument("unknown combine operation");
  }
}

template <int TileM, int TileN, int TileK, int Stages>
void RunCase(int rows, int columns, int reduction, int split, int operation,
             char const* destination) {
  using Body = tilemega::codegen::ServingGemmTaskBody<
      tilemega::arch::Sm89, TileM, TileN, TileK, Stages>;
  if (rows < 1 || columns < 1 || reduction < 1 ||
      (split != 1 && split != 4) || reduction % split)
    throw std::invalid_argument("invalid GEMM matrix case");
  int const output_columns = operation == 2 ? columns / 2 : columns;
  int const argmax_columns = (columns + TileN - 1) / TileN;
  int const output_stride = operation == 3 ? argmax_columns : output_columns;
  auto* a = Managed<Element>(std::size_t(rows) * reduction);
  auto* b = Managed<Element>(std::size_t(columns) * reduction);
  auto* residual = Managed<Element>(std::size_t(rows) * columns);
  auto* output = Managed<Element>(std::size_t(rows) * output_columns);
  auto* partial = Managed<float>(std::size_t(split) * rows * columns);
  auto* argmax_value = Managed<float>(std::size_t(rows) * argmax_columns);
  auto* argmax_index = Managed<int>(std::size_t(rows) * argmax_columns);
  for (int row = 0; row < rows; ++row)
    for (int k = 0; k < reduction; ++k)
      a[std::size_t(row) * reduction + k] =
          Element(float((row * 11 + k * 3) % 7 - 3) / 64);
  for (int column = 0; column < columns; ++column)
    for (int k = 0; k < reduction; ++k)
      b[std::size_t(column) * reduction + k] =
          Element(float((column * 13 + k * 2) % 5 - 2) / 64);
  for (std::size_t i = 0; i < std::size_t(rows) * columns; ++i)
    residual[i] = Element(0.125f);
  auto kernel = RunGemm<Body>;
  if constexpr (Body::kSharedBytes > 48 * 1024)
    CheckCuda(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Body::kSharedBytes),
        "GEMM shared-memory attribute");
  tilemega::codegen::ServingGemmOperands operands{};
  operands.a = a;
  operands.b = b;
  operands.residual = residual;
  operands.output = output;
  operands.partial = partial;
  operands.argmax_value = argmax_value;
  operands.argmax_index = argmax_index;
  operands.m = rows;
  operands.n = columns;
  operands.k_total = reduction;
  operands.a_row_stride = operands.b_row_stride = reduction;
  operands.output_stride = output_stride;
  operands.epilogue = split == 1 ? static_cast<Op>(operation) : Op::kPartial;
  for (int chunk = 0; chunk < split; ++chunk) {
    operands.k_begin = chunk * (reduction / split);
    operands.k_count = reduction / split;
    if (split != 1) {
      operands.partial = partial + std::size_t(chunk) * rows * columns;
      operands.output_stride = columns;
    }
    kernel<<<dim3((rows + TileM - 1) / TileM,
                    (columns + TileN - 1) / TileN),
              128, Body::kSharedBytes>>>(operands);
    CheckCuda(cudaGetLastError(), "GEMM launch");
  }
  if (split != 1)
    DispatchCombine<TileM, TileN>(operation, partial, split, rows,
        columns, output_stride, output, residual,
        argmax_value, argmax_index);
  CheckCuda(cudaDeviceSynchronize(), "GEMM and combine");
  std::ofstream stream(destination, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open output dump");
  if (operation == 3) {
    stream.write(reinterpret_cast<char*>(argmax_value),
        std::size_t(rows) * argmax_columns * sizeof(float));
    stream.write(reinterpret_cast<char*>(argmax_index),
        std::size_t(rows) * argmax_columns * sizeof(int));
  } else {
    stream.write(reinterpret_cast<char*>(output),
        std::size_t(rows) * output_columns * sizeof(Element));
  }
  if (!stream) throw std::runtime_error("cannot write output dump");
  cudaFree(a); cudaFree(b); cudaFree(residual); cudaFree(output);
  cudaFree(partial); cudaFree(argmax_value); cudaFree(argmax_index);
}

int main(int argc, char** argv) {
  if (argc != 8) return 2;
  try {
    int config = std::atoi(argv[1]);
    int rows = std::atoi(argv[2]);
    int columns = std::atoi(argv[3]);
    int reduction = std::atoi(argv[4]);
    int split = std::atoi(argv[5]);
    int operation = std::atoi(argv[6]);
    switch (config) {
      case 0: RunCase<16,128,64,2>(rows,columns,reduction,split,operation,argv[7]); break;
      case 1: RunCase<16,128,64,4>(rows,columns,reduction,split,operation,argv[7]); break;
      case 2: RunCase<16,128,128,2>(rows,columns,reduction,split,operation,argv[7]); break;
      case 3: RunCase<16,64,128,3>(rows,columns,reduction,split,operation,argv[7]); break;
      case 4: RunCase<32,128,64,3>(rows,columns,reduction,split,operation,argv[7]); break;
      case 5: RunCase<64,128,64,3>(rows,columns,reduction,split,operation,argv[7]); break;
      case 6: RunCase<128,128,64,2>(rows,columns,reduction,split,operation,argv[7]); break;
      default: throw std::invalid_argument("unknown GEMM config");
    }
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
