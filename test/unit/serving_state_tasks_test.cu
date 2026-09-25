// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/ServingArgmaxReduceTaskBody.h>
#include <tilemega/Codegen/tasks/ServingEmbeddingTaskBody.h>

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdio>

using Element = cutlass::bfloat16_t;

__global__ void Embed(int const* tokens, Element const* table, Element* output,
                      int seq, int past, int cap, int width, int vocab) {
  tilemega::codegen::ServingEmbeddingTaskBody::RunRow(
      tokens, table, output, int(blockIdx.x), seq, past, cap, width, vocab);
}

__global__ void Reduce(float const* values, int const* indices, int* tokens,
                       int count, int cap, int position) {
  __shared__ tilemega::codegen::ServingArgmaxReduceTaskBody::SharedStorage shared;
  tilemega::codegen::ServingArgmaxReduceTaskBody::RunRow(
      values, indices, tokens, int(blockIdx.x), count, cap, position, &shared);
}

int main() {
  constexpr int kBatch = 3, kSeq = 2, kCap = 17, kWidth = 2051;
  constexpr int kVocab = 9, kParts = 137;
  int *tokens = nullptr, *indices = nullptr;
  float* values = nullptr;
  Element *table = nullptr, *output = nullptr;
  cudaMallocManaged(&tokens, kBatch * kCap * sizeof(int));
  cudaMallocManaged(&indices, kBatch * kParts * sizeof(int));
  cudaMallocManaged(&values, kBatch * kParts * sizeof(float));
  cudaMallocManaged(&table, kVocab * kWidth * sizeof(Element));
  cudaMallocManaged(&output, kBatch * kSeq * kWidth * sizeof(Element));
  for (int i = 0; i < kBatch * kCap; ++i) tokens[i] = 0;
  for (int b = 0; b < kBatch; ++b)
    for (int s = 0; s < kSeq; ++s)
      tokens[b * kCap + 7 + s] = (b * 3 + s) % kVocab;
  for (int i = 0; i < kVocab * kWidth; ++i)
    table[i] = Element(float(i % 127) * 0.015625f);
  Embed<<<kBatch * kSeq, 128>>>(tokens, table, output, kSeq, 7, kCap,
                                 kWidth, kVocab);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  for (int b = 0; b < kBatch; ++b)
    for (int s = 0; s < kSeq; ++s) {
      int id = tokens[b * kCap + 7 + s];
      for (int d = 0; d < kWidth; ++d)
        assert(output[(b * kSeq + s) * kWidth + d] == table[id * kWidth + d]);
    }
  for (int b = 0; b < kBatch; ++b)
    for (int p = 0; p < kParts; ++p) {
      values[b * kParts + p] = p == 50 || p == 90 ? 3.0f : float(p % 19);
      indices[b * kParts + p] = 1000 - p;
    }
  Reduce<<<kBatch, 128>>>(values, indices, tokens, kParts, kCap, 9);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  for (int b = 0; b < kBatch; ++b) {
    float best = -INFINITY;
    int index = INT_MAX;
    for (int p = 0; p < kParts; ++p) {
      float value = values[b * kParts + p];
      int candidate = indices[b * kParts + p];
      if (value > best || (value == best && candidate < index)) {
        best = value;
        index = candidate;
      }
    }
    assert(tokens[b * kCap + 9] == index);
  }
  cudaFree(tokens);
  cudaFree(indices);
  cudaFree(values);
  cudaFree(table);
  cudaFree(output);
  std::puts("serving state tasks: pass");
}
