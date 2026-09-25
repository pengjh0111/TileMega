// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/FusedAttentionTaskBody.h>
#include <tilemega/Codegen/tasks/AttentionMergeTaskBody.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using Body = tilemega::codegen::FusedAttentionTaskBody<
    tilemega::arch::Sm89, 64, 4, 1, 16, 64, false>;
using Element = cutlass::bfloat16_t;

__global__ void Run(tilemega::codegen::ServingAttentionOperands operands) {
  __shared__ typename Body::SharedStorage storage;
  Body::Run(operands, storage, 0, 0, 0, int(blockIdx.x));
}

__global__ void Merge(float const* partial, float const* lse,
                      Element* context, int past) {
  tilemega::codegen::AttentionMergeTaskBody<64, 4, 1>::Run(
      partial, lse, context, 0, 0, 1, 128, 64, past);
}

int main() {
  Element *qkv, *key, *value, *cosine, *sine, *context;
  float *partial, *lse;
  assert(cudaMallocManaged(&qkv, 6 * 64 * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&key, 128 * 64 * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&value, 128 * 64 * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&cosine, 128 * 64 * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&sine, 128 * 64 * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&context, 4 * 64 * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&partial, 2 * 4 * 64 * sizeof(float)) == cudaSuccess);
  assert(cudaMallocManaged(&lse, 2 * 4 * sizeof(float)) == cudaSuccess);
  for (int i = 0; i < 6 * 64; ++i)
    qkv[i] = Element(float((i * 17 % 29) - 14) / 64);
  for (int i = 0; i < 128 * 64; ++i) {
    key[i] = Element(0.0f);
    value[i] = Element(0.0f);
    cosine[i] = Element(1.0f);
    sine[i] = Element(0.0f);
  }
  for (int d = 0; d < 64; ++d) {
    key[d] = Element(float((d * 3 % 13) - 6) / 32);
    value[d] = Element(float((d * 5 % 17) - 8) / 32);
  }
  std::vector<float> expected(4 * 64);
  for (int h = 0; h < 4; ++h) {
    float score[2] = {};
    for (int d = 0; d < 64; ++d) {
      score[0] += float(qkv[h * 64 + d]) * float(key[d]);
      score[1] += float(qkv[h * 64 + d]) * float(qkv[4 * 64 + d]);
    }
    float e0 = std::exp(score[0] / 8), e1 = std::exp(score[1] / 8);
    float sum = e0 + e1;
    for (int d = 0; d < 64; ++d)
      expected[h * 64 + d] =
          (e0 * float(value[d]) + e1 * float(qkv[5 * 64 + d])) / sum;
  }
  tilemega::codegen::ServingAttentionOperands operands{
      qkv, key, value, cosine, sine, nullptr, nullptr, context,
      partial, lse, 1, 1, 128, 1, 128, 1e-6f};
  Run<<<1, 128>>>(operands);
  cudaError_t status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "attention kernel: %s\n", cudaGetErrorString(status));
    return 3;
  }
  for (int i = 0; i < 4 * 64; ++i) {
    float actual = float(context[i]);
    if (std::abs(actual - expected[i]) >
        std::ldexp(std::abs(expected[i]), -7) + std::ldexp(1.0f, -8)) {
      std::fprintf(stderr, "attention i=%d actual=%g expected=%g\n",
                   i, actual, expected[i]);
      return 1;
    }
  }
  for (int d = 0; d < 64; ++d) {
    if (float(key[64 + d]) != float(qkv[4 * 64 + d]) ||
        float(value[64 + d]) != float(qkv[5 * 64 + d])) return 2;
  }
  for (int position = 0; position < 65; ++position)
    for (int d = 0; d < 64; ++d) {
      key[position * 64 + d] = Element(float((position * 11 + d * 3) % 31 - 15) / 64);
      value[position * 64 + d] = Element(float((position * 7 + d * 5) % 37 - 18) / 64);
    }
  operands.past = 65;
  operands.block_extent = 64;
  for (int h = 0; h < 4; ++h) {
    float peak = -INFINITY;
    float score[66];
    for (int position = 0; position < 66; ++position) {
      float dot = 0.0f;
      for (int d = 0; d < 64; ++d)
        dot += float(qkv[h * 64 + d]) *
            float(position < 65 ? key[position * 64 + d]
                                : qkv[4 * 64 + d]);
      score[position] = dot / 8;
      peak = std::max(peak, score[position]);
    }
    float normalizer = 0.0f;
    for (float& item : score) {
      item = std::exp(item - peak);
      normalizer += item;
    }
    for (int d = 0; d < 64; ++d) {
      float output = 0.0f;
      for (int position = 0; position < 66; ++position)
        output += score[position] *
            float(position < 65 ? value[position * 64 + d]
                                : qkv[5 * 64 + d]);
      expected[h * 64 + d] = output / normalizer;
    }
  }
  Run<<<2, 128>>>(operands);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  Merge<<<1, 128>>>(partial, lse, context, 65);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  for (int i = 0; i < 4 * 64; ++i)
    if (std::abs(float(context[i]) - expected[i]) >
        std::ldexp(std::abs(expected[i]), -7) + std::ldexp(1.0f, -8)) {
      std::fprintf(stderr, "merged attention i=%d actual=%g expected=%g\n",
                   i, float(context[i]), expected[i]);
      return 4;
    }
  std::puts("serving fused decode attention: pass");
  cudaFree(qkv); cudaFree(key); cudaFree(value); cudaFree(cosine);
  cudaFree(sine); cudaFree(context); cudaFree(partial); cudaFree(lse);
}
