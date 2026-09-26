// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/FusedAttentionTaskBody.h>
#include <tilemega/Codegen/tasks/AttentionMergeTaskBody.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef TILEMEGA_TEST_KV_TILE
#define TILEMEGA_TEST_KV_TILE 64
#endif
namespace {
using Element = cutlass::bfloat16_t;
constexpr int Cap = 1088;
template <int D, int Q, bool Norm>
using Body = tilemega::codegen::FusedAttentionTaskBody<
    tilemega::arch::Sm89, D, Q, 1, 16, TILEMEGA_TEST_KV_TILE, Norm>;

template <int D, int Q, bool Norm>
__global__ void Run(tilemega::codegen::ServingAttentionOperands operands) {
  extern __shared__ __align__(16) unsigned char bytes[];
  auto& storage = *reinterpret_cast<typename Body<D, Q, Norm>::SharedStorage*>(bytes);
  Body<D, Q, Norm>::Run(operands, storage, 0, 0, 0, int(blockIdx.x));
}

template <int D, int Q>
__global__ void Merge(float const* partial, float const* lse,
                      Element* context, int extent, int past) {
  tilemega::codegen::AttentionMergeTaskBody<D, Q, 1>::Run(
      partial, lse, context, 0, 0, 1, Cap, extent, past);
}

template <int D, bool Norm>
Element Rotate(Element const* x, Element const* norm, Element const* cos,
               Element const* sin, int d) {
  float inverse = 1.0f;
  if constexpr (Norm) {
    float square = 0;
    for (int k = 0; k < D; ++k) square += float(x[k]) * float(x[k]);
    inverse = 1.0f / std::sqrt(square / D + 1e-6f);
  }
  int partner = d < D / 2 ? d + D / 2 : d - D / 2;
  Element a = Norm ? Element(float(Element(float(x[d]) * inverse)) * float(norm[d])) : x[d];
  Element b = Norm ? Element(float(Element(float(x[partner]) * inverse)) *
                             float(norm[partner])) : x[partner];
  Element first = Element(float(a) * float(cos[d]));
  Element second = Element((d < D / 2 ? -float(b) : float(b)) * float(sin[d]));
  return Element(float(first) + float(second));
}

int BF16Ulp(Element actual, Element expected) {
  std::uint16_t a = 0, b = 0;
  static_assert(sizeof(Element) == sizeof(a));
  std::memcpy(&a, &actual, sizeof(a));
  std::memcpy(&b, &expected, sizeof(b));
  auto ordered = [](std::uint16_t bits) {
    return int(bits & 0x8000 ? 0xffffu - bits : 0x8000u + bits);
  };
  return std::abs(ordered(a) - ordered(b));
}

template <int D, int Q, bool Norm>
bool Check(int past, int extent, Element* qkv, Element* key, Element* value,
           Element* cosine, Element* sine, Element* qnorm, Element* knorm,
           Element* context, float* partial, float* lse) {
  int blocks = (Cap + extent - 1) / extent;
  for (int i = 0; i < Q * D; ++i) context[i] = Element(0.0f);
  for (int i = 0; i < blocks * Q; ++i) lse[i] = -INFINITY;
  for (int i = 0; i < blocks * Q * D; ++i) partial[i] = 0;
  auto operands = tilemega::codegen::ServingAttentionOperands{
      qkv, key, value, cosine, sine, qnorm, knorm,
      context, partial, lse, 1, 1, Cap, past, extent, 1e-6f};
  Run<D, Q, Norm><<<blocks, 128,
      sizeof(typename Body<D, Q, Norm>::SharedStorage)>>>(operands);
  if (cudaDeviceSynchronize() != cudaSuccess) return false;
  if (blocks > 1) {
    Merge<D, Q><<<1, 128>>>(partial, lse, context, extent, past);
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
  }
  std::vector<Element> query(Q * D), newest(D);
  for (int h = 0; h < Q; ++h)
    for (int d = 0; d < D; ++d)
      query[h * D + d] = Rotate<D, Norm>(qkv + h * D, qnorm,
                               cosine + past * D, sine + past * D, d);
  for (int d = 0; d < D; ++d)
    newest[d] = Rotate<D, Norm>(qkv + Q * D, knorm,
                       cosine + past * D, sine + past * D, d);
  for (int d = 0; d < D; ++d) {
    if (BF16Ulp(key[past * D + d], newest[d]) > 1 ||
        value[past * D + d] != qkv[(Q + 1) * D + d]) return false;
  }
  for (int h = 0; h < Q; ++h) {
    std::vector<float> scores(past + 1);
    float peak = -INFINITY;
    for (int position = 0; position <= past; ++position) {
      float dot = 0;
      for (int d = 0; d < D; ++d)
        dot += float(query[h * D + d]) * float(key[position * D + d]);
      scores[position] = dot / std::sqrt(float(D));
      peak = std::max(peak, scores[position]);
    }
    float denominator = 0;
    for (auto& score : scores) {
      score = std::exp(score - peak);
      denominator += score;
    }
    for (int d = 0; d < D; ++d) {
      float numerator = 0;
      for (int position = 0; position <= past; ++position)
        numerator += float(Element(scores[position])) *
                     float(value[position * D + d]);
      float expected = numerator / denominator;
      float tolerance = std::ldexp(std::abs(expected), -7) +
                        std::ldexp(1.0f, -8);
      if (std::abs(float(context[h * D + d]) - expected) > tolerance) {
        std::fprintf(stderr,
                     "D=%d Q=%d norm=%d past=%d extent=%d head=%d dim=%d got=%g expected=%g\n",
                     D, Q, int(Norm), past, extent, h, d,
                     float(context[h * D + d]), expected);
        return false;
      }
    }
  }
  return true;
}
}  // namespace

template <int D, int Q, bool Norm>
bool RunSuite() {
  constexpr int GroupWidth = (Q + 2) * D;
  Element *qkv, *key, *value, *cosine, *sine, *qnorm, *knorm, *context;
  float *partial, *lse;
  assert(cudaMallocManaged(&qkv, GroupWidth * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&key, Cap * D * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&value, Cap * D * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&cosine, Cap * D * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&sine, Cap * D * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&qnorm, D * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&knorm, D * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&context, Q * D * sizeof(Element)) == cudaSuccess);
  assert(cudaMallocManaged(&partial, 17 * Q * D * sizeof(float)) == cudaSuccess);
  assert(cudaMallocManaged(&lse, 17 * Q * sizeof(float)) == cudaSuccess);
  assert(cudaFuncSetAttribute(Run<D, Q, Norm>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              sizeof(typename Body<D, Q, Norm>::SharedStorage)) == cudaSuccess);
  for (int i = 0; i < GroupWidth; ++i)
    qkv[i] = Element(float((i * 17 % 41) - 20) / 128);
  for (int d = 0; d < D; ++d) {
    qnorm[d] = Element(0.9f + float(d % 7) / 32);
    knorm[d] = Element(0.8f + float(d % 11) / 32);
  }
  for (int position = 0; position < Cap; ++position)
    for (int d = 0; d < D; ++d) {
      key[position * D + d] = Element(float((position * 3 + d * 5) % 37 - 18) / 128);
      value[position * D + d] = Element(float((position * 7 + d * 11) % 47 - 23) / 128);
      cosine[position * D + d] = Element(0.75f);
      sine[position * D + d] = Element(0.5f);
    }
  for (int past : {1, 63, 64, 65, 1086})
    for (int extent : {64, 256, Cap})
      if (!Check<D, Q, Norm>(past, extent, qkv, key, value, cosine, sine,
                             Norm ? qnorm : nullptr, Norm ? knorm : nullptr,
                             context, partial, lse)) return false;
  cudaFree(qkv); cudaFree(key); cudaFree(value); cudaFree(cosine);
  cudaFree(sine); cudaFree(qnorm); cudaFree(knorm); cudaFree(context);
  cudaFree(partial); cudaFree(lse);
  return true;
}

int main() {
  if (!RunSuite<64, 4, false>() || !RunSuite<64, 4, true>() ||
      !RunSuite<128, 2, false>() || !RunSuite<128, 2, true>()) return 1;
  std::puts("serving decode attention matrix: 60/60 pass");
}
