// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Codegen/tasks/mx/AttentionCommon.h
//
// Attention operands and the q/k RMSNorm + RoPE row transform shared by the
// xcore1000 decode and prefill attention bodies. The rounding sequence of
// RotateRow is the NVIDIA one (bf16 after norm, after weight, after each RoPE
// product), so HF reference agreement does not depend on the target.
#pragma once

#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Codegen/tasks/ServingTaskProfile.h>
#include <tilemega/Target/Maca/Bf16.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace tilemega::codegen {

using codegen::executor::ComputeSync;
using codegen::executor::ComputeThread;
using codegen::executor::kComputeThreads;

struct ServingAttentionOperands {
  tilemega::Bf16 const* qkv;
  tilemega::Bf16* key_cache;
  tilemega::Bf16* value_cache;
  tilemega::Bf16 const* cosine;
  tilemega::Bf16 const* sine;
  tilemega::Bf16 const* q_norm;
  tilemega::Bf16 const* k_norm;
  tilemega::Bf16* context;
  float* partial;
  float* lse;
  int batch, heads_kv, capacity, past, block_extent;
  float epsilon;
  void const* key_tensor_map = nullptr;
  void const* value_tensor_map = nullptr;
#if TILEMEGA_TRACE_TASK
  ServingTaskProfile* profile = nullptr;
#endif
};

namespace mx {

/// Sum over one 32-lane logical warp, broadcast to all its lanes. Offsets
/// below 32 stay inside the half wave (P-5).
__device__ __forceinline__ float WarpSum32(float value) {
#pragma unroll
  for (int shift = 16; shift; shift >>= 1) value += __shfl_xor_sync(maca::kFullWaveMask, value, shift, 32);
  return value;
}

/// The same sum over 32 per-lane partials, in WarpSum32's xor-tree order
/// (lane 0's operand order; every lane's tree is the same pairing), for code
/// that must reproduce it without shuffles. `load(lane)` returns a partial.
template <class Load>
__device__ __forceinline__ float TreeSum32(Load load) {
  float part[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) part[i] = load(i) + load(i + 16);
#pragma unroll
  for (int shift = 8; shift; shift >>= 1)
#pragma unroll
    for (int i = 0; i < shift; ++i) part[i] = part[i] + part[i + shift];
  return part[0];
}

/// The eight-element pieces of RotateRow. The MX-4 Qkv epilogue
/// (Backend/Mx/DecodeGemm.h) calls these too, so both round identically.
__device__ __forceinline__ float SquareEight(tilemega::Bf16 const (&a)[8]) {
  float square = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) square += float(a[i]) * float(a[i]);
  return square;
}
/// `a` holds dimensions [d, d+8), `b` their rotate-half partners; `first`
/// is d < D/2. The result replaces `a`.
template <bool QkNorm>
__device__ __forceinline__ void RotateEight(tilemega::Bf16 (&a)[8], tilemega::Bf16 (&b)[8],
                                            tilemega::Bf16 const (&w)[8], tilemega::Bf16 const (&partner_w)[8],
                                            tilemega::Bf16 const (&cos)[8], tilemega::Bf16 const (&sin)[8],
                                            float inverse, bool first) {
  using Element = tilemega::Bf16;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if constexpr (QkNorm) {
      a[i] = Element(float(Element(float(a[i]) * inverse)) * float(w[i]));
      b[i] = Element(float(Element(float(b[i]) * inverse)) * float(partner_w[i]));
    }
    auto first_product = Element(float(a[i]) * float(cos[i]));
    auto second_product = Element((first ? -float(b[i]) : float(b[i])) * float(sin[i]));
    a[i] = Element(float(first_product) + float(second_product));
  }
}

/// One head row of D elements, eight per lane of a 32-lane logical warp:
/// optional RMSNorm with `norm`, then rotate-half RoPE. `store(d, uint4)`
/// receives the eight results for dimensions [d, d+8).
template <int D, bool QkNorm, class Store>
__device__ void RotateRow(tilemega::Bf16 const* x, tilemega::Bf16 const* norm, tilemega::Bf16 const* cosine,
                          tilemega::Bf16 const* sine, float epsilon, Store store) {
  using Element = tilemega::Bf16;
  int const lane = ComputeThread() & 31, d = lane * 8;
  alignas(16) Element a[8], b[8], w[8], partner_w[8], cos[8], sin[8];
  float square = 0;
  if (d < D) {
    *reinterpret_cast<uint4*>(a) = backend::LoadGlobal16(x + d);
    int const partner = (d + D / 2) % D;
    *reinterpret_cast<uint4*>(b) = backend::LoadGlobal16(x + partner);
    *reinterpret_cast<uint4*>(cos) = backend::LoadGlobal16(cosine + d);
    *reinterpret_cast<uint4*>(sin) = backend::LoadGlobal16(sine + d);
    if constexpr (QkNorm) {
      *reinterpret_cast<uint4*>(w) = backend::LoadGlobal16(norm + d);
      *reinterpret_cast<uint4*>(partner_w) = backend::LoadGlobal16(norm + partner);
      square = SquareEight(a);
    }
  }
  float inverse = 1;
  if constexpr (QkNorm) inverse = maca::Rsqrt(WarpSum32(square) / D + epsilon);
  if (d < D) {
    RotateEight<QkNorm>(a, b, w, partner_w, cos, sin, inverse, d < D / 2);
    store(d, *reinterpret_cast<uint4 const*>(a));
  }
}

inline constexpr float kLog2e = 1.4426950408889634f;

}  // namespace mx
}  // namespace tilemega::codegen
