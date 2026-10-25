// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <cutlass/bfloat16.h>
#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#if TILEMEGA_TRACE_NATIVE_PHASES || TILEMEGA_TRACE_NATIVE_BOUNDARY
#error "native shared-operand ABI does not support per-body phase or boundary diagnostics"
#endif
namespace tilemega::backend {
struct Sm90DenseOperands {
  // A is read only through its tensor map (prompt §5 D3).
  CUtensorMap const* activation;
  cutlass::bfloat16_t const *weight,*residual;
  cutlass::bfloat16_t* output;
  int* argmax_index;
  float epsilon;
};
struct Sm90DenseAttentionOperands {
  // History is read by TMA through these maps; the pointers write row `past`.
  CUtensorMap const *key_map,*value_map;
  cutlass::bfloat16_t const* qkv;
  cutlass::bfloat16_t *key,*value,*output;
  cutlass::bfloat16_t const *cosine,*sine,*q_norm,*k_norm;
  float *partial,*lse;
  std::uint64_t* arrivals;
  int past;
  float epsilon;
};
// Thread zero fully initializes the active member for the next stage. That
// member is immutable from the V3 CTA join until the stage's complete return.
// The other member is not read by the stage or its preissue tail.
struct alignas(16) Sm90DenseOperandBank {
  alignas(16) Sm90DenseOperands gemm;
  alignas(16) Sm90DenseAttentionOperands attention;
};
struct Sm90DenseOperandHeader {
  static constexpr int kOffset=2*sizeof(Sm90DenseRingState);
  static constexpr int kEnd=kOffset+2*sizeof(Sm90DenseOperandBank);
  static_assert(sizeof(void*)==8 && sizeof(Sm90DenseOperands)==48);
  static_assert(sizeof(float)==4 && offsetof(Sm90DenseOperands,epsilon)==40);
  static_assert(sizeof(Sm90DenseAttentionOperands)==112);
  static_assert(sizeof(Sm90DenseOperandBank)==160 && alignof(Sm90DenseOperandBank)==16);
  static_assert(offsetof(Sm90DenseOperandBank,gemm)==0 &&
      offsetof(Sm90DenseOperandBank,attention)==48);
  static_assert(sizeof(Sm90DenseRingState)==256 && alignof(Sm90DenseRingState)==16);
  static_assert(kOffset==512 && kOffset%alignof(Sm90DenseOperandBank)==0);
  static_assert(kEnd==832 && kEnd<=1024);
  static_assert(std::is_standard_layout_v<Sm90DenseOperandBank> &&
      std::is_trivially_copyable_v<Sm90DenseOperandBank>);
  __device__ static __forceinline__ Sm90DenseOperandBank& Bank(char* arena,unsigned stage) {
    return reinterpret_cast<Sm90DenseOperandBank*>(arena+kOffset)[stage%2];
  }
};
}  // namespace tilemega::backend
