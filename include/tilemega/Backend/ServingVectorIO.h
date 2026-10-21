// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime.h>

namespace tilemega::backend {
// Callers predicate the entire aligned 16-byte span; one 128-bit access each.
__device__ __forceinline__ uint4 LoadGlobal16(void const* source) {
  return *reinterpret_cast<uint4 const*>(source);
}
// The same load through the global address space. A pointer read out of the
// plan's parameter block is generic, so LoadGlobal16 compiles to a flat load
// (LDX on xcore1000) and the MetaX wait inserter waits on its out-of-order
// counter as well, which at a control-flow join means every load in flight
// (ARRIVE 4160 in the final machine IR, scripts/mx_mir_report.py). Only for
// pointers known to address device global memory.
__device__ __forceinline__ uint4 LoadGlobal16Gvm(void const* source) {
  using Global = __attribute__((address_space(1))) uint4 const;
  auto const* global = (Global*)source;
  uint4 value;
  value.x = global->x;
  value.y = global->y;
  value.z = global->z;
  value.w = global->w;
  return value;
}
/// Scalar and float4 loads through the global address space, for the same
/// reason as LoadGlobal16Gvm: the split-K combine and the serving epilogue
/// read partials, residuals and deferred-norm sums through generic pointers.
__device__ __forceinline__ float LoadFloatGvm(float const* source) {
  return *(__attribute__((address_space(1))) float const*)source;
}
__device__ __forceinline__ float4 LoadFloat4Gvm(float const* source) {
  typedef float Vector __attribute__((ext_vector_type(4)));
  Vector v = *(__attribute__((address_space(1))) Vector const*)source;
  return make_float4(v.x, v.y, v.z, v.w);
}
__device__ __forceinline__ void StoreGlobal16(void* target,uint4 value) {
  *reinterpret_cast<uint4*>(target)=value;
}
}
