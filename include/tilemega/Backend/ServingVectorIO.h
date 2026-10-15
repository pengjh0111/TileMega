// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime.h>

namespace tilemega::backend {
// Callers predicate the entire aligned 16-byte span; one 128-bit access each.
__device__ __forceinline__ uint4 LoadGlobal16(void const* source) {
  return *reinterpret_cast<uint4 const*>(source);
}
__device__ __forceinline__ void StoreGlobal16(void* target,uint4 value) {
  *reinterpret_cast<uint4*>(target)=value;
}
}
