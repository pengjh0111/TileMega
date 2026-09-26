// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime.h>

namespace tilemega::backend {
// Callers predicate the entire aligned 16-byte span. Explicit PTX keeps the
// BF16 conversion optimizer from scalarizing an otherwise vector C++ store.
__device__ __forceinline__ uint4 LoadGlobal16(void const* source) {
  uint4 value;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(value.x),"=r"(value.y),"=r"(value.z),"=r"(value.w) : "l"(source) : "memory");
  return value;
}
__device__ __forceinline__ void StoreGlobal16(void* target,uint4 value) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::
      "l"(target),"r"(value.x),"r"(value.y),"r"(value.z),"r"(value.w) : "memory");
}
}
