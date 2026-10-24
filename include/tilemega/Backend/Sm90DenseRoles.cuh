// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseRoleContract.h>
#include <cuda_runtime.h>
#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__<12 || \
    (__CUDACC_VER_MAJOR__==12 && __CUDACC_VER_MINOR__<8))
#error "native dense role facts require CUDA 12.8 or newer"
#endif
namespace tilemega::backend {
struct Sm90DenseRoles:Sm90DenseRoleAbi {
  // The native 160x1x1 launch and Run's <128/else dispatch prove these facts.
  // Calling an outlined role outside its range is outside the native dense role ABI.
  __device__ static __forceinline__ void Producer() {
    __builtin_assume(threadIdx.x>=kConsumers && threadIdx.x<kThreads);
  }
  __device__ static __forceinline__ void Consumer() {
    __builtin_assume(threadIdx.x<kConsumers);
  }
};
}  // namespace tilemega::backend
