// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime.h>
namespace tilemega::codegen::executor {
inline constexpr int kComputeThreads=128;
inline constexpr int kLoaderThreads=32;
// Compute warps are the first four warps in both the 128-thread prefill and
// 160-thread paged decode launches. The loader is always the final warp.
__device__ inline int ComputeThread() { return static_cast<int>(threadIdx.x); }
__device__ inline bool IsCompute() { return ComputeThread()<kComputeThreads; }
__device__ inline int LoaderLane() { return static_cast<int>(threadIdx.x)-kComputeThreads; }
__device__ inline void ComputeSync() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}
} // namespace tilemega::codegen::executor
