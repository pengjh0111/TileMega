// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime.h>
namespace tilemega::codegen::executor {
// 128 compute threads = two 64-lane waves; "warp" below means a 32-thread
// logical group (half a wave). The nonpaged executor launches exactly the
// compute group, so its barrier is the CTA barrier (P-6: ~20 cycles).
inline constexpr int kComputeThreads=128;
inline constexpr int kLoaderThreads=64;
__device__ inline int ComputeThread() { return static_cast<int>(threadIdx.x); }
__device__ inline bool IsCompute() { return ComputeThread()<kComputeThreads; }
__device__ inline int LoaderLane() { return static_cast<int>(threadIdx.x)-kComputeThreads; }
__device__ inline void ComputeSync() { __syncthreads(); }
} // namespace tilemega::codegen::executor
