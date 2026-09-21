// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.3. R8 BE-3/BE-4: cross-thread reductions for TaskBodies.
//
// Before R8 every reduction in a SIMT body was a loop on one lane while the
// other 127 threads waited at the next barrier. These are the replacements:
// a warp reduction on shuffles, and a CTA reduction that combines the warp
// results through shared memory. Both are FP32 internally whatever the
// operands' storage type, which is the dtype rule the round fixes for every
// rewritten body.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace tilemega::codegen {

struct SumOp {
  __device__ static float Identity() { return 0.0f; }
  __device__ static float Apply(float a, float b) { return a + b; }
};

struct MaxOp {
  __device__ static float Identity() { return -INFINITY; }
  __device__ static float Apply(float a, float b) { return fmaxf(a, b); }
};

/// Reduce `value` across one warp. Every lane returns the result, so callers
/// do not need a follow-up broadcast.
template <class Op>
__device__ inline float WarpReduce(float value) {
  unsigned const active = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value = Op::Apply(value, __shfl_xor_sync(active, value, offset));
  return value;
}

/// Reduce across a CTA of `Threads` threads using `scratch`, which needs one
/// float per warp. Every thread returns the result.
///
/// The second stage runs in warp 0 over `kWarps` lanes and is itself a
/// shuffle reduction, so the whole CTA reduction costs two shuffles' worth of
/// latency plus two barriers -- no lane ever walks the array.
template <class Op, int Threads>
__device__ inline float BlockReduce(float value, float* scratch) {
  static_assert(Threads % 32 == 0, "a CTA reduction expects whole warps");
  constexpr int kWarps = Threads / 32;
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const warp = static_cast<int>(threadIdx.x) >> 5;
  value = WarpReduce<Op>(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  // Warp 0 folds the per-warp results; the identity keeps the lanes past
  // kWarps from contributing.
  float folded = (threadIdx.x < kWarps) ? scratch[threadIdx.x] : Op::Identity();
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      folded = Op::Apply(folded, __shfl_xor_sync(0xffffffffu, folded, offset));
    if (lane == 0) scratch[0] = folded;
  }
  __syncthreads();
  float const result = scratch[0];
  __syncthreads();
  return result;
}

}  // namespace tilemega::codegen
