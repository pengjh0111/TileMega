// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeAttentionV2.h>
#include <cuda_runtime.h>
#ifndef MX5_TEST_DIM
#define MX5_TEST_DIM 128
#endif
#ifndef MX5_TEST_Q
#define MX5_TEST_Q 8
#endif
#ifndef MX5_TEST_THREADS
#define MX5_TEST_THREADS 576
#endif
using namespace tilemega::backend::mx;
struct AttentionArgs { MxAttentionOperandsV2 ops; unsigned* calls; int b, hkv; };
template <class Body>
__global__ void __launch_bounds__(MX5_TEST_THREADS) AttentionKernel(AttentionArgs args) {
  extern __shared__ __align__(16) char smem[];
  MxAttentionShapeV2<MX5_TEST_THREADS> const shape{args.b, args.hkv};
  auto pf = Body::Prefetch(args.ops, shape);
  Body::Run(pf, args.ops, shape, smem, [&] {
    if (threadIdx.x % 64 == 0) atomicAdd(args.calls + blockIdx.x * 8 + threadIdx.x / 64, 1u);
    return MxNoPrefetchedV2{};
  });
}
template <int Bmin, bool Norm, int Kind>
int Launch(AttentionArgs const& args, cudaStream_t stream) {
  using Body = MxAttentionV2<MX5_TEST_DIM, MX5_TEST_Q, Bmin, Norm, Kind>;
  int const bytes = max(32784, Body::kSharedBytes);
  if (cudaFuncSetAttribute(AttentionKernel<Body>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess) return 2;
  AttentionKernel<Body><<<104, MX5_TEST_THREADS, bytes, stream>>>(args);
  return int(cudaGetLastError());
}
template <int Bmin, bool Norm>
int Kind(AttentionArgs const& a, int kind, cudaStream_t stream) {
  if (kind == 0) return Launch<Bmin, Norm, 0>(a, stream);
  if (kind == 1) return Launch<Bmin, Norm, 1>(a, stream);
  if (kind == 2) return Launch<Bmin, Norm, 2>(a, stream);
  return 3;
}
template <int Bmin>
int Norm(AttentionArgs const& a, int norm, int kind, cudaStream_t stream) {
  if constexpr (MX5_TEST_DIM == 128) if (norm) return Kind<Bmin, true>(a, kind, stream);
  return Kind<Bmin, false>(a, kind, stream);
}
extern "C" int mx5_attention(AttentionArgs const* args, int bmin, int norm, int kind, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  if (bmin == 1) return Norm<1>(*args, norm, kind, s);
  if (bmin == 8) return Norm<8>(*args, norm, kind, s);
  return 4;
}
