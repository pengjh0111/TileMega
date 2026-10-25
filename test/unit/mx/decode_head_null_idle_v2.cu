// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeHeadV2.h>
#include <cuda_runtime.h>

#ifndef MX5_TEST_TN
#define MX5_TEST_TN 16
#endif
#ifndef MX5_TEST_THREADS
#define MX5_TEST_THREADS 576
#endif
using namespace tilemega;
using namespace tilemega::backend::mx;
struct HeadArgs {
  Bf16 const *a, *w;
  MxHeadPartialV2* partial;
  unsigned *ticket, *calls;
  int* tokens;
  int n, k, b, capacity, past;
};
template <class Body>
__global__ void __launch_bounds__(MX5_TEST_THREADS) HeadKernel(HeadArgs args) {
  extern __shared__ __align__(16) char smem[];
  MxRuntimeShapeV2<MX5_TEST_THREADS> const shape{args.n, args.k, args.b};
  MxGemmOperandsV2 gemm{};
  int const wave = int(threadIdx.x) / 64, wk = 8 / Body::Core::kWN;
  int const tiles = args.n / MX5_TEST_TN;
  int const first = int(blockIdx.x) * tiles / 104;
  int const last = (int(blockIdx.x) + 1) * tiles / 104;
  bool const consumer = wave < 8 && first + wave / wk < last && args.k / 32 > wave % wk;
  gemm.a = args.a; gemm.w = consumer ? args.w : nullptr;
  gemm.a_stride = args.k; gemm.epsilon = 1e-6f;
  MxHeadOperandsV2 const ops{gemm, args.partial, args.ticket, args.tokens, args.capacity, args.past};
  auto pf = Body::Prefetch(ops, shape);
  Body::Run(pf, ops, shape, smem, [&] {
    if (threadIdx.x % 64 == 0) atomicAdd(args.calls + blockIdx.x * 8 + threadIdx.x / 64, 1u);
    return MxNoPrefetchedV2{};
  });
}
template <int WN, int D>
int Launch(HeadArgs const& args, cudaStream_t stream) {
  if constexpr (!MxDepthFits(MX5_TEST_TN, D)) return 4;
  else {
    using Body = MxHead<MX5_TEST_TN, WN, D>;
    int const bytes = max(32784, Body::SmemBytes(args.b));
    if (bytes > 65536) return 2;
    if (cudaFuncSetAttribute(HeadKernel<Body>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess) return 3;
    HeadKernel<Body><<<104, MX5_TEST_THREADS, bytes, stream>>>(args);
    return int(cudaGetLastError());
  }
}
extern "C" int mx5_head(HeadArgs const* args, int wn, int d, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  if (wn == 4 && d == 1) return Launch<4, 1>(*args, s);
  if (wn == 4 && d == 2) return Launch<4, 2>(*args, s);
  if (wn == 4 && d == 3) return Launch<4, 3>(*args, s);
  if (wn == 4 && d == 4) return Launch<4, 4>(*args, s);
  if (wn == 8 && d == 1) return Launch<8, 1>(*args, s);
  if (wn == 8 && d == 2) return Launch<8, 2>(*args, s);
  if (wn == 8 && d == 3) return Launch<8, 3>(*args, s);
  if (wn == 8 && d == 4) return Launch<8, 4>(*args, s);
  return 4;
}
