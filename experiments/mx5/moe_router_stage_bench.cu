// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/RouterGemmV2.h>
#include <cuda_runtime.h>

#ifndef MX5_TEST_THREADS
#define MX5_TEST_THREADS 576
#endif
#ifndef MX5_TEST_TN
#define MX5_TEST_TN 32
#endif
#ifndef MX5_TEST_DEPTH
#define MX5_TEST_DEPTH 1
#endif
#ifndef MX5_BENCH_PW
#define MX5_BENCH_PW MX5_TEST_DEPTH
#endif
#ifndef MX5_MOE_STREAM_K
#define MX5_MOE_STREAM_K 1
#endif
using namespace tilemega::backend::mx;
struct RouterBenchArgs {
  MxMoeRouterOperandsV2 router;
  unsigned long long *counter, *stamps;
  int* resources;
  int n, k, b, iterations, copies;
};

template<int B>
using RouterBody = MxRouterGemmV2<MX5_TEST_TN, MX5_TEST_DEPTH,
    MX5_MOE_STREAM_K ? MxPart::kStreamK : MxPart::kTileRange,
    MxMoeDispatchV2<B,2048,768,64,64,true,MX5_TEST_THREADS>, MX5_BENCH_PW>;

template<int B>
__global__ void __launch_bounds__(MX5_TEST_THREADS) MoeRouterBench(RouterBenchArgs args) {
  unsigned const grid = gridDim.x;
  __builtin_assume(grid == 104);
  using Body = RouterBody<B>;
  using Sync = tilemega::codegen::mx::StageSyncV2<MX5_TEST_THREADS>;
  extern __shared__ __align__(16) char smem[];
  MxRuntimeShapeV2<MX5_TEST_THREADS> const shape{args.n, args.k, args.b};
  int const body_bytes = max(32784, MxGemmSmemV2(args.b, MX5_TEST_TN, 1).bytes + 16);
  auto* generation = reinterpret_cast<unsigned*>(smem + body_bytes);
  if (Sync::IsRole()) *generation = 0;
  Sync::CtaSync();
  auto pf = Body::Prefetch(args.router, shape);
#pragma unroll 1
  for (int i = 0; i < args.iterations; ++i) {
    auto current = args.router;
    current.w += std::size_t(i % args.copies) * args.n * args.k;
    pf = Body::Run(pf, current, shape, smem, [&] {
      auto following = args.router;
      following.w += std::size_t((i + 1) % args.copies) * args.n * args.k;
      return Body::Prefetch(following, shape);
    });
    Sync::Barrier(args.counter, (i + 1ull) * 104, generation, unsigned(i + 1), [] {});
    if (blockIdx.x == 0 && Sync::IsRole()) args.stamps[i] = tilemega::maca::GlobalTimerNs();
  }
}

template<int B>
int LaunchRouter(RouterBenchArgs const& args, cudaStream_t stream) {
  constexpr auto part = MX5_MOE_STREAM_K ? MxPart::kStreamK : MxPart::kTileRange;
  if (!MxGemmLegalV2(args.n, args.k, B, MX5_TEST_TN, 1, MX5_TEST_DEPTH, MxOpV2::kStore, part)) return 4;
  int const bytes = max(32784, MxGemmSmemV2(B, MX5_TEST_TN, 1).bytes + 16) + 16;
  auto kernel = MoeRouterBench<B>;
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess) return 5;
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, kernel) != cudaSuccess) return 6;
  int resident = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, MX5_TEST_THREADS, bytes) != cudaSuccess || resident != 1) return 7;
  if (args.resources) {
    args.resources[0] = a.numRegs; args.resources[1] = int(a.localSizeBytes);
    args.resources[2] = int(a.sharedSizeBytes); args.resources[3] = resident;
  }
  if (a.localSizeBytes || a.sharedSizeBytes || a.numRegs > (MX5_TEST_THREADS == 576 ? 227 : 255)) return 8;
  if (!args.iterations) return 0;
  kernel<<<104,MX5_TEST_THREADS,bytes,stream>>>(args);
  return int(cudaGetLastError());
}
extern "C" int mx5_moe_router_bench(RouterBenchArgs const* args, void* stream) {
  if (!args || args->n != 128 || args->k != 2048 || args->iterations < 0 || args->copies < 1) return 4;
#define CASE(B) case B: return LaunchRouter<B>(*args, static_cast<cudaStream_t>(stream))
  switch (args->b) { CASE(1); CASE(2); CASE(4); CASE(8); CASE(16); default: return 4; }
#undef CASE
}
extern "C" int mx5_moe_router_bench_args_size() { return sizeof(RouterBenchArgs); }
