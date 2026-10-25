// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeHeadV2.h>
#include <cuda_runtime.h>

#ifndef MX5_TEST_THREADS
#define MX5_TEST_THREADS 576
#endif
#ifndef MX5_TEST_TN
#define MX5_TEST_TN 16
#endif
#ifndef MX5_TEST_DEPTH
#define MX5_TEST_DEPTH 1
#endif
#ifndef MX5_BENCH_PW
#define MX5_BENCH_PW MX5_TEST_DEPTH
#endif

using namespace tilemega::backend::mx;
struct HeadBenchArgs {
  MxHeadOperandsV2 head;
  unsigned long long *counter, *stamps;
  int* resources;
  int n, k, b, iterations, copies;
};

template <class Body>
__global__ void __launch_bounds__(MX5_TEST_THREADS) HeadBench(HeadBenchArgs args) {
  using Sync = tilemega::codegen::mx::StageSyncV2<MX5_TEST_THREADS>;
  extern __shared__ __align__(16) char smem[];
  MxRuntimeShapeV2<MX5_TEST_THREADS> const shape{args.n, args.k, args.b};
  auto* const generation = reinterpret_cast<unsigned*>(smem + max(32784, Body::SmemBytes(args.b)));
  if (Sync::IsRole()) *generation = 0;
  Sync::CtaSync();
  auto current = args.head;
  auto pf = Body::Prefetch(current, shape);
#pragma unroll 1
  for (int iteration = 0; iteration < args.iterations; ++iteration) {
    auto following = args.head;
    following.gemm.w += std::size_t((iteration + 1) % args.copies) * args.n * args.k;
    pf = Body::Run(pf, current, shape, smem, [&] { return Body::Prefetch(following, shape); });
    Sync::Barrier(args.counter, (iteration + 1ull) * gridDim.x, generation, unsigned(iteration + 1), [] {});
    if (blockIdx.x == 0 && Sync::IsRole()) args.stamps[iteration] = tilemega::maca::GlobalTimerNs();
    current = following;
  }
}

template <int WN>
int Launch(HeadBenchArgs const& args, cudaStream_t stream) {
  constexpr int tn = MX5_TEST_TN, depth = MX5_TEST_DEPTH;
  if constexpr (!MxDepthFits(tn, depth)) return 4;
  else {
    if (!MxGemmLegalV2(args.n, args.k, args.b, tn, WN, depth, MxOpV2::kStore, MxPart::kTileRange))
      return 4;
    using Body = MxHead<tn, WN, depth, MX5_BENCH_PW>;
    int const bytes = max(32784, Body::SmemBytes(args.b)) + 16;
    if (bytes > 65536) return 4;
    if (cudaFuncSetAttribute(HeadBench<Body>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess)
      return 5;
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, HeadBench<Body>) != cudaSuccess) return 6;
    int occupancy = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, HeadBench<Body>, MX5_TEST_THREADS, bytes)
        != cudaSuccess || occupancy != 1) return 7;
    if (args.resources) {
      args.resources[0] = attributes.numRegs;
      args.resources[1] = int(attributes.localSizeBytes);
      args.resources[2] = int(attributes.sharedSizeBytes);
      args.resources[3] = occupancy;
    }
    if (!args.iterations) return 0;
    HeadBench<Body><<<104, MX5_TEST_THREADS, bytes, stream>>>(args);
    return int(cudaGetLastError());
  }
}

extern "C" int mx5_head_bench(HeadBenchArgs const* args, int wn, void* stream) {
  if (!args || args->iterations < 0 || args->copies < 1 || args->head.past < 0 ||
      args->head.past + 1 >= args->head.capacity) return 4;
  auto s = static_cast<cudaStream_t>(stream);
  if (wn == 4) return Launch<4>(*args, s);
  if (wn == 8) return Launch<8>(*args, s);
  return 4;
}
extern "C" int mx5_head_bench_args_size() { return sizeof(HeadBenchArgs); }
extern "C" void mx5_head_bench_coordinates(int* values) {
  values[0] = MX5_TEST_TN;
  values[1] = MX5_TEST_DEPTH;
  values[2] = MX5_BENCH_PW;
  values[3] = MX5_TEST_THREADS;
}
