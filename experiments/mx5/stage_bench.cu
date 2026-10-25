// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeGemmV2.h>
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
struct BenchArgs {
  MxGemmOperandsV2 gemm;
  unsigned long long *counter, *stamps;
  int* resources;
  int n, k, b, iterations, copies;
};

template <class Body>
__global__ void __launch_bounds__(MX5_TEST_THREADS) GemmBench(BenchArgs args) {
  using Sync = tilemega::codegen::mx::StageSyncV2<MX5_TEST_THREADS>;
  extern __shared__ __align__(16) char smem[];
  MxRuntimeShapeV2<MX5_TEST_THREADS> const shape{args.n, args.k, args.b};
  int const body_bytes = max(32784, MxGemmSmemV2(args.b, Body::kTN, Body::kWN).bytes);
  auto* const generation = reinterpret_cast<unsigned*>(smem + body_bytes);
  if (Sync::IsRole()) *generation = 0;
  Sync::CtaSync();
  // Dense benchmark rows have no token indirection.
  args.gemm.a_tokens = nullptr;
  args.gemm.residual_tokens = nullptr;
  MxGemmOperandsV2 current = args.gemm;
  auto pf = Body::Prefetch(current, shape);
#pragma unroll 1
  for (int iteration = 0; iteration < args.iterations; ++iteration) {
    auto following = args.gemm;
    following.w += std::size_t((iteration + 1) % args.copies) * args.n * args.k;
    pf = Body::Run(pf, current, shape, smem, [&] { return Body::Prefetch(following, shape); });
    Sync::Barrier(args.counter, (iteration + 1ull) * gridDim.x, generation, unsigned(iteration + 1), [] {});
    if (blockIdx.x == 0 && Sync::IsRole()) args.stamps[iteration] = tilemega::maca::GlobalTimerNs();
    current = following;
  }
}

template <int WN, MxOpV2 Op, bool Norm, MxPart Part>
int Launch(BenchArgs const& args, cudaStream_t stream) {
  constexpr int tn = MX5_TEST_TN, d = MX5_TEST_DEPTH;
  if constexpr (!MxDepthFits(tn, d) || (Op == MxOpV2::kSwiGLU && tn < 32) ||
                (Part == MxPart::kStreamK && WN != 1)) return 4;
  else {
    if (!MxGemmLegalV2(args.n, args.k, args.b, tn, WN, d, Op, Part)) return 4;
    using Body = MxGemm<tn, WN, d, Op, Norm, Part, MX5_BENCH_PW>;
    int const bytes = max(32784, MxGemmSmemV2(args.b, tn, WN).bytes) + 16;
    if (bytes > 65536) return 4;
    if (cudaFuncSetAttribute(GemmBench<Body>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess)
      return 5;
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, GemmBench<Body>) != cudaSuccess) return 6;
    int occupancy = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, GemmBench<Body>, MX5_TEST_THREADS, bytes)
        != cudaSuccess || occupancy != 1) return 7;
    if (args.resources) {
      args.resources[0] = attributes.numRegs;
      args.resources[1] = int(attributes.localSizeBytes);
      args.resources[2] = int(attributes.sharedSizeBytes);
      args.resources[3] = occupancy;
    }
    if (!args.iterations) return 0;
    GemmBench<Body><<<104, MX5_TEST_THREADS, bytes, stream>>>(args);
    return int(cudaGetLastError());
  }
}
template <int WN, MxOpV2 Op>
int NormPart(BenchArgs const& a, int norm, int part, cudaStream_t stream) {
  if (norm) {
    if (part) return Launch<WN, Op, true, MxPart::kTileRange>(a, stream);
    return Launch<WN, Op, true, MxPart::kStreamK>(a, stream);
  }
  if (part) return Launch<WN, Op, false, MxPart::kTileRange>(a, stream);
  return Launch<WN, Op, false, MxPart::kStreamK>(a, stream);
}
template <int WN>
int Op(BenchArgs const& a, int op, int norm, int part, cudaStream_t stream) {
  if (op == 0) return NormPart<WN, MxOpV2::kStore>(a, norm, part, stream);
  if (op == 1) return NormPart<WN, MxOpV2::kResidual>(a, norm, part, stream);
  if (op == 2) return NormPart<WN, MxOpV2::kSwiGLU>(a, norm, part, stream);
  return 4;
}
extern "C" int mx5_bench(BenchArgs const* args, int wn, int op, int norm, int part, void* stream) {
  if (!args || args->iterations < 0 || args->copies < 1) return 4;
  auto s = static_cast<cudaStream_t>(stream);
  if (wn == 1) return Op<1>(*args, op, norm, part, s);
  if (wn == 2) return Op<2>(*args, op, norm, part, s);
  if (wn == 4) return Op<4>(*args, op, norm, part, s);
  if (wn == 8) return Op<8>(*args, op, norm, part, s);
  return 4;
}
extern "C" int mx5_bench_args_size() { return sizeof(BenchArgs); }
