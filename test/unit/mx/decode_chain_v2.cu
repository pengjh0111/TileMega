// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeGemmV2.h>
#include <cuda_runtime.h>
#ifndef MX5_TEST_THREADS
#define MX5_TEST_THREADS 576
#endif

using namespace tilemega;
using namespace tilemega::backend::mx;
struct ChainArgs {
  Bf16 const *x, *w0, *w1, *w2, *residual;
  Bf16 *y0, *y1, *y2;
  float* partial;
  unsigned *tickets, *calls;
  unsigned long long* counters;
  int batch, iterations;
};
__global__ void __launch_bounds__(MX5_TEST_THREADS) ChainKernel(ChainArgs args) {
  using Sync = tilemega::codegen::mx::StageSyncV2<MX5_TEST_THREADS>;
  using A = MxGemm<16, 1, 1, MxOpV2::kStore, true, MxPart::kTileRange, 4>;
  using B = MxGemm<32, 1, 2, MxOpV2::kSwiGLU, true, MxPart::kStreamK, 2>;
  using C = MxGemm<16, 8, 1, MxOpV2::kResidual, false, MxPart::kTileRange, 1>;
  extern __shared__ __align__(16) char smem[];
  auto* const generation = reinterpret_cast<unsigned*>(smem + 32784);
  if (Sync::IsRole()) *generation = 0;
  Sync::CtaSync();
  MxRuntimeShapeV2<MX5_TEST_THREADS> const s0{1664, 256, args.batch}, s1{1280, 1664, args.batch}, s2{13312, 640, args.batch};
  MxGemmOperandsV2 const a{args.x, args.w0, nullptr, args.y0, nullptr, nullptr, nullptr, nullptr, 256, 0, 1e-6f};
  MxGemmOperandsV2 const b{args.y0, args.w1, nullptr, args.y1, args.partial, args.tickets,
                         nullptr, nullptr, 1664, 0, 1e-6f};
  MxGemmOperandsV2 const c{args.y1, args.w2, args.residual, args.y2, nullptr, nullptr,
                         nullptr, nullptr, 640, 13312, 1e-6f};
  auto count = [&](int stage) {
    if (threadIdx.x % 64 == 0) atomicAdd(args.calls + stage * 104 * 8 + blockIdx.x * 8 + threadIdx.x / 64, 1u);
  };
  auto pf = A::Prefetch(a, s0);
#pragma unroll 1
  for (int iteration = 0; iteration < args.iterations; ++iteration) {
    auto pb = A::Run(pf, a, s0, smem, [&] { count(0); return B::Prefetch(b, s1); });
    Sync::Barrier(args.counters, (iteration + 1ull) * 104, generation, iteration * 3 + 1, [] {});
    auto pc = B::Run(pb, b, s1, smem, [&] { count(1); return C::Prefetch(c, s2); });
    Sync::Barrier(args.counters + 8, (iteration + 1ull) * 104, generation, iteration * 3 + 2, [] {});
    pf = C::Run(pc, c, s2, smem, [&] { count(2); return A::Prefetch(a, s0); });
    Sync::Barrier(args.counters + 16, (iteration + 1ull) * 104, generation, iteration * 3 + 3, [] {});
  }
}
extern "C" int mx5_chain(ChainArgs const* args, void* stream) {
  ChainKernel<<<104, MX5_TEST_THREADS, 32800, static_cast<cudaStream_t>(stream)>>>(*args);
  return int(cudaGetLastError());
}
