// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/GroupedGemmV2.h>
#include <cuda_runtime.h>
#include <type_traits>

#ifndef MX5_TEST_THREADS
#define MX5_TEST_THREADS 576
#endif
#ifndef MX5_TEST_TN
#define MX5_TEST_TN 64
#endif
#ifndef MX5_TEST_WN
#define MX5_TEST_WN 1
#endif
#ifndef MX5_TEST_DEPTH
#define MX5_TEST_DEPTH 1
#endif
#ifndef MX5_BENCH_PW
#define MX5_BENCH_PW 0
#endif
#ifndef MX5_MOE_GATE_UP
#define MX5_MOE_GATE_UP 1
#endif

using namespace tilemega;
using namespace tilemega::backend::mx;
using namespace tilemega::codegen;
using Rms = DmEpilogueStep<DmEpilogueKind::kDeferredRMSNorm, DmActivation::kRelu,
    DmGatePair::kSimpleGate, 1, DmRounding::kFP32, DmRounding::kBF16, DmWriteKind::kDense, 1>;
using Gate = DmEpilogueStep<DmEpilogueKind::kGatePair, DmActivation::kRelu,
    DmGatePair::kSwiGLU, 16, DmRounding::kBF16, DmRounding::kBF16, DmWriteKind::kDense, 1>;
using Completion = std::conditional_t<bool(MX5_MOE_GATE_UP), MxNoUnitCompletionV2, MxMoeDownCompletionV2<true>>;
using Body = MxGroupedGemm<MX5_TEST_TN, MX5_TEST_WN, MX5_TEST_DEPTH,
    MX5_MOE_GATE_UP ? MxOpV2::kSwiGLU : MxOpV2::kStore, bool(MX5_MOE_GATE_UP),
    MxMoeGateChainV2<Rms, Gate>, Completion, MX5_BENCH_PW>;

struct GroupedBenchArgs {
  MxGroupedGemmOperandsV2 grouped;
  unsigned long long *counter, *stamps;
  int* resources;
  int n, k, b, iterations, copies;
};

__global__ void __launch_bounds__(MX5_TEST_THREADS) MoeGroupedBench(GroupedBenchArgs args) {
  unsigned const grid = gridDim.x;
  __builtin_assume(grid == 104);
  using Sync = mx::StageSyncV2<MX5_TEST_THREADS>;
  extern __shared__ __align__(16) char smem[];
  MxRuntimeShapeV2<MX5_TEST_THREADS> const shape{args.n, args.k, args.b};
  int const body_bytes = max(32784, MxGemmSmemV2(args.b, Body::kTN, Body::kWN).bytes);
  auto* generation = reinterpret_cast<unsigned*>(smem + body_bytes);
  if (Sync::IsRole()) *generation = 0;
  Sync::CtaSync();
  auto pf = Body::Prefetch(args.grouped, shape);
#pragma unroll 1
  for (int i = 0; i < args.iterations; ++i) {
    auto current = args.grouped;
    current.w += std::size_t(i % args.copies) * 128 * args.n * args.k;
    pf = Body::Run(pf, current, shape, smem, [&] {
      auto following = args.grouped;
      following.w += std::size_t((i + 1) % args.copies) * 128 * args.n * args.k;
      return Body::Prefetch(following, shape);
    });
    Sync::Barrier(args.counter, (i + 1ull) * 104, generation, unsigned(i + 1), [] {});
    if (blockIdx.x == 0 && Sync::IsRole()) args.stamps[i] = maca::GlobalTimerNs();
  }
}

extern "C" int mx5_moe_grouped_bench(GroupedBenchArgs const* args, void* stream) {
  if (!args || args->iterations < 0 || args->copies < 1 || args->b < 1 || args->b > 16 ||
      args->n != (MX5_MOE_GATE_UP ? 1536 : 2048) || args->k != (MX5_MOE_GATE_UP ? 2048 : 768) ||
      args->grouped.expert_stride != unsigned(args->n * args->k)) return 4;
  unsigned const tiles = args->n / Body::kTN;
  if (!args->grouped.units || args->grouped.units % tiles || args->grouped.units > 128 * tiles ||
      Body::kWN > int((args->grouped.units + 103) / 104)) return 4;
  if (!MxGemmLegalV2(int(args->grouped.units) * Body::kTN, args->k, args->b,
      Body::kTN, Body::kWN, Body::kDepth, MX5_MOE_GATE_UP ? MxOpV2::kSwiGLU : MxOpV2::kStore,
      MxPart::kTileRange)) return 4;
  int const bytes = max(32784, MxGemmSmemV2(args->b, Body::kTN, Body::kWN).bytes) + 16;
  if (bytes > 65536) return 4;
  if (cudaFuncSetAttribute(MoeGroupedBench, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess) return 5;
  cudaFuncAttributes attributes{};
  if (cudaFuncGetAttributes(&attributes, MoeGroupedBench) != cudaSuccess) return 6;
  int resident = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, MoeGroupedBench, MX5_TEST_THREADS, bytes) != cudaSuccess || resident != 1) return 7;
  if (args->resources) {
    args->resources[0] = attributes.numRegs;
    args->resources[1] = int(attributes.localSizeBytes);
    args->resources[2] = int(attributes.sharedSizeBytes);
    args->resources[3] = resident;
  }
  if (attributes.localSizeBytes || attributes.sharedSizeBytes || attributes.numRegs > (MX5_TEST_THREADS == 576 ? 227 : 255)) return 8;
  if (!args->iterations) return 0;
  MoeGroupedBench<<<104, MX5_TEST_THREADS, bytes, static_cast<cudaStream_t>(stream)>>>(*args);
  return int(cudaGetLastError());
}
extern "C" int mx5_moe_grouped_bench_args_size() { return sizeof(GroupedBenchArgs); }
