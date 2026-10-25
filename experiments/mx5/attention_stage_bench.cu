// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeAttentionV2.h>
#include <cuda_runtime.h>

#ifndef MX5_TEST_THREADS
#define MX5_TEST_THREADS 576
#endif
#ifndef MX5_TEST_DIM
#define MX5_TEST_DIM 128
#endif
#ifndef MX5_TEST_Q
#define MX5_TEST_Q 8
#endif
#ifndef MX5_BENCH_PREFETCH_KIND
#define MX5_BENCH_PREFETCH_KIND 2
#endif

using namespace tilemega::backend::mx;
struct AttentionBenchArgs {
  MxAttentionOperandsV2 attention;
  unsigned long long *counter, *stamps;
  int* resources;
  int b, hkv, iterations, copies;
};

template <class Body>
__global__ void __launch_bounds__(MX5_TEST_THREADS) AttentionBench(AttentionBenchArgs args) {
  using Sync = tilemega::codegen::mx::StageSyncV2<MX5_TEST_THREADS>;
  extern __shared__ __align__(16) char smem[];
  MxAttentionShapeV2<MX5_TEST_THREADS> const shape{args.b, args.hkv};
  auto* const generation = reinterpret_cast<unsigned*>(smem + max(32784, Body::kSharedBytes));
  if (Sync::IsRole()) *generation = 0;
  Sync::CtaSync();
  auto current = args.attention;
  auto pf = Body::Prefetch(current, shape);
#pragma unroll 1
  for (int iteration = 0; iteration < args.iterations; ++iteration) {
    auto following = args.attention;
    std::size_t const offset = std::size_t((iteration + 1) % args.copies) * args.b * args.hkv *
                               args.attention.capacity * MX5_TEST_DIM;
    following.k_cache += offset;
    following.v_cache += offset;
    pf = Body::Run(pf, current, shape, smem, [&] { return Body::Prefetch(following, shape); });
    Sync::Barrier(args.counter, (iteration + 1ull) * gridDim.x, generation, unsigned(iteration + 1), [] {});
    if (blockIdx.x == 0 && Sync::IsRole()) args.stamps[iteration] = tilemega::maca::GlobalTimerNs();
    current = following;
  }
}

template <int Bmin, bool Norm>
int Launch(AttentionBenchArgs const& args, cudaStream_t stream) {
  using Body = MxAttentionV2<MX5_TEST_DIM, MX5_TEST_Q, Bmin, Norm, MX5_BENCH_PREFETCH_KIND>;
  int const bytes = max(32784, Body::kSharedBytes) + 16;
  if (bytes > 65536) return 4;
  if (cudaFuncSetAttribute(AttentionBench<Body>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess)
    return 5;
  cudaFuncAttributes attributes{};
  if (cudaFuncGetAttributes(&attributes, AttentionBench<Body>) != cudaSuccess) return 6;
  int occupancy = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, AttentionBench<Body>, MX5_TEST_THREADS, bytes)
      != cudaSuccess || occupancy != 1) return 7;
  if (args.resources) {
    args.resources[0] = attributes.numRegs;
    args.resources[1] = int(attributes.localSizeBytes);
    args.resources[2] = int(attributes.sharedSizeBytes);
    args.resources[3] = occupancy;
  }
  if (!args.iterations) return 0;
  AttentionBench<Body><<<104, MX5_TEST_THREADS, bytes, stream>>>(args);
  return int(cudaGetLastError());
}
template <int Bmin>
int Norm(AttentionBenchArgs const& args, int norm, cudaStream_t stream) {
  if constexpr (MX5_TEST_DIM == 128) if (norm) return Launch<Bmin, true>(args, stream);
  if (norm) return 4;
  return Launch<Bmin, false>(args, stream);
}

extern "C" int mx5_attention_bench(AttentionBenchArgs const* args, int bmin, int norm, void* stream) {
  if (!args || args->iterations < 0 || args->copies < 1 || args->b < 1 || args->b > 16 || args->hkv < 1 ||
      args->attention.capacity < 32 || args->attention.capacity > 1088 || args->attention.capacity % 32 ||
      args->attention.past < 0 || args->attention.past >= args->attention.capacity || (norm != 0 && norm != 1))
    return 4;
  auto s = static_cast<cudaStream_t>(stream);
  if (bmin == 1) return Norm<1>(*args, norm, s);
  if (bmin == 2) return Norm<2>(*args, norm, s);
  if (bmin == 4) return Norm<4>(*args, norm, s);
  if (bmin == 8) return Norm<8>(*args, norm, s);
  return 4;
}
extern "C" int mx5_attention_bench_args_size() { return sizeof(AttentionBenchArgs); }
extern "C" void mx5_attention_bench_coordinates(int* values) {
  values[0] = MX5_TEST_DIM;
  values[1] = MX5_TEST_Q;
  values[2] = MX5_BENCH_PREFETCH_KIND;
  values[3] = MX5_TEST_THREADS;
}
