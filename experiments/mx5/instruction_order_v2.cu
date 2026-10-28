// SPDX-License-Identifier: BSD-3-Clause
// Diagnostic only: one kernel and allocation for repeat and program order.
#include <tilemega/Backend/Mx/DecodeAttentionV2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace tilemega::backend::mx;
using tilemega::Bf16;
using Sync = tilemega::codegen::mx::StageSyncV2<512>;
constexpr int kCapacity = 1088, kShared = 32784 + 16;
constexpr std::size_t kRotateBytes = 256ull << 20;

struct Pools {
  Bf16 *weights, *a, *output, *qkv, *keys, *values, *norm, *cosine, *sine;
  float* partial;
  unsigned* tickets;
  unsigned long long *arrivals, *stamps;
};

void Check(cudaError_t code, char const* operation) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(code));
    std::exit(2);
  }
}
template <class T> void Allocate(T*& pointer, std::size_t elements) {
  Check(cudaMalloc(&pointer, elements * sizeof(T)), "allocate");
}
__global__ void Fill(Bf16* output, std::size_t count, unsigned seed, int constant) {
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < count; i += std::size_t(gridDim.x) * blockDim.x) {
    unsigned x = unsigned(i) * 2654435761u + seed;
    x ^= x >> 15;
    output[i] = Bf16(constant >= 0 ? float(constant) : float(int(x % 257) - 128) / 1024.f);
  }
}

template <int C>
__device__ __forceinline__ void Body(Pools const& p, int ordinal, char* smem) {
  auto next = [] { return MxNoPrefetchedV2{}; };
  if constexpr (C == 1) {
    using Attention = MxAttentionV2<128, 2, 1, true, 0>;
    constexpr std::size_t stride = std::size_t(8) * kCapacity * 128;
    constexpr int copies = int((kRotateBytes + 4 * stride - 1) / (4 * stride));
    auto* keys = p.keys + std::size_t(ordinal % copies) * stride;
    auto* values = p.values + std::size_t(ordinal % copies) * stride;
    MxAttentionOperandsV2 const ops{p.qkv, p.norm, p.norm, p.cosine, p.sine,
        keys, values, p.output + 32768, p.partial, p.tickets, kCapacity, 575, 1e-6f};
    MxStaticAttentionShapeV2<1, 8, 512> const shape{};
    auto pf = Attention::Prefetch(ops, shape);
    Attention::Run(pf, ops, shape, smem, next);
  } else {
    constexpr int n = C == 0 ? 4096 : C == 3 ? 6144 : 1024;
    constexpr int k = C == 0 || C == 3 ? 1024 : C == 2 ? 2048 : 3072;
    using Gemm = std::conditional_t<C == 0,
        MxGemm<32, 2, 2, MxOpV2::kStore, true, MxPart::kTileRange, 2>,
        std::conditional_t<C == 3,
          MxGemm<64, 1, 1, MxOpV2::kSwiGLU, true, MxPart::kTileRange, 1>,
          MxGemm<16, 1, 2, MxOpV2::kResidual, false, MxPart::kTileRange, 2>>>;
    constexpr std::size_t stride = std::size_t(n) * k;
    constexpr int copies = int((kRotateBytes + 2 * stride - 1) / (2 * stride));
    MxGemmOperandsV2 const ops{p.a, p.weights + std::size_t(ordinal % copies) * stride,
        p.a, p.output, p.partial, p.tickets, nullptr, nullptr, 0, 0, 1e-6f};
    MxStaticShapeV2<n, k, 1, 512> const shape{};
    auto pf = Gemm::Prefetch(ops, shape);
    Gemm::Run(pf, ops, shape, smem, next);
  }
}

__global__ void __launch_bounds__(512) InstructionOrder(Pools p, int count, int fixed) {
  unsigned const grid = gridDim.x;
  __builtin_assume(grid == 104);
  extern __shared__ __align__(16) char smem[];
  auto* generation = reinterpret_cast<unsigned*>(smem + 32784);
  if (Sync::IsRole()) *generation = 0;
  Sync::CtaSync();
#pragma unroll 1
  for (int i = 0; i < count; ++i) {
    int const stage = __builtin_mxc_readfirstlane(fixed < 0 ? i % 5 : fixed);
    int const ordinal = __builtin_mxc_readfirstlane(fixed < 0 ? i / 5 : i);
    switch (stage) {
      case 0: Body<0>(p, ordinal, smem); break;
      case 1: Body<1>(p, ordinal, smem); break;
      case 2: Body<2>(p, ordinal, smem); break;
      case 3: Body<3>(p, ordinal, smem); break;
      case 4: Body<4>(p, ordinal, smem); break;
    }
    Sync::Barrier(p.arrivals + (i % 64) * 8, 104ull * (i / 64 + 1), generation,
                  unsigned(i + 1), [] {});
    if (blockIdx.x == 0 && Sync::IsRole()) p.stamps[i] = tilemega::maca::GlobalTimerNs();
  }
}

template <int C>
__global__ void __launch_bounds__(512) InstructionSize(Pools p, int ordinal) {
  unsigned const grid = gridDim.x;
  __builtin_assume(grid == 104);
  extern __shared__ __align__(16) char smem[];
  Body<C>(p, ordinal, smem);
}

template <class Kernel> void Resource(char const* name, Kernel kernel) {
  Check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared), "set smem");
  cudaFuncAttributes attr{};
  Check(cudaFuncGetAttributes(&attr, kernel), "attributes");
  int resident = 0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, 512, kShared), "occupancy");
  std::printf("RESOURCE,%s,%d,%zu,%zu,%d\n", name, attr.numRegs, attr.localSizeBytes,
              attr.sharedSizeBytes, resident);
  if (attr.numRegs > 255 || attr.localSizeBytes || attr.sharedSizeBytes || resident != 1) std::exit(3);
}
int main(int argc, char** argv) {
  if (argc != 3) return 4;
  int const iterations = std::atoi(argv[1]);
  if (iterations < 0 || (iterations > 0 && iterations < 32)) return 4;
  Resource("program", InstructionOrder);
  Resource("qkv", InstructionSize<0>);
  Resource("attention", InstructionSize<1>);
  Resource("o", InstructionSize<2>);
  Resource("gate_up", InstructionSize<3>);
  Resource("down", InstructionSize<4>);
  if (!iterations) return 0;
  Pools p{};
  Allocate(p.weights, (kRotateBytes + (24ull << 20)) / 2);
  Allocate(p.a, 32768); Allocate(p.output, 65536); Allocate(p.qkv, 4096);
  constexpr std::size_t stride = std::size_t(8) * kCapacity * 128;
  constexpr int copies = int((kRotateBytes + 4 * stride - 1) / (4 * stride));
  Allocate(p.keys, stride * copies); Allocate(p.values, stride * copies);
  Allocate(p.norm, 128); Allocate(p.cosine, kCapacity * 128); Allocate(p.sine, kCapacity * 128);
  Allocate(p.partial, std::size_t(16) << 20); Allocate(p.tickets, 16384);
  Allocate(p.arrivals, 64 * 8); Allocate(p.stamps, std::size_t(iterations) * 5);
  Fill<<<104, 256>>>(p.weights, (kRotateBytes + (24ull << 20)) / 2, 7, -1);
  Fill<<<104, 256>>>(p.a, 32768, 11, -1); Fill<<<104, 256>>>(p.qkv, 4096, 13, -1);
  Fill<<<104, 256>>>(p.keys, stride * copies, 17, -1);
  Fill<<<104, 256>>>(p.values, stride * copies, 19, -1);
  Fill<<<1, 128>>>(p.norm, 128, 0, 1);
  Fill<<<104, 256>>>(p.cosine, kCapacity * 128, 0, 1);
  Fill<<<104, 256>>>(p.sine, kCapacity * 128, 0, 0);
  Check(cudaDeviceSynchronize(), "initialize");
  FILE* out = std::fopen(argv[2], "wx"); if (!out) return 5;
  std::fprintf(out, "probe,model,batch,class,mode,median_ns,p90_ns,iterations\n");
  char const* names[] = {"qkv", "attention", "o", "gate_up", "down"};
  for (int fixed : {0, 1, 2, 3, 4, -1}) {
    int const count = fixed < 0 ? iterations * 5 : iterations;
    for (int pass = 0; pass < 2; ++pass) {
      Check(cudaMemset(p.arrivals, 0, 64 * 64), "reset counters");
      Check(cudaMemset(p.tickets, 0, 16384 * sizeof(unsigned)), "reset tickets");
      InstructionOrder<<<104, 512, kShared>>>(p, count, fixed);
      Check(cudaGetLastError(), "I1 launch"); Check(cudaDeviceSynchronize(), "I1 complete");
    }
    std::vector<unsigned long long> stamps(count);
    Check(cudaMemcpy(stamps.data(), p.stamps, count * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "stamps");
    for (int c = 0; c < 5; ++c) {
      if (fixed >= 0 && c != fixed) continue;
      std::vector<unsigned long long> durations;
      for (int i = 10; i < count; ++i)
        if (fixed >= 0 || i % 5 == c) durations.push_back(stamps[i] - stamps[i - 1]);
      std::sort(durations.begin(), durations.end());
      std::fprintf(out, "I1_v2,qwen3_0_6b,1,%s,%s,%llu,%llu,%d\n", names[c],
                   fixed < 0 ? "program_order" : "repeat", durations[durations.size()/2],
                   durations[durations.size()*9/10], iterations);
    }
  }
  std::fclose(out);
}
