// SPDX-License-Identifier: BSD-3-Clause
// MX-4 D1: which CTA shape streams decode weights best on xcore1000.
//
// The load form is the one MX-4's DecodeGemm uses (MX4 prompt §3.5): every
// wave issues 1 KiB loads (64 lanes x 16 B) through the global address space,
// in a straight-line ring of D loads in flight with compile-time slots and no
// branch around a load (addresses past the range are clamped). A CTA owns a
// contiguous range of 1 KiB units; its W waves interleave over it (wave w
// takes units w, w+W, ...), so together they read consecutive addresses.
// With `mma`, each load feeds two 16x16x16 MMAs as both operands.
//
//   stream  : one pass over a 256 MiB hash-filled buffer, R launches; GB/s.
//             `active` < grid leaves the other CTAs idle (104x512 only).
//   stage   : 1000 stages of 4 MiB each, rotating through the 256 MiB buffer,
//             each followed by the V3 grid barrier; ns per stage.
//
// Usage: shape_probe <stream|stage> <grid> <threads> <depth> <mma 0|1> [active]
// Prints one CSV row: mode,grid,threads,depth,mma,active,value,unit.
#include <tilemega/Codegen/tasks/EventSync.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

using tilemega::codegen::RedRelease;
using tilemega::codegen::WaitAtLeast;

namespace {
typedef float Accum __attribute__((ext_vector_type(4)));
typedef __fp16 Operand __attribute__((__vector_size__(4 * sizeof(__fp16))));
using Global = __attribute__((address_space(1))) unsigned const;

__device__ __forceinline__ uint4 Load16(unsigned char const* p) {
  auto const* g = (Global*)p;
  uint4 v;
  v.x = g[0]; v.y = g[1]; v.z = g[2]; v.w = g[3];
  return v;
}
__device__ __forceinline__ Operand Lo(uint4 v) {
  Operand r; unsigned b[2] = {v.x, v.y}; __builtin_memcpy(&r, b, 8); return r;
}
__device__ __forceinline__ Operand Hi(uint4 v) {
  Operand r; unsigned b[2] = {v.z, v.w}; __builtin_memcpy(&r, b, 8); return r;
}

// Units [begin, end) of 1 KiB; this wave takes begin+wave, +W, ...
template <int D, bool kMma>
__device__ __forceinline__ void Stream(unsigned char const* base, long begin, long end, int wave, int waves,
                                       int lane, Accum& acc, unsigned& fold) {
  long const count = end - begin <= wave ? 0 : (end - begin - wave + waves - 1) / waves;
  if (count == 0) return;  // wave-uniform, outside the loop
  long const last = count - 1;
  auto unit = [&](long i) {
    long const j = i < last ? i : last;  // clamp, no branch around the load
    return base + (begin + wave + j * waves) * 1024 + lane * 16;
  };
  uint4 ring[D];
  auto consume = [&](uint4 v) {
    if constexpr (kMma) {
      acc = __builtin_mxc_mma_16x16x16bf16(Lo(v), Hi(v), acc);
      acc = __builtin_mxc_mma_16x16x16bf16(Hi(v), Lo(v), acc);
    } else {
      fold ^= v.x ^ v.y ^ v.z ^ v.w;
      asm volatile("" : "+r"(fold));  // keep the fold at its slot (no sinking)
    }
  };
  // Issue the prologue in slot order: the wait counts below assume it.
  [&]<std::size_t... k>(std::index_sequence<k...>) {
    (([&] {
       ring[k] = Load16(unit(long(k)));
       asm volatile("" ::: "memory");
     }()),
     ...);
  }(std::make_index_sequence<D>{});
  long const groups = (count + D - 1) / D;
#pragma unroll 1
  for (long g = 0; g < groups; ++g) {
    [&]<std::size_t... k>(std::index_sequence<k...>) {
      (([&] {
         // Consume the slot before refilling it: the slot's old and new
         // values must not be live together, or the allocator copies the
         // pending load at the back edge and the inserter drains there.
         __builtin_mxc_arrive_gvmcnt(D - 1);
         consume(ring[k]);
         ring[k] = Load16(unit((g + 1) * D + long(k)));
       }()),
       ...);
    }(std::make_index_sequence<D>{});
  }
  __builtin_mxc_arrive_gvmcnt(0);
}

template <int D, bool kMma>
__global__ void StreamKernel(unsigned char const* data, long units, int active, unsigned* sink) {
  int const waves = blockDim.x / 64, wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  Accum acc = {0, 0, 0, 0};
  unsigned fold = 0;
  if (int(blockIdx.x) < active) {
    long const begin = units * blockIdx.x / active, end = units * (blockIdx.x + 1) / active;
    Stream<D, kMma>(data, begin, end, wave, waves, lane, acc, fold);
  }
  if ((fold ^ unsigned(acc.x) ^ unsigned(acc.w)) == 0x9e3779b9u) sink[blockIdx.x] = fold;
}

template <int D, bool kMma>
__global__ void StageKernel(unsigned char const* data, long units_total, long stage_units, int stages,
                            unsigned long long* arrivals, unsigned* sink) {
  int const waves = blockDim.x / 64, wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  Accum acc = {0, 0, 0, 0};
  unsigned fold = 0;
  long const regions = units_total / stage_units;
  for (int s = 0; s < stages; ++s) {
    unsigned char const* base = data + (s % regions) * stage_units * 1024;
    long const begin = stage_units * blockIdx.x / gridDim.x, end = stage_units * (blockIdx.x + 1) / gridDim.x;
    Stream<D, kMma>(base, begin, end, wave, waves, lane, acc, fold);
    __syncthreads();
    if (threadIdx.x == 0) {
      RedRelease(&arrivals[s % 64], 1ull);
      WaitAtLeast(&arrivals[s % 64], static_cast<unsigned long long>(gridDim.x) * (s / 64 + 1));
    }
    __syncthreads();
  }
  if ((fold ^ unsigned(acc.x) ^ unsigned(acc.w)) == 0x9e3779b9u) sink[blockIdx.x] = fold;
}

__global__ void Fill(unsigned* p, long n) {
  for (long i = blockIdx.x * long(blockDim.x) + threadIdx.x; i < n; i += long(gridDim.x) * blockDim.x) {
    unsigned x = unsigned(i) * 2654435761u + 0x9e3779b9u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    p[i] = x;
  }
}

template <class K>
float Time(K launch, int reps) {
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  std::vector<float> v;
  for (int r = 0; r < reps + 1; ++r) {
    cudaEventRecord(a); launch(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    if (r) v.push_back(ms);
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <int D, bool kMma>
int Run(char const* mode, int grid, int threads, int active) {
  long const bytes = 256l << 20, units = bytes / 1024;
  unsigned char* data; unsigned* sink; unsigned long long* arrivals;
  cudaMalloc(&data, bytes + 4096); cudaMalloc(&sink, grid * 4); cudaMalloc(&arrivals, 64 * 8);
  Fill<<<1024, 256>>>(reinterpret_cast<unsigned*>(data), (bytes + 4096) / 4);
  cudaMemset(arrivals, 0, 64 * 8);
  // Dynamic shared memory floor(65536/(G/104+1)) + 16 B, as the MX-4 program
  // asks (§3.1): an AP then holds at most G/104 CTAs. Checked below.
  int const per_ap = grid / 104, smem = 65536 / (per_ap + 1) + 16;
  cudaFuncSetAttribute(StreamKernel<D, kMma>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(StageKernel<D, kMma>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  float value;
  char const* unit;
  auto occupancy = [&](auto kernel) {
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, smem);
    if (blocks != per_ap) { std::printf("ERROR occupancy %d != %d\n", blocks, per_ap); std::exit(3); }
  };
  if (!std::strcmp(mode, "stream")) occupancy(StreamKernel<D, kMma>); else occupancy(StageKernel<D, kMma>);
  if (!std::strcmp(mode, "stream")) {
    float const ms = Time([&] { StreamKernel<D, kMma><<<grid, threads, smem>>>(data, units, active, sink); }, 20);
    value = bytes / (ms * 1e-3) / 1e9; unit = "GB/s";
  } else {
    int const stages = 1000;
    long const stage_units = (4l << 20) / 1024;
    // The arrivals are monotonic across launches: reset before each.
    float const ms = Time([&] {
      cudaMemsetAsync(arrivals, 0, 64 * 8);
      StageKernel<D, kMma><<<grid, threads, smem>>>(data, units, stage_units, stages, arrivals, sink);
    }, 5);
    value = ms * 1e6 / stages; unit = "ns/stage";
  }
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("ERROR %s\n", cudaGetErrorString(e)); return 1; }
  std::printf("%s,%d,%d,%d,%d,%d,%.1f,%s\n", mode, grid, threads, D, int(kMma), active, value, unit);
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) { std::fprintf(stderr, "usage: shape_probe <stream|stage> grid threads depth mma [active]\n"); return 2; }
  char const* mode = argv[1];
  int const grid = std::atoi(argv[2]), threads = std::atoi(argv[3]), depth = std::atoi(argv[4]);
  bool const mma = std::atoi(argv[5]) != 0;
  int const active = argc > 6 ? std::atoi(argv[6]) : grid;
#define TM_CASE(DD) \
  if (depth == DD) return mma ? Run<DD, true>(mode, grid, threads, active) : Run<DD, false>(mode, grid, threads, active);
  TM_CASE(1) TM_CASE(2) TM_CASE(4)
#undef TM_CASE
  return 2;
}
