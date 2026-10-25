// SPDX-License-Identifier: BSD-3-Clause
// MX-5 L1: the same register ring under W, K and V-transpose address maps.
#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

constexpr unsigned kBytes = 256u << 20;
constexpr int kGrid = 104, kThreads = 576, kSmem = 32784;
constexpr unsigned kHeadBytes = 128 * 1088 * 2;

#define CHECK(x) do { auto e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(2); } } while (0)

template <int Shape>
__device__ __forceinline__ uint4 Load(unsigned char const* data, unsigned i, int n) {
  unsigned const wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  unsigned const r = lane & 15, g = lane >> 4;
  unsigned const base = ((i * 997u + blockIdx.x * 8 + wave) * kHeadBytes) & (kBytes - 1);
  unsigned offset;
  if constexpr (Shape == 0) offset = (n * 64 + lane) * 16;
  if constexpr (Shape == 1) {
    unsigned const a = (n / 4) % 2, dc = n % 4;
    unsigned const token = (n / 8) * 32 + 8 * (r / 4) + 4 * a + r % 4;
    offset = (token * 128 + dc * 32 + 8 * g) * 2;
  }
  if constexpr (Shape == 2)
    offset = ((16 * (n % 8) + r) * 1088 + (n / 8) * 32 + 8 * g) * 2;
  return tilemega::backend::LoadGlobal16Gvm(data + ((base + offset) & (kBytes - 1)));
}

template <int Shape, int N, std::size_t... J>
__device__ __forceinline__ unsigned Ring(uint4 (&slots)[N], unsigned char const* input, unsigned i,
                                        std::index_sequence<J...>) {
  unsigned sum = 0;
  (([&] {
    __builtin_mxc_arrive_gvmcnt(N - 1);
    auto const v = slots[J];
    sum += v.x + v.y + v.z + v.w;
    asm volatile("" :: "r"(sum) : "memory");
    slots[J] = Load<Shape>(input, i, int(J));
  }()), ...);
  return sum;
}

template <int Shape, int N, bool Latency>
__global__ void __launch_bounds__(kThreads)
LoadProbe(unsigned char const* input, unsigned* sink, unsigned long long* times, int active, int iterations) {
  __syncthreads();
  if (blockIdx.x >= unsigned(active) || threadIdx.x >= 512) return;
  auto const begin = tilemega::maca::GlobalTimerNs();
  uint4 slots[N];
#pragma unroll
  for (int n = 0; n < N; ++n) slots[n] = Load<Shape>(input, 0, n);
  asm volatile("" ::: "memory");
  unsigned sum = 0;
  if constexpr (!Latency) {
#pragma unroll 1
    for (int i = 1; i < iterations; ++i) sum += Ring<Shape, N>(slots, input, unsigned(i), std::make_index_sequence<N>{});
  }
#pragma unroll
  for (int n = 0; n < N; ++n) sum += slots[n].x + slots[n].y + slots[n].z + slots[n].w;
  asm volatile("" :: "r"(sum) : "memory");
  auto const end = tilemega::maca::GlobalTimerNs();
  if (threadIdx.x % 64 == 0) {
    unsigned const slot = blockIdx.x * 8 + threadIdx.x / 64;
    sink[slot] = sum;
    times[slot] = end - begin;
  }
}

template <int Shape, int N>
void Run(unsigned char const* input, unsigned* sink, unsigned long long* times, int active, int iterations,
         FILE* output) {
  int resident = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, LoadProbe<Shape, N, false>, kThreads, kSmem));
  if (resident != 1) { std::fprintf(stderr, "resident=%d, expected 1\n", resident); std::exit(3); }
  LoadProbe<Shape, N, false><<<kGrid, kThreads, kSmem>>>(input, sink, times, active, iterations);
  CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
  LoadProbe<Shape, N, false><<<kGrid, kThreads, kSmem>>>(input, sink, times, active, iterations);
  CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
  std::vector<unsigned long long> host(active * 8);
  CHECK(cudaMemcpy(host.data(), times, host.size() * sizeof(host[0]), cudaMemcpyDeviceToHost));
  auto const longest = *std::max_element(host.begin(), host.end());
  double const bytes = double(active) * 512 * N * 16 * iterations;
  double const gbps = bytes / double(longest);
  std::vector<unsigned long long> latency;
  for (int repeat = 0; repeat < 32; ++repeat) {
    // Advance the cold starting point without changing the address geometry.
    unsigned const shift = (unsigned(repeat) * 1052672u) & (kBytes - 1);
    // Allocate two copies so the shifted address maps remain in bounds.
    LoadProbe<Shape, N, true><<<kGrid, kThreads, kSmem>>>(input + shift, sink, times, active, 1);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(host.data(), times, host.size() * sizeof(host[0]), cudaMemcpyDeviceToHost));
    latency.insert(latency.end(), host.begin(), host.end());
  }
  std::sort(latency.begin(), latency.end());
  auto const count = latency.size();
  double const median = .5 * (latency[(count - 1) / 2] + latency[count / 2]);
  char const* shapes[] = {"W", "K", "Vt"};
  std::fprintf(output, "L1,%s,%d,%d,%.6f,%.6f,%.1f,%d,%d,%d,128,1088\n", shapes[Shape], N * 8,
               active, gbps, gbps / active, median, iterations, kGrid, kThreads);
  std::fflush(output);
}

template <int Shape>
void ShapeCases(unsigned char const* input, unsigned* sink, unsigned long long* times, int iterations, FILE* output) {
  for (int active : {26, 52, 104}) {
    Run<Shape, 1>(input, sink, times, active, iterations, output);
    Run<Shape, 2>(input, sink, times, active, iterations, output);
    Run<Shape, 4>(input, sink, times, active, iterations, output);
    Run<Shape, 8>(input, sink, times, active, iterations, output);
    Run<Shape, 16>(input, sink, times, active, iterations, output);
  }
}

int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "load_probe <iterations> <out.csv>\n"); return 2; }
  int const iterations = std::atoi(argv[1]);
  if (iterations < 16) return 2;
  unsigned char* input;
  unsigned* sink;
  unsigned long long* times;
  CHECK(cudaMalloc(&input, std::size_t(kBytes) * 2));
  CHECK(cudaMemset(input, 0x5a, std::size_t(kBytes) * 2));
  CHECK(cudaMalloc(&sink, kGrid * 8 * sizeof(unsigned)));
  CHECK(cudaMalloc(&times, kGrid * 8 * sizeof(unsigned long long)));
  FILE* output = std::fopen(argv[2], "w");
  if (!output) return 2;
  std::fprintf(output, "probe,shape,outstanding_kib,active_ctas,gbps,per_cta_gbps,latency_ns,iterations,ctas,threads,dim,capacity\n");
  ShapeCases<0>(input, sink, times, iterations, output);
  ShapeCases<1>(input, sink, times, iterations, output);
  ShapeCases<2>(input, sink, times, iterations, output);
  std::fclose(output);
  CHECK(cudaFree(times)); CHECK(cudaFree(sink)); CHECK(cudaFree(input));
}
