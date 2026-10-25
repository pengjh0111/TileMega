// SPDX-License-Identifier: BSD-3-Clause
// MX-5 S1--S4. Each case runs all 104 CTAs; the ninth wave observes releases
// without issuing data loads. N is the number of cold 16 B loads per lane.
#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using tilemega::maca::GlobalTimerNs;
constexpr int kGrid = 104, kThreads = 576, kWaves = 8;
constexpr int kSmem = 32784;
constexpr unsigned kVectors = (256u << 20) / sizeof(uint4);

#define CHECK(x) do { auto e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(2); } } while (0)

enum Probe { kLoad, kCta, kLds, kExchange, kPoll, kRelease };

__device__ __forceinline__ unsigned Peek(unsigned const* p) {
  return *(__attribute__((address_space(1))) volatile unsigned const*)p;
}

template <int N>
__device__ __forceinline__ void Issue(uint4 (&pending)[N ? N : 1], uint4 const* input, int it) {
  unsigned const wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  if (wave < kWaves) {
#pragma unroll
    for (int n = 0; n < N; ++n) {
      unsigned const offset = ((((unsigned(it) * kGrid + blockIdx.x) * 8 + wave) * 8 + n) * 64 + lane)
                              & (kVectors - 1);
      pending[n] = tilemega::backend::LoadGlobal16Gvm(input + offset);
    }
  }
  asm volatile("" ::: "memory");
}

template <int N>
__device__ __forceinline__ unsigned Consume(uint4 const (&pending)[N ? N : 1]) {
  unsigned sum = 0;
  if (threadIdx.x < 512) {
#pragma unroll
    for (int n = 0; n < N; ++n)
      sum += pending[n].x + pending[n].y + pending[n].z + pending[n].w;
  }
  asm volatile("" :: "r"(sum) : "memory");
  return sum;
}

template <int Variant>
__device__ __forceinline__ void CtaSync(unsigned char* shared, uint4 const* input) {
  if constexpr (Variant == 0) __syncthreads();
  if constexpr (Variant == 1) __builtin_mxc_barrier();
  if constexpr (Variant == 2) __builtin_mxc_barrier_shared();
  if constexpr (Variant == 3) __builtin_mxc_barrier_inst();
  if constexpr (Variant == 4) __builtin_mxc_barrier_ex(4);
  if constexpr (Variant == 5) {
    __builtin_mxc_arrive_bsmcnt(0);
    __builtin_mxc_barrier_inst();
    asm volatile("" ::: "memory");
  }
  // These three builtins consume an async-copy token, not a scalar constant.
  if constexpr (Variant == 6)
    __builtin_mxc_barrier_and_wait1(1, memcpy_async<4>(shared + 4096 + threadIdx.x * 16,
                                                      const_cast<uint4*>(input)));
  if constexpr (Variant == 7)
    __builtin_mxc_barrier_and_wait2(1, memcpy_async<8>(shared + 4096 + threadIdx.x * 16,
                                                      const_cast<uint4*>(input)));
  if constexpr (Variant == 8)
    __builtin_mxc_barrier_and_wait4(1, memcpy_async<16>(shared + 4096 + threadIdx.x * 16,
                                                       const_cast<uint4*>(input)));
}

__device__ __forceinline__ void CloseIteration(unsigned* counter, unsigned it) {
  __syncthreads();
  if (threadIdx.x == 512) {
    __threadfence();
    atomicAdd(counter, 1u);
    while (Peek(counter) < kGrid * (it + 1)) {}
    __threadfence();
  }
  __syncthreads();
}

template <int Kind, int Variant, int N>
__global__ void __launch_bounds__(kThreads)
SyncProbe(uint4 const* input, unsigned* flags, unsigned long long* starts,
          unsigned* closing, unsigned long long* result, unsigned* sink, int iterations) {
  extern __shared__ __align__(16) unsigned char storage[];
  volatile unsigned* const ready = reinterpret_cast<volatile unsigned*>(storage);
  volatile unsigned* const generation = ready + 8;
  volatile unsigned long long* const released = reinterpret_cast<volatile unsigned long long*>(storage + 48);
  unsigned* const exchange = reinterpret_cast<unsigned*>(storage + 64);
  unsigned const wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  if (threadIdx.x < 9) ready[threadIdx.x] = 0;
  __syncthreads();
  for (int it = 0; it < iterations; ++it) {
    if constexpr (Kind == kPoll) {
      if (threadIdx.x == 512) {
        __threadfence();
        atomicAdd(flags, 1u);
      }
      __syncthreads();
    }
    auto const issued = GlobalTimerNs();
    uint4 pending[N ? N : 1];
    Issue<N>(pending, input, it);
    auto const begin = GlobalTimerNs();
    unsigned long long elapsed = 0, end = begin;
    if constexpr (Kind == kCta) CtaSync<Variant>(storage, input);
    if constexpr (Kind == kLds) {
      if (wave < 8) {
        if (lane == 0) ready[wave] = it + 1;
        while (*generation < unsigned(it + 1)) {}
        end = GlobalTimerNs();
        elapsed = end - *released;
        asm volatile("" ::: "memory");
      } else if (lane == 0) {
        for (int w = 0; w < 8; ++w) while (ready[w] < unsigned(it + 1)) {}
        *released = GlobalTimerNs();
        __builtin_mxc_arrive_bsmcnt(0);
        *generation = it + 1;
      }
    }
    if constexpr (Kind == kExchange) {
      if (wave < 8) {
        exchange[wave * 64 + lane] = unsigned(it) * 64 + lane;
        __syncwarp();
        unsigned x = 0;
#pragma unroll
        for (int g = 0; g < 4; ++g) x += exchange[wave * 64 + (lane % 16) + 16 * g];
        asm volatile("" :: "r"(x) : "memory");
      }
    }
    if constexpr (Kind == kPoll) {
      if (threadIdx.x == 0) {
        unsigned value;
        do {
          if constexpr (Variant == 0) value = Peek(flags);
          if constexpr (Variant == 1) value = atomicAdd(flags, 0u);
          if constexpr (Variant == 2)
            value = __maca_atomic_load(flags, __ATOMIC_RELAXED, memory_scope_device);
        } while (value < unsigned(kGrid * (it + 1)));
        end = GlobalTimerNs();
        elapsed = end - begin;
      }
    }
    if constexpr (Kind == kRelease) {
      if (threadIdx.x == 0) {
        starts[blockIdx.x] = GlobalTimerNs();
        if constexpr (Variant == 0) {
          __threadfence();
          atomicAdd(flags + blockIdx.x, 1u);
        } else {
          __maca_atomic_fetch_add(flags + blockIdx.x, 1u, __ATOMIC_RELEASE, memory_scope_device);
        }
      }
      if (threadIdx.x == 512) {
        unsigned const other = (blockIdx.x + kGrid - 1) % kGrid;
        while (Peek(flags + other) < unsigned(it + 1)) {}
        end = GlobalTimerNs();
        auto const start = *(__attribute__((address_space(1))) unsigned long long const*)(starts + other);
        elapsed = end - start;
      }
    }
    if constexpr (Kind == kLoad || Kind == kCta || Kind == kExchange) {
      end = GlobalTimerNs();
      elapsed = end - begin;
    }
    unsigned const checksum = Consume<N>(pending);
    auto const available = GlobalTimerNs();
    bool emit = wave < 8 && lane == 0;
    if constexpr (Kind == kPoll) emit = threadIdx.x == 0;
    if constexpr (Kind == kRelease) emit = threadIdx.x == 512;
    if (emit) {
      int const slot = ((it * kGrid + blockIdx.x) * 8 + (wave % 8)) * 3;
      result[slot] = elapsed;
      result[slot + 1] = available - issued;
      result[slot + 2] = available - end;
      sink[blockIdx.x * 8 + (wave % 8)] = checksum;
    }
    CloseIteration(closing, unsigned(it));
  }
}

struct Buffers {
  uint4* input;
  unsigned *flags, *closing, *sink;
  unsigned long long *starts, *result;
};

template <int Kind, int Variant, int N>
void Run(Buffers const& b, int iterations, char const* probe, char const* variant, FILE* output) {
  CHECK(cudaMemset(b.flags, 0, kGrid * sizeof(unsigned)));
  CHECK(cudaMemset(b.closing, 0, sizeof(unsigned)));
  int resident = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, SyncProbe<Kind, Variant, N>, kThreads, kSmem));
  if (resident != 1) { std::fprintf(stderr, "resident=%d, expected 1\n", resident); std::exit(3); }
  SyncProbe<Kind, Variant, N><<<kGrid, kThreads, kSmem>>>(b.input, b.flags, b.starts, b.closing,
                                                       b.result, b.sink, iterations);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  std::vector<unsigned long long> samples(std::size_t(iterations) * kGrid * 8 * 3);
  CHECK(cudaMemcpy(samples.data(), b.result, samples.size() * sizeof(samples[0]), cudaMemcpyDeviceToHost));
  double median[3];
  std::size_t count = 0;
  for (int c = 0; c < 3; ++c) {
    std::vector<unsigned long long> values;
    for (int it = 8; it < iterations; ++it)
      for (int cta = 0; cta < kGrid; ++cta)
        for (int wave = 0; wave < ((Kind == kPoll || Kind == kRelease) ? 1 : 8); ++wave)
          values.push_back(samples[((it * kGrid + cta) * 8 + wave) * 3 + c]);
    std::sort(values.begin(), values.end());
    count = values.size();
    median[c] = .5 * (values[(count - 1) / 2] + values[count / 2]);
  }
  std::fprintf(output, "%s,%s,%d,%.1f,%.1f,%.1f,%zu,%d,%d,%d\n", probe, variant, N,
               median[0], median[1], median[2], count, iterations, kGrid, kThreads);
  std::fflush(output);
}

template <int Kind, int Variant>
void Loads(Buffers const& b, int iterations, char const* probe, char const* variant, FILE* output) {
  Run<Kind, Variant, 0>(b, iterations, probe, variant, output);
  Run<Kind, Variant, 2>(b, iterations, probe, variant, output);
  Run<Kind, Variant, 8>(b, iterations, probe, variant, output);
}

int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "sync_probe <iterations> <out.csv>\n"); return 2; }
  int const iterations = std::atoi(argv[1]);
  if (iterations < 16) return 2;
  Buffers b{};
  CHECK(cudaMalloc(&b.input, std::size_t(kVectors) * sizeof(uint4)));
  CHECK(cudaMemset(b.input, 0x5a, std::size_t(kVectors) * sizeof(uint4)));
  CHECK(cudaMalloc(&b.flags, kGrid * sizeof(unsigned)));
  CHECK(cudaMalloc(&b.closing, sizeof(unsigned)));
  CHECK(cudaMalloc(&b.sink, kGrid * 8 * sizeof(unsigned)));
  CHECK(cudaMalloc(&b.starts, kGrid * sizeof(unsigned long long)));
  CHECK(cudaMalloc(&b.result, std::size_t(iterations) * kGrid * 8 * 3 * sizeof(unsigned long long)));
  FILE* output = std::fopen(argv[2], "w");
  if (!output) return 2;
  std::fprintf(output, "probe,variant,n_kib,operation_ns,load_ready_ns,post_operation_ns,samples,iterations,ctas,threads\n");
  Loads<kLoad, 0>(b, iterations, "L8", "no_operation", output);
  Loads<kCta, 0>(b, iterations, "S1", "syncthreads", output);
  Loads<kCta, 1>(b, iterations, "S1", "builtin_barrier", output);
  Loads<kCta, 2>(b, iterations, "S1", "barrier_shared", output);
  Loads<kCta, 3>(b, iterations, "S1", "barrier_inst", output);
  Loads<kCta, 4>(b, iterations, "S1", "barrier_ex4", output);
  Loads<kCta, 5>(b, iterations, "S1", "lds_wait_barrier_inst", output);
  Loads<kCta, 6>(b, iterations, "S1", "barrier_and_wait1", output);
  Loads<kCta, 7>(b, iterations, "S1", "barrier_and_wait2", output);
  Loads<kCta, 8>(b, iterations, "S1", "barrier_and_wait4", output);
  Loads<kLds, 0>(b, iterations, "S2", "lds_release", output);
  Loads<kExchange, 0>(b, iterations, "S2b", "syncwarp_exchange", output);
  Loads<kPoll, 0>(b, iterations, "S3", "volatile", output);
  Loads<kPoll, 1>(b, iterations, "S3", "atomic_add_zero", output);
  Loads<kPoll, 2>(b, iterations, "S3", "atomic_load_relaxed", output);
  Loads<kRelease, 0>(b, iterations, "S4", "fence_atomic_add", output);
  Loads<kRelease, 1>(b, iterations, "S4", "atomic_add_release", output);
  std::fclose(output);
  CHECK(cudaFree(b.result)); CHECK(cudaFree(b.starts)); CHECK(cudaFree(b.sink));
  CHECK(cudaFree(b.closing)); CHECK(cudaFree(b.flags)); CHECK(cudaFree(b.input));
}
