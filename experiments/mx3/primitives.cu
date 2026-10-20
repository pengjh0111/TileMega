// SPDX-License-Identifier: BSD-3-Clause
// MX3-1 item 4: the primitives the stage ledger attributes cost to, measured
// on this C550 rather than inferred. Each probe times a dependent chain in
// one kernel with GlobalTimerNs (100 MHz realtime counter, P-11) and divides
// by the iteration count, so per-operation cost is independent of launch.
//
// Usage: primitives <probe> [iters] [ctas]   (probe `all` runs every one)
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

using tilemega::maca::GlobalTimerNs;

#define CHECK(call)                                                         \
  do {                                                                      \
    cudaError_t e = (call);                                                  \
    if (e != cudaSuccess) {                                                  \
      std::printf("%s failed: %s\n", #call, cudaGetErrorString(e));          \
      return 2;                                                              \
    }                                                                        \
  } while (0)

/// Dependent pointer chase over `span` bytes: each load's address comes from
/// the previous load, so the chain cannot be overlapped and the per-step time
/// is the memory latency at that working set (F11 claims ~1.06 us for DRAM).
__global__ void ChaseKernel(unsigned const* next, int iters, unsigned long long* out,
                            unsigned* sink) {
  if (threadIdx.x) return;
  unsigned at = 0;
  unsigned long long begin = GlobalTimerNs();
  for (int i = 0; i < iters; ++i) at = next[at];
  unsigned long long end = GlobalTimerNs();
  *sink = at;
  out[blockIdx.x] = end - begin;
}

__global__ void FenceKernel(int iters, unsigned long long* out, unsigned* sink) {
  if (threadIdx.x) return;
  unsigned long long begin = GlobalTimerNs();
  for (int i = 0; i < iters; ++i) {
    __threadfence();
    __asm__ __volatile__("" ::: "memory");
  }
  unsigned long long end = GlobalTimerNs();
  *sink = 1;
  out[blockIdx.x] = end - begin;
}

/// Uncontended when every CTA owns a counter, contended when they share one.
__global__ void AtomicKernel(unsigned* counters, int iters, int shared,
                             unsigned long long* out) {
  if (threadIdx.x) return;
  unsigned* at = shared ? counters : counters + blockIdx.x;
  unsigned long long begin = GlobalTimerNs();
  for (int i = 0; i < iters; ++i) atomicAdd(at, 1u);
  unsigned long long end = GlobalTimerNs();
  out[blockIdx.x] = end - begin;
}

/// Dependent atomics: each add waits for the previous one's returned value,
/// which is what a handoff on the critical path pays. Independent adds
/// pipeline and measure throughput instead, so both are reported.
__global__ void AtomicChainKernel(unsigned* counters, int iters, int shared,
                                  unsigned long long* out, unsigned* sink) {
  if (threadIdx.x) return;
  unsigned* base = shared ? counters : counters + blockIdx.x;
  unsigned carried = 0;
  unsigned long long begin = GlobalTimerNs();
  for (int i = 0; i < iters; ++i) carried = atomicAdd(base + (carried & 0u), 1u);
  unsigned long long end = GlobalTimerNs();
  *sink = carried;
  out[blockIdx.x] = end - begin;
}

/// One CTA publishes a rising ticket; another polls until it sees each value.
/// The round trip is what a stage handoff pays to become visible (F7).
__global__ void VisibilityKernel(unsigned volatile* flag, int iters,
                                 unsigned long long* out) {
  if (threadIdx.x) return;
  if (blockIdx.x == 0) {
    for (int i = 1; i <= iters; ++i) {
      __threadfence();
      atomicAdd(const_cast<unsigned*>(flag), 1u);
      while (*flag < unsigned(i)) {}
    }
  } else if (blockIdx.x == 1) {
    unsigned long long begin = GlobalTimerNs();
    for (int i = 1; i <= iters; ++i)
      while (*flag < unsigned(i)) {}
    unsigned long long end = GlobalTimerNs();
    out[0] = end - begin;
  }
}

/// The V3 grid barrier: CTA barrier, then thread 0 fences and adds, then every
/// CTA spins for the generation to advance.
__global__ void GridBarrierKernel(unsigned* arrive, int iters, int ctas,
                                  unsigned long long* out) {
  unsigned long long begin = GlobalTimerNs();
  for (int i = 1; i <= iters; ++i) {
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      atomicAdd(arrive, 1u);
      unsigned target = unsigned(i) * unsigned(ctas);
      while (*reinterpret_cast<unsigned volatile*>(arrive) < target) {}
      __threadfence();
    }
    __syncthreads();
  }
  unsigned long long end = GlobalTimerNs();
  if (threadIdx.x == 0) out[blockIdx.x] = end - begin;
}

__global__ void ShuffleKernel(int iters, unsigned long long* out, float* sink) {
  float v = float(threadIdx.x);
  unsigned long long begin = GlobalTimerNs();
  for (int i = 0; i < iters; ++i) {
    v += __shfl_xor(v, 1);
    v += __shfl_xor(v, 2);
    v += __shfl_xor(v, 4);
    v += __shfl_xor(v, 8);
  }
  unsigned long long end = GlobalTimerNs();
  if (threadIdx.x == 0) {
    *sink = v;
    out[blockIdx.x] = (end - begin);
  }
}

__global__ void SyncthreadsKernel(int iters, unsigned long long* out) {
  unsigned long long begin = GlobalTimerNs();
  for (int i = 0; i < iters; ++i) __syncthreads();
  unsigned long long end = GlobalTimerNs();
  if (threadIdx.x == 0) out[blockIdx.x] = end - begin;
}

__global__ void NanosleepKernel(int iters, unsigned ns, unsigned long long* out) {
  if (threadIdx.x) return;
  unsigned long long begin = GlobalTimerNs();
  for (int i = 0; i < iters; ++i) __nanosleep(ns);
  unsigned long long end = GlobalTimerNs();
  out[blockIdx.x] = end - begin;
}

static void Report(char const* name, char const* unit, std::vector<double> per) {
  std::sort(per.begin(), per.end());
  double median = per[per.size() / 2];
  double p99 = per[std::min(per.size() - 1, std::size_t(per.size() * 0.99))];
  std::printf("%-34s %10.3f %10.3f  %s (n=%zu)\n", name, median, p99, unit, per.size());
}

int main(int argc, char** argv) {
  char const* probe = argc > 1 ? argv[1] : "all";
  int iters = argc > 2 ? std::atoi(argv[2]) : 2000;
  int ctas = argc > 3 ? std::atoi(argv[3]) : 104;
  bool all = std::strcmp(probe, "all") == 0;
  auto want = [&](char const* n) { return all || std::strcmp(probe, n) == 0; };

  unsigned long long* out;
  unsigned* scratch;
  float* fsink;
  CHECK(cudaMalloc(&out, 512 * sizeof(unsigned long long)));
  CHECK(cudaMalloc(&scratch, 512 * sizeof(unsigned)));
  CHECK(cudaMalloc(&fsink, sizeof(float)));
  std::vector<unsigned long long> host(512);
  auto collect = [&](int n, int divisor, char const* name, char const* unit) {
    CHECK(cudaMemcpy(host.data(), out, n * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost));
    std::vector<double> per;
    for (int i = 0; i < n; ++i)
      if (host[i]) per.push_back(double(host[i]) / divisor);
    if (per.empty()) { std::printf("%-34s no samples\n", name); return 0; }
    Report(name, unit, per);
    return 0;
  };

  std::printf("%-34s %10s %10s\n", "probe", "median", "p99");

  if (want("chase")) {
    // 256 MiB of 128-byte-strided links: far past the 8 MiB L2 knee, so every
    // step misses. A 64 KiB span stays inside L2 for the comparison.
    for (long long span : {64ll << 10, 8ll << 20, 256ll << 20}) {
      long long count = span / 128;
      std::vector<unsigned> links(count);
      for (long long i = 0; i < count; ++i) links[i] = unsigned((i + 1) % count);
      // Shuffle so the prefetcher cannot predict the order.
      for (long long i = count - 1; i > 0; --i) std::swap(links[i], links[(i * 2654435761u) % (i + 1)]);
      unsigned* device;
      CHECK(cudaMalloc(&device, count * 128));
      std::vector<unsigned> strided(count * 32, 0u);
      for (long long i = 0; i < count; ++i) strided[i * 32] = links[i] * 32;
      CHECK(cudaMemcpy(device, strided.data(), count * 128, cudaMemcpyHostToDevice));
      CHECK(cudaMemset(out, 0, 512 * sizeof(unsigned long long)));
      ChaseKernel<<<1, 64>>>(device, iters, out, scratch);
      CHECK(cudaDeviceSynchronize());
      char name[64];
      std::snprintf(name, sizeof name, "dependent read chain %lldKiB", span >> 10);
      collect(1, iters, name, "ns/load");
      CHECK(cudaFree(device));
    }
  }
  if (want("fence")) {
    CHECK(cudaMemset(out, 0, 512 * sizeof(unsigned long long)));
    FenceKernel<<<1, 64>>>(iters, out, scratch);
    CHECK(cudaDeviceSynchronize());
    collect(1, iters, "__threadfence", "ns");
  }
  if (want("atomic")) {
    for (int shared : {0, 1}) {
      for (int n : {1, ctas, 2 * ctas}) {
        if (!shared && n != 1) continue;
        CHECK(cudaMemset(scratch, 0, 512 * sizeof(unsigned)));
        CHECK(cudaMemset(out, 0, 512 * sizeof(unsigned long long)));
        AtomicKernel<<<n, 64>>>(scratch, iters, shared, out);
        CHECK(cudaDeviceSynchronize());
        char name[64];
        std::snprintf(name, sizeof name, "atomicAdd %s %d CTA",
                      shared ? "same address," : "private,", n);
        collect(n, iters, name, "ns");
      }
    }
  }
  if (want("atomicchain")) {
    for (int n : {1, ctas}) {
      CHECK(cudaMemset(scratch, 0, 512 * sizeof(unsigned)));
      CHECK(cudaMemset(out, 0, 512 * sizeof(unsigned long long)));
      AtomicChainKernel<<<n, 64>>>(scratch, iters, 1, out, scratch + 400);
      CHECK(cudaDeviceSynchronize());
      char name[64];
      std::snprintf(name, sizeof name, "atomicAdd dependent, %d CTA", n);
      collect(n, iters, name, "ns");
    }
  }
  if (want("visibility")) {
    CHECK(cudaMemset(scratch, 0, 512 * sizeof(unsigned)));
    CHECK(cudaMemset(out, 0, 512 * sizeof(unsigned long long)));
    VisibilityKernel<<<2, 64>>>(scratch, iters, out);
    CHECK(cudaDeviceSynchronize());
    collect(1, iters, "cross-AP publish to observe", "ns");
  }
  if (want("gridbarrier")) {
    for (int n : {ctas, 2 * ctas}) {
      CHECK(cudaMemset(scratch, 0, 512 * sizeof(unsigned)));
      CHECK(cudaMemset(out, 0, 512 * sizeof(unsigned long long)));
      GridBarrierKernel<<<n, 128>>>(scratch, iters, n, out);
      CHECK(cudaDeviceSynchronize());
      char name[64];
      std::snprintf(name, sizeof name, "V3 grid barrier %d CTA", n);
      collect(n, iters, name, "ns");
    }
  }
  if (want("shuffle")) {
    CHECK(cudaMemset(out, 0, 512 * sizeof(unsigned long long)));
    ShuffleKernel<<<1, 128>>>(iters, out, fsink);
    CHECK(cudaDeviceSynchronize());
    collect(1, iters * 4, "__shfl_xor", "ns");
  }
  if (want("syncthreads")) {
    for (int threads : {128, 256}) {
      CHECK(cudaMemset(out, 0, 512 * sizeof(unsigned long long)));
      SyncthreadsKernel<<<1, threads>>>(iters, out);
      CHECK(cudaDeviceSynchronize());
      char name[64];
      std::snprintf(name, sizeof name, "__syncthreads %d threads", threads);
      collect(1, iters, name, "ns");
    }
  }
  if (want("nanosleep")) {
    for (unsigned ns : {0u, 1u, 100u, 1000u}) {
      CHECK(cudaMemset(out, 0, 512 * sizeof(unsigned long long)));
      NanosleepKernel<<<1, 64>>>(iters, ns, out);
      CHECK(cudaDeviceSynchronize());
      char name[64];
      std::snprintf(name, sizeof name, "__nanosleep(%u)", ns);
      collect(1, iters, name, "ns");
    }
  }
  return 0;
}
