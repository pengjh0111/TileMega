// SPDX-License-Identifier: BSD-3-Clause
// EX-D1 §3.5: measure %globaltimer before any per-slot timestamp is read.
//
// Two questions decide how trace v2 may be interpreted:
//   1. the increment granularity -- the smallest non-zero step between two
//      back-to-back reads, which bounds every latency the offline analysis
//      reconstructs from a pair of timestamps;
//   2. whether one instant reads the same on every SM, because a per-hop
//      latency subtracts a producer's timestamp from a consumer's and the two
//      run on different SMs.
//
// The spread is taken after a software arrival barrier over resident CTAs, not
// at launch, so it measures clock skew rather than the order the CTAs started.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do { cudaError_t e = (expr); \
  if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, \
    __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

__device__ inline unsigned long long TraceNow() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}

__device__ inline unsigned int TraceSmid() {
  unsigned int s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

/// One thread, N back-to-back reads.  The stores are pipelined and do not
/// serialize the loop; what the minimum non-zero difference exposes is the
/// timer's own tick, not the cost of the store.
__global__ void SerialProbe(unsigned long long* timer,
                            unsigned long long* clk, int n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int i = 0; i < n; ++i) {
    timer[i] = TraceNow();
    clk[i] = clock64();
  }
}

/// Every CTA arrives, then all spin until the last one has arrived, then each
/// reads once.  The reads are therefore separated by the barrier release
/// latency rather than by the launch schedule.
__global__ void InstantProbe(unsigned long long* timer, unsigned long long* clk,
                             unsigned int* smid, unsigned int* arrivals,
                             unsigned int blocks) {
  if (threadIdx.x != 0) return;
  atomicAdd(arrivals, 1u);
  while (atomicAdd(arrivals, 0u) < blocks) { /* resident spin */ }
  timer[blockIdx.x] = TraceNow();
  clk[blockIdx.x] = clock64();
  smid[blockIdx.x] = TraceSmid();
}

static unsigned long long Percentile(std::vector<unsigned long long> const& v,
                                     double p) {
  if (v.empty()) return 0;
  std::size_t at = static_cast<std::size_t>(p * (v.size() - 1) + 0.5);
  return v[std::min(at, v.size() - 1)];
}

int main(int argc, char** argv) {
  int const n = argc > 1 ? std::atoi(argv[1]) : 20000;
  int const blocks = argc > 2 ? std::atoi(argv[2]) : 128;
  char const* tsv = argc > 3 ? argv[3] : "resolution.tsv";
  char const* ctas_tsv = argc > 4 ? argv[4] : "resolution_ctas.tsv";
  if (n < 10000) { std::fprintf(stderr, "N must be >= 10000\n"); return 2; }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  unsigned long long *d_timer = nullptr, *d_clk = nullptr;
  CUDA_CHECK(cudaMalloc(&d_timer, sizeof(unsigned long long) * n));
  CUDA_CHECK(cudaMalloc(&d_clk, sizeof(unsigned long long) * n));
  SerialProbe<<<1, 1>>>(d_timer, d_clk, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  std::vector<unsigned long long> timer(n), clk(n);
  CUDA_CHECK(cudaMemcpy(timer.data(), d_timer, sizeof(unsigned long long) * n,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(clk.data(), d_clk, sizeof(unsigned long long) * n,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> timer_delta, clk_delta;
  std::size_t timer_zero = 0, clk_zero = 0, timer_back = 0, clk_back = 0;
  for (int i = 1; i < n; ++i) {
    if (timer[i] < timer[i - 1]) ++timer_back;
    else {
      unsigned long long d = timer[i] - timer[i - 1];
      if (d == 0) ++timer_zero; else timer_delta.push_back(d);
    }
    if (clk[i] < clk[i - 1]) ++clk_back;
    else {
      unsigned long long d = clk[i] - clk[i - 1];
      if (d == 0) ++clk_zero; else clk_delta.push_back(d);
    }
  }
  std::sort(timer_delta.begin(), timer_delta.end());
  std::sort(clk_delta.begin(), clk_delta.end());

  unsigned long long *d_itimer = nullptr, *d_iclk = nullptr;
  unsigned int *d_smid = nullptr, *d_arrivals = nullptr;
  CUDA_CHECK(cudaMalloc(&d_itimer, sizeof(unsigned long long) * blocks));
  CUDA_CHECK(cudaMalloc(&d_iclk, sizeof(unsigned long long) * blocks));
  CUDA_CHECK(cudaMalloc(&d_smid, sizeof(unsigned int) * blocks));
  CUDA_CHECK(cudaMalloc(&d_arrivals, sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(d_arrivals, 0, sizeof(unsigned int)));
  InstantProbe<<<blocks, 32>>>(d_itimer, d_iclk, d_smid, d_arrivals, blocks);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  std::vector<unsigned long long> itimer(blocks), iclk(blocks);
  std::vector<unsigned int> smid(blocks);
  CUDA_CHECK(cudaMemcpy(itimer.data(), d_itimer,
                        sizeof(unsigned long long) * blocks, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(iclk.data(), d_iclk,
                        sizeof(unsigned long long) * blocks, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(smid.data(), d_smid, sizeof(unsigned int) * blocks,
                        cudaMemcpyDeviceToHost));

  unsigned long long const tmin = *std::min_element(itimer.begin(), itimer.end());
  unsigned long long const tmax = *std::max_element(itimer.begin(), itimer.end());
  unsigned long long distinct_sms = 0;
  {
    std::vector<unsigned int> s = smid;
    std::sort(s.begin(), s.end());
    distinct_sms = static_cast<unsigned long long>(
        std::unique(s.begin(), s.end()) - s.begin());
  }

  std::FILE* out = std::fopen(tsv, "w");
  if (!out) { std::fprintf(stderr, "cannot write %s\n", tsv); return 2; }
  std::fprintf(out, "metric\tvalue\n");
  std::fprintf(out, "device\t%s\n", prop.name);
  std::fprintf(out, "compute_cap\t%d.%d\n", prop.major, prop.minor);
  std::fprintf(out, "serial_reads\t%d\n", n);
  std::fprintf(out, "globaltimer_min_nonzero_delta_ns\t%llu\n",
               timer_delta.empty() ? 0ull : timer_delta.front());
  std::fprintf(out, "globaltimer_p50_delta_ns\t%llu\n", Percentile(timer_delta, 0.50));
  std::fprintf(out, "globaltimer_p99_delta_ns\t%llu\n", Percentile(timer_delta, 0.99));
  std::fprintf(out, "globaltimer_max_delta_ns\t%llu\n",
               timer_delta.empty() ? 0ull : timer_delta.back());
  std::fprintf(out, "globaltimer_zero_deltas\t%zu\n", timer_zero);
  std::fprintf(out, "globaltimer_zero_delta_fraction\t%.6f\n",
               n > 1 ? static_cast<double>(timer_zero) / (n - 1) : 0.0);
  std::fprintf(out, "globaltimer_backward_deltas\t%zu\n", timer_back);
  std::fprintf(out, "clock64_min_nonzero_delta\t%llu\n",
               clk_delta.empty() ? 0ull : clk_delta.front());
  std::fprintf(out, "clock64_p50_delta\t%llu\n", Percentile(clk_delta, 0.50));
  std::fprintf(out, "clock64_p99_delta\t%llu\n", Percentile(clk_delta, 0.99));
  std::fprintf(out, "clock64_zero_deltas\t%zu\n", clk_zero);
  std::fprintf(out, "clock64_backward_deltas\t%zu\n", clk_back);
  std::fprintf(out, "instant_blocks\t%d\n", blocks);
  std::fprintf(out, "instant_distinct_smids\t%llu\n", distinct_sms);
  std::fprintf(out, "instant_globaltimer_min_ns\t%llu\n", tmin);
  std::fprintf(out, "instant_globaltimer_max_ns\t%llu\n", tmax);
  std::fprintf(out, "instant_globaltimer_spread_ns\t%llu\n", tmax - tmin);
  std::fprintf(out, "needs_clock64_columns\t%d\n",
               (!timer_delta.empty() && timer_delta.front() > 100ull) ? 1 : 0);
  std::fclose(out);

  std::FILE* per = std::fopen(ctas_tsv, "w");
  if (!per) { std::fprintf(stderr, "cannot write %s\n", ctas_tsv); return 2; }
  std::fprintf(per, "block\tsmid\tglobaltimer_ns\tclock64\tdelta_from_min_ns\n");
  for (int b = 0; b < blocks; ++b)
    std::fprintf(per, "%d\t%u\t%llu\t%llu\t%llu\n", b, smid[b], itimer[b],
                 iclk[b], itimer[b] - tmin);
  std::fclose(per);

  std::printf("RESOLUTION min_nonzero_ns=%llu p50_ns=%llu p99_ns=%llu "
              "zero_fraction=%.6f spread_ns=%llu distinct_smids=%llu "
              "needs_clock64=%d\n",
              timer_delta.empty() ? 0ull : timer_delta.front(),
              Percentile(timer_delta, 0.50), Percentile(timer_delta, 0.99),
              n > 1 ? static_cast<double>(timer_zero) / (n - 1) : 0.0,
              tmax - tmin, distinct_sms,
              (!timer_delta.empty() && timer_delta.front() > 100ull) ? 1 : 0);
  return 0;
}
