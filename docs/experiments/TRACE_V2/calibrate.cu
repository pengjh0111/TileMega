// SPDX-License-Identifier: BSD-3-Clause
// EX-D1 §3.5: per-SM clock64 offset estimation, required because %globaltimer
// increments in 1024 ns steps on this device (resolution.tsv).
//
// %globaltimer is one counter broadcast to every SM -- 128 CTAs on 128 distinct
// SMs read it bit-identically -- so its error is common-mode quantization, not
// a per-SM offset.  clock64 is the opposite: fine-grained, but each SM starts
// from its own value (measured spread 3.51e9 cycles) and counts at a
// DVFS-dependent rate.
//
// The estimator exploits exactly that asymmetry.  Every round, all resident
// CTAs meet at a software barrier and each reads the pair
// (clock64, %globaltimer).  %globaltimer is common, so it is a shared ruler;
// a least-squares fit of globaltimer against clock64 per SM yields that SM's
// rate `a` and offset `b`.  Individual anchors carry up to one 1024 ns tick of
// error, so the residual spread over K anchors -- reported here, not assumed --
// is the error bound for mapping that SM's clock64 onto the global timeline.
#include <cstdio>
#include <cstdlib>
#include <cmath>
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

__global__ void Calibrate(unsigned long long* timer, unsigned long long* clk,
                          unsigned int* smid, unsigned int* count,
                          unsigned int* sense, unsigned int blocks, int rounds,
                          int spin_cycles) {
  if (threadIdx.x != 0) return;
  unsigned int local_sense = 0;
  if (blockIdx.x == 0) smid[0] = 0;  // keep the store below uniform
  smid[blockIdx.x] = TraceSmid();
  for (int r = 0; r < rounds; ++r) {
    // Sense-reversing barrier over resident CTAs.  The counter is cleared by
    // the last arriver before the release, so the next round cannot observe a
    // stale count.
    local_sense ^= 1u;
    unsigned int const arrived = atomicAdd(count, 1u) + 1u;
    if (arrived == blocks) {
      atomicExch(count, 0u);
      __threadfence();
      atomicExch(sense, local_sense);
    } else {
      while (atomicAdd(sense, 0u) != local_sense) { /* resident spin */ }
    }
    clk[static_cast<std::size_t>(r) * blocks + blockIdx.x] = clock64();
    timer[static_cast<std::size_t>(r) * blocks + blockIdx.x] = TraceNow();
    long long const until = clock64() + spin_cycles;
    while (clock64() < until) { /* spread the anchors in time */ }
  }
}

int main(int argc, char** argv) {
  int const blocks = argc > 1 ? std::atoi(argv[1]) : 128;
  int const rounds = argc > 2 ? std::atoi(argv[2]) : 256;
  int const spin = argc > 3 ? std::atoi(argv[3]) : 20000;
  char const* tsv = argc > 4 ? argv[4] : "calibration.tsv";

  std::size_t const n = static_cast<std::size_t>(blocks) * rounds;
  unsigned long long *d_timer = nullptr, *d_clk = nullptr;
  unsigned int *d_smid = nullptr, *d_count = nullptr, *d_sense = nullptr;
  CUDA_CHECK(cudaMalloc(&d_timer, n * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_clk, n * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_smid, blocks * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_sense, sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(d_sense, 0, sizeof(unsigned int)));

  Calibrate<<<blocks, 32>>>(d_timer, d_clk, d_smid, d_count, d_sense,
                            static_cast<unsigned int>(blocks), rounds, spin);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  std::vector<unsigned long long> timer(n), clk(n);
  std::vector<unsigned int> smid(blocks);
  CUDA_CHECK(cudaMemcpy(timer.data(), d_timer, n * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(clk.data(), d_clk, n * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(smid.data(), d_smid, blocks * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  std::FILE* out = std::fopen(tsv, "w");
  if (!out) { std::fprintf(stderr, "cannot write %s\n", tsv); return 2; }
  // The fit is stored anchored, as t_ns(clk) = t0_ns + ns_per_cycle*(clk - clk0),
  // not as a bare intercept: %globaltimer readings are ~1.8e18 ns and a double
  // intercept would lose the low nanosecond digits the estimate is about.
  std::fprintf(out, "block\tsmid\tanchors\tns_per_cycle\timplied_mhz\t"
                    "clk0\tt0_ns\tresidual_rms_ns\tresidual_max_ns\n");

  double worst_rms = 0.0, worst_max = 0.0;
  double min_mhz = 1e30, max_mhz = 0.0;
  unsigned long long span_min = ~0ull, span_max = 0ull;
  for (int b = 0; b < blocks; ++b) {
    // Anchor the fit at each SM's own first sample so the normal equations stay
    // conditioned; clock64 values are ~1.8e12 and would otherwise lose
    // precision against a zero origin.
    double const clk0 = static_cast<double>(clk[b]);
    double const t0 = static_cast<double>(timer[b]);
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int r = 0; r < rounds; ++r) {
      double const x = static_cast<double>(clk[static_cast<std::size_t>(r) * blocks + b]) - clk0;
      double const y = static_cast<double>(timer[static_cast<std::size_t>(r) * blocks + b]) - t0;
      sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    double const den = rounds * sxx - sx * sx;
    double const a = den != 0.0 ? (rounds * sxy - sx * sy) / den : 0.0;
    double const b0 = (sy - a * sx) / rounds;
    double rms = 0.0, mx = 0.0;
    for (int r = 0; r < rounds; ++r) {
      double const x = static_cast<double>(clk[static_cast<std::size_t>(r) * blocks + b]) - clk0;
      double const y = static_cast<double>(timer[static_cast<std::size_t>(r) * blocks + b]) - t0;
      double const e = y - (a * x + b0);
      rms += e * e;
      mx = std::max(mx, std::fabs(e));
    }
    rms = std::sqrt(rms / rounds);
    double const mhz = a > 0.0 ? 1000.0 / a : 0.0;
    worst_rms = std::max(worst_rms, rms);
    worst_max = std::max(worst_max, mx);
    if (mhz > 0.0) { min_mhz = std::min(min_mhz, mhz); max_mhz = std::max(max_mhz, mhz); }
    span_min = std::min(span_min, clk[b]);
    span_max = std::max(span_max, clk[b]);
    std::fprintf(out, "%d\t%u\t%d\t%.9f\t%.3f\t%llu\t%llu\t%.1f\t%.1f\n", b,
                 smid[b], rounds, a, mhz, clk[b], timer[b], rms, mx);
  }
  std::fclose(out);

  // Window actually spanned by the anchors, measured on the common ruler.
  unsigned long long window = 0;
  for (int b = 0; b < blocks; ++b) {
    unsigned long long const first = timer[b];
    unsigned long long const last =
        timer[static_cast<std::size_t>(rounds - 1) * blocks + b];
    window = std::max(window, last > first ? last - first : 0ull);
  }
  // One anchor carries up to one 1024 ns tick of quantization, whose standard
  // deviation is 1024/sqrt(12) = 295.6 ns.  Averaged over K anchors the offset
  // uncertainty falls as 1/sqrt(K), and the slope is pinned far tighter than
  // that across the window, so this is the figure that bounds mapping one SM's
  // clock64 onto the global timeline.
  double const offset_sigma = worst_rms / std::sqrt(static_cast<double>(rounds));

  std::printf("CALIBRATION blocks=%d rounds=%d window_ns=%llu "
              "clock64_offset_spread_cycles=%llu "
              "implied_mhz_min=%.3f implied_mhz_max=%.3f "
              "residual_rms_max_ns=%.1f residual_max_ns=%.1f "
              "quantization_sigma_ns=%.1f offset_sigma_ns=%.2f\n",
              blocks, rounds, window, span_max - span_min, min_mhz, max_mhz,
              worst_rms, worst_max, 1024.0 / std::sqrt(12.0), offset_sigma);
  return 0;
}
