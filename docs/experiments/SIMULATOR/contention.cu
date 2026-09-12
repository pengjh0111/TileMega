// SPDX-License-Identifier: BSD-3-Clause
// EX-S1 §5.2: the calibration the round-one cost model was missing.
//
// `headroom.py` priced every hop at the 1024 ns %globaltimer tick and modelled
// no contention at all, and so underestimated mode 5 by 1.7-3.7x (F-139).  Mode
// 5 concentrates fan-in onto a few event rows, so the quantity the simulator
// needs is not one hop latency but a surface: how long from a publish until the
// *last* of N polling CTAs sees it, when those N are spread over R rows.
//
// Nothing about the protocol changes here.  The wait is the generated
// `while (EventPoll(ev) < need) __nanosleep(64)` and the publish is
// `atomicExch`, both lifted from ModelHarness.cuh, so the curve prices the
// primitive the runtime actually executes -- the `__nanosleep(64)` backoff
// included, since a consumer asleep at the moment of publication really does
// pay for it.
//
// Resolution.  %globaltimer steps in 1024 ns on sm_89 (TRACE_V2/resolution.md),
// far coarser than one hop.  It is one counter broadcast to every SM and read
// bit-identically across them, so a publish-to-observe difference carries no
// per-SM offset, only two independent quantizations with zero mean.  The mean
// over K rounds therefore converges with standard error
// 1024 * sqrt(2/12) / sqrt(K) ns -- 9.2 ns at K = 2048 -- which is why the mean
// columns are the calibration and the per-round max is reported as a spread
// rather than fitted.  No per-SM clock64 mapping is needed, and none is done.
#include <tilemega/Codegen/tasks/EventSync.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do { cudaError_t e = (expr); \
  if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, \
    __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

namespace {

__device__ inline unsigned long long Now() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}

/// One event row, padded exactly as EventCounter is, so a row's traffic does
/// not land on a neighbour's cache line and R is really the number of lines.
struct alignas(128) Row {
  unsigned long long epoch;
  unsigned char padding[120];
};

/// CTAs [0, rows) publish, CTAs [rows, rows + consumers) poll.  Consumer c
/// polls row c % rows, so `consumers / rows` CTAs share a line.
__global__ void Contend(Row* rows_data, unsigned long long* publish_ns,
                        unsigned long long* observe_ns, unsigned int* count,
                        unsigned int* sense, unsigned int participants,
                        int rows, int consumers, int rounds, int guard_cycles,
                        int backoff_ns) {
  if (threadIdx.x != 0) return;
  unsigned int local_sense = 0;
  bool const publisher = blockIdx.x < static_cast<unsigned int>(rows);
  int const row = publisher ? static_cast<int>(blockIdx.x)
                            : (static_cast<int>(blockIdx.x) - rows) % rows;
  int const consumer = publisher ? -1 : static_cast<int>(blockIdx.x) - rows;

  for (int r = 0; r < rounds; ++r) {
    // Sense-reversing barrier over the resident participants: every consumer
    // must be inside its poll loop before any publish, or the measurement is
    // of arrival rather than propagation.  Same shape as TRACE_V2/calibrate.cu.
    local_sense ^= 1u;
    unsigned int const arrived = atomicAdd(count, 1u) + 1u;
    if (arrived == participants) {
      atomicExch(count, 0u);
      __threadfence();
      atomicExch(sense, local_sense);
    } else {
      while (atomicAdd(sense, 0u) != local_sense) { /* resident spin */ }
    }

    unsigned long long const need = static_cast<unsigned long long>(r) + 1ull;
    if (publisher) {
      // The guard covers barrier release skew; it is outside the stamp, so it
      // costs wall clock and not accuracy.  The dither is what makes averaging
      // work: without it the loop is periodic, the publish lands at a fixed
      // phase inside the 1024 ns tick, and every round reports exactly one tick
      // (measured -- that was this benchmark's first result).  The range spans
      // more than a tick at any plausible clock, so the phase is uniform and
      // the mean of the quantized differences converges on the true delay.
      // The dither depends on the round alone, so all R publishers release
      // together: R rows contending at the same instant is the regime mode 5
      // puts the device in, and staggering them would measure a gentler one.
      unsigned int const dither = (static_cast<unsigned int>(r) * 2654435761u) % 4096u;
      long long const until = clock64() + guard_cycles + dither;
      while (clock64() < until) { /* let the consumers settle into the poll */ }
      publish_ns[static_cast<std::size_t>(r) * rows + row] = Now();
      __threadfence();
      atomicExch(&rows_data[row].epoch, need);
    } else {
      // backoff_ns = 64 is the generated wait verbatim; 0 is the same loop with
      // the backoff removed, which is the only way to tell "contention is small"
      // apart from "the backoff hides it".  Neither arm changes the protocol.
      if (backoff_ns > 0)
        while (::tilemega::codegen::EventPoll(&rows_data[row].epoch) < need)
          __nanosleep(64);
      else
        while (::tilemega::codegen::EventPoll(&rows_data[row].epoch) < need) {}
      observe_ns[static_cast<std::size_t>(r) * consumers + consumer] = Now();
    }
  }
}

double Quantile(std::vector<double>& v, double q) {
  if (v.empty()) return 0.0;
  std::size_t const k = static_cast<std::size_t>(q * (v.size() - 1) + 0.5);
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

/// Mean with the top `frac` of samples dropped.  One process of the load-poll
/// arm measured a single ~2.3 ms hop in every cell while p50 and p90 stayed at
/// one and two ticks -- a device-level stall, not a hop -- and 4096 rounds are
/// not enough for one such sample to wash out of a 1.2 us mean.  The untrimmed
/// mean is still reported beside this one; nothing is clamped or discarded from
/// the table.
double TrimmedMean(std::vector<double>& v, double frac) {
  std::sort(v.begin(), v.end());
  std::size_t const drop = static_cast<std::size_t>(v.size() * frac);
  std::size_t const keep = v.size() > drop ? v.size() - drop : v.size();
  double sum = 0.0;
  for (std::size_t i = 0; i < keep; ++i) sum += v[i];
  return sum / static_cast<double>(keep);
}

}  // namespace

int main(int argc, char** argv) {
  int const rounds = argc > 1 ? std::atoi(argv[1]) : 2048;
  int const guard = argc > 2 ? std::atoi(argv[2]) : 20000;
  char const* tsv = argc > 3 ? argv[3] : "contention.tsv";
  int const warmup = 64;

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  int max_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
      reinterpret_cast<void const*>(Contend), 32, 0));
  int const resident = max_blocks * prop.multiProcessorCount;

  std::FILE* out = std::fopen(tsv, "w");
  if (!out) { std::fprintf(stderr, "cannot write %s\n", tsv); return 2; }
  std::fprintf(out, "poll_mode\tbackoff_ns\tconsumers\trows\tper_row\trounds\tparticipants\t"
                    "hop_mean_ns\thop_se_ns\thop_p50_ns\thop_p90_ns\thop_max_ns\t"
                    "last_mean_ns\tlast_p90_ns\tlast_max_ns\tinversions\t"
                    "hop_trim_mean_ns\thop_trim_se_ns\tlast_trim_mean_ns\n");

  char const* mode = TILEMEGA_EVENT_LOAD_POLL ? "load" : "rmw";
  int const consumer_set[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  int const row_set[] = {1, 4, 16, 64};
  int const backoff_set[] = {64, 0};

  for (int backoff : backoff_set)
  for (int rows : row_set)
    for (int consumers : consumer_set) {
      // R rows are only meaningful once there is a consumer for each; a cell
      // with fewer consumers than rows would silently measure a smaller R.
      if (consumers < rows) continue;
      unsigned int const participants = static_cast<unsigned int>(rows + consumers);
      if (static_cast<int>(participants) > resident) {
        std::fprintf(stderr, "skip backoff=%d consumers=%d rows=%d: %u CTAs exceed "
                             "the resident %d\n", backoff, consumers, rows,
                     participants, resident);
        continue;
      }

      Row* d_rows = nullptr;
      unsigned long long *d_publish = nullptr, *d_observe = nullptr;
      unsigned int *d_count = nullptr, *d_sense = nullptr;
      std::size_t const total = static_cast<std::size_t>(rounds + warmup);
      CUDA_CHECK(cudaMalloc(&d_rows, rows * sizeof(Row)));
      CUDA_CHECK(cudaMalloc(&d_publish, total * rows * sizeof(unsigned long long)));
      CUDA_CHECK(cudaMalloc(&d_observe, total * consumers * sizeof(unsigned long long)));
      CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned int)));
      CUDA_CHECK(cudaMalloc(&d_sense, sizeof(unsigned int)));
      CUDA_CHECK(cudaMemset(d_rows, 0, rows * sizeof(Row)));
      CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned int)));
      CUDA_CHECK(cudaMemset(d_sense, 0, sizeof(unsigned int)));

      Contend<<<participants, 32>>>(d_rows, d_publish, d_observe, d_count,
                                    d_sense, participants, rows, consumers,
                                    rounds + warmup, guard, backoff);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());

      std::vector<unsigned long long> publish(total * rows), observe(total * consumers);
      CUDA_CHECK(cudaMemcpy(publish.data(), d_publish,
                            publish.size() * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(observe.data(), d_observe,
                            observe.size() * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_rows)); CUDA_CHECK(cudaFree(d_publish));
      CUDA_CHECK(cudaFree(d_observe)); CUDA_CHECK(cudaFree(d_count));
      CUDA_CHECK(cudaFree(d_sense));

      // Two quantities, and they are not the same thing.  `hops` is one
      // consumer seeing one publish, which is the edge weight for a single
      // dependency.  `last` is the slowest consumer *of one row*, which is what
      // a producer's whole fan-out costs; taking the max across rows instead
      // would mix independent publishes and measure the dither, not contention.
      std::vector<double> hops, last;
      long inversions = 0;
      double sum = 0.0, sum_sq = 0.0;
      std::vector<double> row_max(rows);
      for (int r = warmup; r < rounds + warmup; ++r) {
        std::fill(row_max.begin(), row_max.end(), -1e30);
        for (int c = 0; c < consumers; ++c) {
          unsigned long long const o = observe[static_cast<std::size_t>(r) * consumers + c];
          unsigned long long const p =
              publish[static_cast<std::size_t>(r) * rows + c % rows];
          // A negative hop is the two quantizations disagreeing by a tick, not
          // a causality violation; it is counted and kept, never clamped.
          double const hop = static_cast<double>(o) - static_cast<double>(p);
          if (hop < 0.0) ++inversions;
          hops.push_back(hop);
          sum += hop; sum_sq += hop * hop;
          row_max[c % rows] = std::max(row_max[c % rows], hop);
        }
        for (double m : row_max) last.push_back(m);
      }
      double const n = static_cast<double>(hops.size());
      double const mean = sum / n;
      double const variance = std::max(0.0, sum_sq / n - mean * mean);
      // The tick quantization dominates a single sample; the standard error of
      // the mean is what the simulator is entitled to use.
      double const se = std::sqrt(variance / n);
      std::vector<double> hop_copy = hops, last_copy = last;
      // 0.1% trimmed: at 4096 rounds that is 4 samples per consumer, enough to
      // absorb the rare device stall without touching the tick distribution.
      std::vector<double> hop_trim = hops, last_trim = last;
      double const trim_mean = TrimmedMean(hop_trim, 0.001);
      double trim_var = 0.0;
      std::size_t const trim_keep = hop_trim.size() - static_cast<std::size_t>(hop_trim.size() * 0.001);
      for (std::size_t i = 0; i < trim_keep; ++i) {
        double const e = hop_trim[i] - trim_mean;
        trim_var += e * e;
      }
      double const trim_se = std::sqrt(trim_var / static_cast<double>(trim_keep) /
                                       static_cast<double>(trim_keep));
      std::fprintf(out, "%s\t%d\t%d\t%d\t%d\t%d\t%u\t%.1f\t%.2f\t%.1f\t%.1f\t%.1f\t"
                        "%.1f\t%.1f\t%.1f\t%ld\t%.1f\t%.2f\t%.1f\n",
                   mode, backoff, consumers, rows, consumers / rows, rounds, participants,
                   mean, se, Quantile(hop_copy, 0.50), Quantile(hop_copy, 0.90),
                   *std::max_element(hops.begin(), hops.end()),
                   std::accumulate(last.begin(), last.end(), 0.0) / last.size(),
                   Quantile(last_copy, 0.90),
                   *std::max_element(last.begin(), last.end()), inversions,
                   trim_mean, trim_se, TrimmedMean(last_trim, 0.001));
      std::fflush(out);
      std::printf("CONTENTION mode=%s backoff=%d consumers=%d rows=%d per_row=%d "
                  "hop_mean_ns=%.1f se=%.2f hop_trim_ns=%.1f last_mean_ns=%.1f "
                  "inversions=%ld\n",
                  mode, backoff, consumers, rows, consumers / rows, mean, se,
                  trim_mean, std::accumulate(last.begin(), last.end(), 0.0) / last.size(),
                  inversions);
    }
  std::fclose(out);
  std::printf("CONTENTION_DONE device=%s sms=%d resident_ctas=%d rounds=%d "
              "poll_mode=%s tsv=%s\n", prop.name, prop.multiProcessorCount,
              resident, rounds, mode, tsv);
  return 0;
}
