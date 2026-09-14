// SPDX-License-Identifier: BSD-3-Clause
// EX-E3 step 0 (R3 §5.1): hop_ns as a function of the wait policy, not just of
// the contention geometry.
//
// F-145 fitted one hop curve for the wait the generator emits verbatim --
// `while (EventPoll(ev) < need) __nanosleep(64)` -- and found 910 of its
// 1235 ns to be the backoff granularity rather than coherence traffic.  That
// makes the backoff a parameter worth calibrating, so this sweep re-measures
// the same primitive across a policy grid: pure spin, the status quo, and the
// graded forms in between.  The contention axis is kept, reduced, because the
// policy is what is under test and F-145 already showed the surface is flat in
// N and R; if that flatness breaks under a different policy the fit will say so.
//
// `GradedWait` below is written as it will appear in the runtime header, so the
// policy that gets calibrated here is the policy that gets executed.  The
// `literal64` arm is the verbatim generated loop with an immediate operand, and
// exists to check that routing the same policy through a register does not move
// the number.
//
// Everything about the harness -- the padded row, the sense-reversing barrier,
// the per-round phase dither, the 0.1% trimmed mean, the kept inversions -- is
// carried over from SIMULATOR/contention.cu, whose header explains why each is
// needed.  That file is frozen this round (R3 H1), hence the copy.
#include <tilemega/Codegen/tasks/EventSync.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

struct alignas(128) Row {
  unsigned long long epoch;
  unsigned char padding[120];
};

/// The graded wait EX-E3 step 0 proposes.  `backoff_ns = 0` is a pure spin
/// whatever `spin_iters` says, because the sleep is the only thing that yields;
/// `spin_iters = 0, backoff_ns = 64, grow = 1` is today's generated wait.
__device__ inline void GradedWait(unsigned long long* ev, unsigned long long need,
                                  int spin_iters, int backoff_ns, int grow, int cap) {
  int spun = 0;
  int nap = backoff_ns;
  while (::tilemega::codegen::EventPoll(ev) < need) {
    if (spun < spin_iters) { ++spun; continue; }
    if (nap > 0) {
      __nanosleep(nap);
      if (grow > 1) { nap *= grow; if (nap > cap) nap = cap; }
    }
  }
}

__global__ void Contend(Row* rows_data, unsigned long long* publish_ns,
                        unsigned long long* observe_ns, unsigned int* count,
                        unsigned int* sense, unsigned int participants,
                        int rows, int consumers, int rounds, int guard_cycles,
                        int spin_iters, int backoff_ns, int grow, int cap,
                        int literal) {
  if (threadIdx.x != 0) return;
  unsigned int local_sense = 0;
  bool const publisher = blockIdx.x < static_cast<unsigned int>(rows);
  int const row = publisher ? static_cast<int>(blockIdx.x)
                            : (static_cast<int>(blockIdx.x) - rows) % rows;
  int const consumer = publisher ? -1 : static_cast<int>(blockIdx.x) - rows;

  for (int r = 0; r < rounds; ++r) {
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
      unsigned int const dither = (static_cast<unsigned int>(r) * 2654435761u) % 4096u;
      long long const until = clock64() + guard_cycles + dither;
      while (clock64() < until) { /* let the consumers settle into the poll */ }
      publish_ns[static_cast<std::size_t>(r) * rows + row] = Now();
      __threadfence();
      atomicExch(&rows_data[row].epoch, need);
    } else {
      if (literal) {
        // The generated wait, immediate operand, unchanged from Codegen.cpp.
        while (::tilemega::codegen::EventPoll(&rows_data[row].epoch) < need)
          __nanosleep(64);
      } else {
        GradedWait(&rows_data[row].epoch, need, spin_iters, backoff_ns, grow, cap);
      }
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

double TrimmedMean(std::vector<double>& v, double frac) {
  std::sort(v.begin(), v.end());
  std::size_t const drop = static_cast<std::size_t>(v.size() * frac);
  std::size_t const keep = v.size() > drop ? v.size() - drop : v.size();
  double sum = 0.0;
  for (std::size_t i = 0; i < keep; ++i) sum += v[i];
  return sum / static_cast<double>(keep);
}

struct Policy {
  char const* name;
  int spin_iters, backoff_ns, grow, cap, literal;
};

}  // namespace

int main(int argc, char** argv) {
  int const rounds = argc > 1 ? std::atoi(argv[1]) : 4096;
  int const guard = argc > 2 ? std::atoi(argv[2]) : 20000;
  char const* tsv = argc > 3 ? argv[3] : "backoff.tsv";
  // Arm order is a parameter because the first pass ran the arms in one fixed
  // order; a policy effect and a clock or warm-up drift across the sweep look
  // the same until the order moves under them.
  int const rotate = argc > 4 ? std::atoi(argv[4]) : 0;
  int const warmup = 64;

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int max_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
      reinterpret_cast<void const*>(Contend), 32, 0));
  int const resident = max_blocks * prop.multiProcessorCount;

  std::FILE* out = std::fopen(tsv, "w");
  if (!out) { std::fprintf(stderr, "cannot write %s\n", tsv); return 2; }
  std::fprintf(out, "rotation\tarm\tpoll_mode\tspin_iters\tbackoff_ns\tgrow\tcap_ns\t"
                    "consumers\trows\tper_row\trounds\tparticipants\t"
                    "hop_mean_ns\thop_se_ns\thop_p50_ns\thop_p90_ns\thop_max_ns\t"
                    "last_mean_ns\tlast_p90_ns\tlast_max_ns\tinversions\t"
                    "hop_trim_mean_ns\thop_trim_se_ns\tlast_trim_mean_ns\n");

  char const* mode = TILEMEGA_EVENT_LOAD_POLL ? "load" : "rmw";
  // Pure spin, the status quo, and four graded points between them.  `literal64`
  // and `bo64` are the same policy reached two ways and must agree.
  Policy const policies[] = {
    {"literal64",     0,  64, 1,   64, 1},
    {"spin",          0,   0, 1,    0, 0},
    {"bo64",          0,  64, 1,   64, 0},
    {"bo16",          0,  16, 1,   16, 0},
    {"spin64_bo64",  64,  64, 1,   64, 0},
    {"spin256_bo64",256,  64, 1,   64, 0},
    {"spin64_bo16",  64,  16, 1,   16, 0},
    {"grow16_1024",  64,  16, 2, 1024, 0},
  };
  int const consumer_set[] = {1, 16, 64, 256};
  int const row_set[] = {1, 16};

  int const policy_count = static_cast<int>(sizeof(policies) / sizeof(policies[0]));
  for (int pi = 0; pi < policy_count; ++pi)
  for (int rows : row_set)
    for (int consumers : consumer_set) {
      Policy const& pol = policies[(pi + rotate) % policy_count];
      if (consumers < rows) continue;
      unsigned int const participants = static_cast<unsigned int>(rows + consumers);
      if (static_cast<int>(participants) > resident) {
        std::fprintf(stderr, "skip %s consumers=%d rows=%d: %u CTAs exceed the "
                             "resident %d\n", pol.name, consumers, rows,
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
                                    rounds + warmup, guard, pol.spin_iters,
                                    pol.backoff_ns, pol.grow, pol.cap, pol.literal);
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
      double const se = std::sqrt(variance / n);
      std::vector<double> hop_copy = hops, last_copy = last;
      std::vector<double> hop_trim = hops, last_trim = last;
      double const trim_mean = TrimmedMean(hop_trim, 0.001);
      double trim_var = 0.0;
      std::size_t const trim_keep =
          hop_trim.size() - static_cast<std::size_t>(hop_trim.size() * 0.001);
      for (std::size_t i = 0; i < trim_keep; ++i) {
        double const e = hop_trim[i] - trim_mean;
        trim_var += e * e;
      }
      double const trim_se = std::sqrt(trim_var / static_cast<double>(trim_keep) /
                                       static_cast<double>(trim_keep));
      std::fprintf(out, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%u\t"
                        "%.1f\t%.2f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%ld\t"
                        "%.1f\t%.2f\t%.1f\n",
                   rotate, pol.name, mode, pol.spin_iters, pol.backoff_ns, pol.grow,
                   pol.cap, consumers, rows, consumers / rows, rounds, participants,
                   mean, se, Quantile(hop_copy, 0.50), Quantile(hop_copy, 0.90),
                   *std::max_element(hops.begin(), hops.end()),
                   std::accumulate(last.begin(), last.end(), 0.0) / last.size(),
                   Quantile(last_copy, 0.90),
                   *std::max_element(last.begin(), last.end()), inversions,
                   trim_mean, trim_se, TrimmedMean(last_trim, 0.001));
      std::fflush(out);
      std::printf("BACKOFF arm=%-13s mode=%s consumers=%3d rows=%2d "
                  "hop_mean_ns=%7.1f se=%5.2f hop_trim_ns=%7.1f inversions=%ld\n",
                  pol.name, mode, consumers, rows, mean, se, trim_mean, inversions);
    }
  std::fclose(out);
  std::printf("BACKOFF_DONE device=%s sms=%d resident_ctas=%d rounds=%d "
              "poll_mode=%s rotation=%d tsv=%s\n", prop.name,
              prop.multiProcessorCount, resident, rounds, mode, rotate, tsv);
  return 0;
}
