// SPDX-License-Identifier: BSD-3-Clause
// EX-E3 step 0, second mandatory question (R3 §5.1): does a spinning worker
// slow a *computing* worker on the same SM by consuming issue bandwidth?
//
// Round two measured contention between pollers and found it zero (F-145).
// That is a different question from this one: it compared pollers against
// pollers, so a cost paid by the compute co-resident on the SM was outside its
// frame.  It matters here because dropping the backoff is only free if the
// warp slots a spinner burns are slots nothing else wanted.  §8.3 already says
// the backoff is a performance rule and not a correctness one, and gives
// "compact polling can saturate the memory subsystem and slow the producer" as
// the reason -- stated there, measured here.
//
// Shape: 2 CTAs per SM, the low half of the grid computes and the high half
// waits.  Even/odd was the first attempt and paired nothing: with 256 blocks on
// 128 SMs the scheduler puts block s and block s+128 on the same SM, so every
// SM received two blocks of the same parity (measured, paired_fraction 0.0000).
// Splitting at the midpoint makes s and s+128 a computer and a waiter.  Waiters
// hold until every compute block has finished, so a compute block is under
// interference for its whole run.  Co-residency is not assumed: each block
// reports its %smid and the pairing is reconstructed offline, so a launch the
// scheduler laid out differently shows up in the histogram instead of
// corrupting the answer.
//
// Two compute shapes, because they contend for different things.  `fma` is
// eight independent FMA chains: a dependent chain would measure latency, which
// a co-resident spinner cannot affect, while eight at once is issue-bound.
// `mem` streams a per-block slice of a 64 MiB buffer, because §8.3's claim is
// about the *memory subsystem* and an FMA loop never touches it -- the first
// pass measured only `fma`, found exactly zero, and so had not tested the
// stated mechanism at all.  clock64 is read on one SM either side of the loop,
// so no cross-SM calibration is involved.
#include <tilemega/Codegen/tasks/EventSync.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do { cudaError_t e = (expr); \
  if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, \
    __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)

namespace {

__device__ inline unsigned int SmId() {
  unsigned int id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

enum : int { kIdle = 0, kSpin = 1, kBackoff = 2 };

enum : int { kFma = 0, kMem = 1 };

__global__ void Interfere(unsigned long long* flag, unsigned long long* done,
                          unsigned int* smid, unsigned long long* cycles,
                          float* sink, float const* buf, long long words,
                          int compute_blocks, int iters, int waiter_mode,
                          int backoff_ns, int compute_mode) {
  bool const compute = blockIdx.x < static_cast<unsigned>(compute_blocks);
  if (threadIdx.x == 0) smid[blockIdx.x] = SmId();

  if (!compute) {
    // The control arm leaves the SM to the compute block: same launch shape,
    // same occupancy accounting, no instructions issued during the measurement.
    if (waiter_mode == kIdle) return;
    if (threadIdx.x == 0) {
      unsigned long long const need = static_cast<unsigned long long>(compute_blocks);
      if (waiter_mode == kSpin) {
        while (::tilemega::codegen::EventPoll(flag) < need) {}
      } else {
        while (::tilemega::codegen::EventPoll(flag) < need) __nanosleep(backoff_ns);
      }
    }
    __syncthreads();
    return;
  }

  __syncthreads();
  unsigned long long const begin = clock64();
  float a0 = threadIdx.x * 1.0f, a1 = 1.5f, a2 = 2.5f, a3 = 3.5f;
  float a4 = 4.5f, a5 = 5.5f, a6 = 6.5f, a7 = 7.5f;
  float const k = 1.0000001f;
  if (compute_mode == kFma) {
    for (int i = 0; i < iters; ++i) {
      a0 = fmaf(a0, k, 1.0f); a1 = fmaf(a1, k, 1.0f);
      a2 = fmaf(a2, k, 1.0f); a3 = fmaf(a3, k, 1.0f);
      a4 = fmaf(a4, k, 1.0f); a5 = fmaf(a5, k, 1.0f);
      a6 = fmaf(a6, k, 1.0f); a7 = fmaf(a7, k, 1.0f);
    }
  } else {
    // Coalesced streaming reads over a slice this block owns, so the arms
    // differ only in the waiter and not in which lines they fight over.
    long long const chunk = words / compute_blocks;
    float const* p = buf + static_cast<long long>(blockIdx.x) * chunk;
    int const passes = iters > 0 ? iters : 1;
    for (int r = 0; r < passes; ++r)
      for (long long i = threadIdx.x; i < chunk; i += blockDim.x)
        a0 = fmaf(p[i], k, a0);
  }
  unsigned long long const end = clock64();
  // The sink keeps the loop alive; without it the whole measurement folds away.
  if (threadIdx.x == 0) {
    sink[blockIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    cycles[blockIdx.x] = end - begin;
    __threadfence();
    atomicAdd(flag, 1ull);
    atomicAdd(done, 1ull);
  }
}

double Median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  int const rounds = argc > 1 ? std::atoi(argv[1]) : 41;
  int const iters = argc > 2 ? std::atoi(argv[2]) : 20000;
  char const* tsv = argc > 3 ? argv[3] : "spin_interference.tsv";
  int const threads = 256;  // the megakernel's block size

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int max_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
      reinterpret_cast<void const*>(Interfere), threads, 0));
  if (max_blocks < 2) {
    std::fprintf(stderr, "occupancy is %d CTA/SM; this test needs 2\n", max_blocks);
    return 2;
  }
  int const blocks = 2 * prop.multiProcessorCount;
  int const compute_blocks = blocks / 2;

  unsigned long long *d_flag = nullptr, *d_done = nullptr, *d_cycles = nullptr;
  unsigned int* d_smid = nullptr;
  float* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_flag, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_done, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_cycles, blocks * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_smid, blocks * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&d_sink, blocks * sizeof(float)));
  long long const words = 64ll << 18;  // 64 MiB of float, 512 KiB per block
  float* d_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, words * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_buf, 0x3f, words * sizeof(float)));

  std::FILE* out = std::fopen(tsv, "w");
  if (!out) { std::fprintf(stderr, "cannot write %s\n", tsv); return 2; }
  std::fprintf(out, "compute_mode\tarm\tbackoff_ns\trounds\tblocks\tcompute_blocks\titers\t"
                    "paired_fraction\tcompute_p50_cycles\tcompute_p90_cycles\t"
                    "compute_mean_cycles\tsamples\n");

  struct Arm { char const* name; int mode; int backoff; };
  Arm const arms[] = {{"idle", kIdle, 0}, {"spin", kSpin, 0}, {"backoff64", kBackoff, 64}};
  struct Shape { char const* name; int mode; int iters; };
  // The mem shape needs far fewer passes to run as long: one pass already
  // reads 512 KiB per block.
  Shape const shapes[] = {{"fma", kFma, iters}, {"mem", kMem, std::max(1, iters / 512)}};

  for (Shape const& shape : shapes) {
  double baseline = 0.0;
  for (Arm const& arm : arms) {
    std::vector<double> paired;   // compute blocks sharing an SM with a waiter
    long long pairs = 0, total = 0;
    for (int r = 0; r < rounds; ++r) {
      CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(unsigned long long)));
      CUDA_CHECK(cudaMemset(d_done, 0, sizeof(unsigned long long)));
      CUDA_CHECK(cudaMemset(d_cycles, 0, blocks * sizeof(unsigned long long)));
      Interfere<<<blocks, threads>>>(d_flag, d_done, d_smid, d_cycles, d_sink,
                                     d_buf, words, compute_blocks, shape.iters,
                                     arm.mode, arm.backoff, shape.mode);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());

      std::vector<unsigned long long> cycles(blocks);
      std::vector<unsigned int> smid(blocks);
      CUDA_CHECK(cudaMemcpy(cycles.data(), d_cycles,
                            blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(smid.data(), d_smid, blocks * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));
      if (r == 0) continue;  // warm-up round, discarded whole

      std::map<unsigned int, int> waiters;
      for (int b = compute_blocks; b < blocks; ++b) ++waiters[smid[b]];
      for (int b = 0; b < compute_blocks; ++b) {
        ++total;
        if (waiters.count(smid[b])) {
          ++pairs;
          paired.push_back(static_cast<double>(cycles[b]));
        }
      }
    }
    double const p50 = Median(paired);
    std::vector<double> sorted = paired;
    std::sort(sorted.begin(), sorted.end());
    double const p90 = sorted.empty() ? 0.0 : sorted[static_cast<std::size_t>(0.9 * (sorted.size() - 1))];
    double sum = 0.0;
    for (double v : paired) sum += v;
    double const mean = paired.empty() ? 0.0 : sum / paired.size();
    double const frac = total ? static_cast<double>(pairs) / static_cast<double>(total) : 0.0;
    std::fprintf(out, "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%.4f\t%.1f\t%.1f\t%.1f\t%zu\n",
                 shape.name, arm.name, arm.backoff, rounds, blocks,
                 compute_blocks, shape.iters, frac, p50, p90, mean, paired.size());
    std::fflush(out);
    if (arm.mode == kIdle) baseline = p50;
    std::printf("SPIN_INTERFERENCE compute=%-4s arm=%-9s paired_fraction=%.4f "
                "compute_p50_cycles=%.1f ratio_to_idle=%.4f samples=%zu\n",
                shape.name, arm.name, frac, p50,
                baseline > 0.0 ? p50 / baseline : 0.0, paired.size());
  }
  }
  std::fclose(out);
  CUDA_CHECK(cudaFree(d_flag)); CUDA_CHECK(cudaFree(d_done));
  CUDA_CHECK(cudaFree(d_cycles)); CUDA_CHECK(cudaFree(d_smid));
  CUDA_CHECK(cudaFree(d_sink)); CUDA_CHECK(cudaFree(d_buf));
  std::printf("SPIN_INTERFERENCE_DONE device=%s sms=%d occupancy=%d blocks=%d "
              "tsv=%s\n", prop.name, prop.multiProcessorCount, max_blocks, blocks, tsv);
  return 0;
}
