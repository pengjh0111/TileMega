// SPDX-License-Identifier: BSD-3-Clause
//
// P4.1 microbenchmarks: pipeline rates and synchronization latencies.
//
// Two rules the whole file obeys.  Every rate is measured against a kernel
// whose result is consumed, so nothing here is dead code the compiler may
// delete; and every latency is a *differential* against the same loop without
// the primitive, so the loop's own bookkeeping never lands inside the number.
#include <tilemega/Target/Calibration.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::calib {
namespace {

void CheckCuda(cudaError_t status, char const* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("calibrate: ") + operation + ": " +
                             cudaGetErrorString(status));
  }
}

struct Stat {
  double median = 0.0;
  double rel_stddev = 0.0;
  int samples = 0;
};

Stat Summarize(std::vector<double> values) {
  Stat stat;
  stat.samples = static_cast<int>(values.size());
  if (values.empty()) return stat;
  std::sort(values.begin(), values.end());
  stat.median = values[values.size() / 2];
  double mean = 0.0;
  for (double v : values) mean += v;
  mean /= static_cast<double>(values.size());
  double variance = 0.0;
  for (double v : values) variance += (v - mean) * (v - mean);
  variance /= static_cast<double>(values.size());
  stat.rel_stddev = mean != 0.0 ? std::sqrt(variance) / std::fabs(mean) : 0.0;
  return stat;
}

/// Median wall time in milliseconds of `launch`, over `repeats` timed runs
/// preceded by two untimed warmups.  Warmups matter here: the first launch of
/// a kernel pays module load and the clock has not yet ramped.
template <class Launch>
Stat TimeMs(int repeats, Launch&& launch) {
  cudaEvent_t start, stop;
  CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
  CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");
  for (int i = 0; i < 2; ++i) launch();
  CheckCuda(cudaDeviceSynchronize(), "warmup");
  std::vector<double> samples;
  for (int i = 0; i < repeats; ++i) {
    CheckCuda(cudaEventRecord(start), "cudaEventRecord");
    launch();
    CheckCuda(cudaEventRecord(stop), "cudaEventRecord");
    CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    samples.push_back(static_cast<double>(ms));
  }
  CheckCuda(cudaGetLastError(), "kernel launch");
  CheckCuda(cudaEventDestroy(start), "cudaEventDestroy");
  CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy");
  return Summarize(std::move(samples));
}

void Record(TargetSpec& spec, char const* name, double value, char const* unit,
            Stat const& stat, char const* method) {
  TargetSpec::Measurement record;
  record.name = name;
  record.value = value;
  record.unit = unit;
  record.samples = stat.samples;
  record.rel_stddev = stat.rel_stddev;
  record.method = method;
  spec.calib.measurements.push_back(std::move(record));
}

constexpr int kThreads = 256;
constexpr int kChains = 8;

// ---------------------------------------------------------------- pipelines

__global__ __launch_bounds__(kThreads) void FfmaKernel(float* sink, int iters) {
  float acc[kChains];
#pragma unroll
  for (int i = 0; i < kChains; ++i)
    acc[i] = static_cast<float>(threadIdx.x + i) * 1e-3f;
  float const scale = 1.0000001f;
  float const bias = 0.9999999f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < kChains; ++i) acc[i] = fmaf(acc[i], scale, bias);
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kChains; ++i) sum += acc[i];
  // The guard is never true at run time but is opaque to the compiler, so the
  // whole chain stays live without a store on the timed path.
  if (sum == 1.0e30f) sink[blockIdx.x] = sum;
}

/// The IMAD chain is emitted as inline PTX on purpose.  Written in C, an
/// affine chain `acc = acc * s + b` with compile-time `s` and `b` folds two
/// iterations into one (`acc * s^2 + (b * s + b)`) under unrolling, and the
/// kernel then reports several times the hardware's IMAD rate.
__global__ __launch_bounds__(kThreads) void ImadKernel(int* sink, int iters) {
  int acc[kChains];
#pragma unroll
  for (int i = 0; i < kChains; ++i) acc[i] = static_cast<int>(threadIdx.x) + i;
  int const scale = 1103515245;
  int const bias = 12345;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < kChains; ++i)
      asm volatile("mad.lo.s32 %0, %0, %1, %2;"
                   : "+r"(acc[i])
                   : "r"(scale), "r"(bias));
  }
  int sum = 0;
#pragma unroll
  for (int i = 0; i < kChains; ++i) sum += acc[i];
  if (sum == 0x7fffffff) sink[blockIdx.x] = sum;
}

/// `Op` 0 = ex2.approx.f32, 1 = rsqrt.approx.f32.  Both are single MUFU
/// instructions, so the loop measures SFU issue rate rather than a libm
/// expansion.
template <int Op>
__global__ __launch_bounds__(kThreads) void SfuKernel(float* sink, int iters) {
  float acc[kChains];
#pragma unroll
  for (int i = 0; i < kChains; ++i)
    acc[i] = 1.0f + static_cast<float>(threadIdx.x + i) * 1e-3f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < kChains; ++i) {
      float x = acc[i];
      if (Op == 0) asm volatile("ex2.approx.f32 %0, %1;" : "=f"(x) : "f"(x));
      else asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(x) : "f"(x));
      acc[i] = x * 0.5f + 1.0f;
    }
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kChains; ++i) sum += acc[i];
  if (sum == 1.0e30f) sink[blockIdx.x] = sum;
}

/// One warp-wide mma.m16n8k16 per chain.  The accumulator feeds back into the
/// next iteration's C operand, which is how a real mainloop uses it.
__global__ __launch_bounds__(kThreads) void MmaKernel(float* sink, int iters) {
  std::uint32_t a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
  std::uint32_t b[2] = {0x3c003c00u, 0x3c003c00u};
  float c[4][4];
#pragma unroll
  for (int chain = 0; chain < 4; ++chain)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[chain][i] = static_cast<float>(threadIdx.x + i);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int chain = 0; chain < 4; ++chain) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[chain][0]), "+f"(c[chain][1]), "+f"(c[chain][2]),
            "+f"(c[chain][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
  }
  float sum = 0.0f;
#pragma unroll
  for (int chain = 0; chain < 4; ++chain)
#pragma unroll
    for (int i = 0; i < 4; ++i) sum += c[chain][i];
  if (sum == 1.0e30f) sink[blockIdx.x] = sum;
}

constexpr int kSmemFloats = 1024;  ///< 4 KiB: 32 banks x 32 rows

/// `Ways` selects the bank-conflict degree: lane `l` addresses bank
/// `(l * Ways) % 32`, so `Ways` lanes of every warp collide.  Adding multiples
/// of 32 inside the unrolled body keeps the bank fixed while changing the
/// address, so the conflict degree stays constant.
///
/// The load is inline PTX because consecutive iterations read overlapping
/// address windows: written in C, the compiler unrolls and reuses seven of
/// every eight values, and the kernel reports well above the 32-bank limit.
template <int Ways>
__global__ __launch_bounds__(kThreads) void SmemKernel(float* sink, int iters) {
  __shared__ float buffer[kSmemFloats];
  for (int i = static_cast<int>(threadIdx.x); i < kSmemFloats; i += kThreads)
    buffer[i] = static_cast<float>(i);
  __syncthreads();
  int index = (static_cast<int>(threadIdx.x) * Ways) & (kSmemFloats - 1);
  float sum = 0.0f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < kChains; ++u) {
      unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(
          &buffer[(index + u * 32) & (kSmemFloats - 1)]));
      float value;
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(value) : "r"(address));
      sum += value;
    }
    index = (index + 32) & (kSmemFloats - 1);
  }
  if (sum == 1.0e30f) sink[blockIdx.x] = sum;
}

/// Fills the stream buffer with a per-element hash mapped into [1, 2).  A
/// zero-filled buffer would let the memory subsystem's lossless compression
/// answer part of the traffic, which shows up as a DRAM rate above the pin
/// bandwidth.
__global__ __launch_bounds__(kThreads) void FillKernel(float4* data,
                                                       std::size_t elements) {
  std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       i < elements; i += stride) {
    unsigned h = static_cast<unsigned>(i) * 2654435761u;
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      h = h * 1664525u + 1013904223u;
      v[k] = __uint_as_float((h & 0x007fffffu) | 0x3f800000u);
    }
    data[i] = make_float4(v[0], v[1], v[2], v[3]);
  }
}

/// Dependent-load latency: a single thread walks a permutation whose stride is
/// larger than a page-worth of lines, so neither the prefetcher nor the L1's
/// sectoring can overlap two hops.  One warp, one CTA: the number that comes
/// out is a round trip, not a throughput.
__global__ void ChaseKernel(unsigned const* next, int hops, unsigned* sink) {
  unsigned index = 0;
  for (int i = 0; i < hops; ++i) index = next[index];
  if (threadIdx.x == 0) *sink = index;
}

/// Streaming read with `ld.global.cg`, which caches in L2 and bypasses L1.
/// That is what makes the working-set sweep a measurement of L2 rather than of
/// L1 plus L2: without it, a buffer smaller than the aggregate L1 hides the
/// knee entirely.  `rotate` shifts the block-to-address map every pass so no
/// SM sees the same slice twice in a row.
__global__ __launch_bounds__(kThreads) void StreamReadKernel(
    float4 const* data, std::size_t elements, int passes, int rotate,
    float* sink) {
  std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int pass = 0; pass < passes; ++pass) {
    std::size_t shift =
        (static_cast<std::size_t>(pass) * static_cast<std::size_t>(rotate) *
         static_cast<std::size_t>(blockDim.x)) % elements;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
         i < elements; i += stride) {
      std::size_t j = i + shift;
      if (j >= elements) j -= elements;
      float4 value;
      asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
                   : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
                   : "l"(data + j)
                   : "memory");
      acc.x += value.x; acc.y += value.y; acc.z += value.z; acc.w += value.w;
    }
  }
  float sum = acc.x + acc.y + acc.z + acc.w;
  if (sum == 1.0e30f) sink[blockIdx.x] = sum;
}

// ----------------------------------------------------------------- sync

/// A dependent chain of atomics: the returned value feeds the next address,
/// so the loop measures latency rather than issue rate.  `value >> 63` is
/// always zero but the compiler cannot prove it, which is what keeps the
/// dependency alive without changing the address.
__global__ __launch_bounds__(32) void AtomicKernel(
    unsigned long long* counter, int iters) {
  if (threadIdx.x != 0) return;
  unsigned long long value = 0;
  for (int i = 0; i < iters; ++i)
    value = atomicAdd(counter + (value >> 63), 1ull);
  if (value == ~0ull) counter[1] = value;
}

/// `Mode` 0 = store only (the baseline), 1 = + __threadfence, 2 =
/// + __syncthreads, 3 = + named barrier 1.  Each primitive's cost is the
/// difference against mode 0 at the same grid.
template <int Mode>
__global__ __launch_bounds__(kThreads) void BarrierKernel(unsigned* scratch,
                                                          int iters) {
  // volatile, so mode 0 really performs `iters` stores.  A plain store to a
  // loop-invariant address is dead but for the last iteration, which would
  // leave the baseline empty and charge the store to the primitive.
  volatile unsigned* slot = scratch + blockIdx.x * kThreads + threadIdx.x;
  for (int i = 0; i < iters; ++i) {
    *slot = static_cast<unsigned>(i);
    if (Mode == 1) __threadfence();
    if (Mode == 2) __syncthreads();
    if (Mode == 3) asm volatile("bar.sync 1, %0;" ::"r"(kThreads));
  }
}

/// The megakernel's own stage barrier, transcribed from ModelHarness.cuh's
/// `GridBarrier` with the generated WAIT/NOTIFY macro bodies inlined: release
/// fence, CTA convergence, one monotonic arrival, and either the epoch publish
/// or a backoff poll on it.  It is measured rather than composed from
/// `threadfence_ns + syncthreads_ns + atomic_*` because the poll is a
/// contended global atomic whose cost is set by how many CTAs are spinning,
/// which none of those three carries.
///
/// The grid must be resident or the poll never ends; every caller sizes it
/// from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
/// One barrier's counters, laid out as ModelRuntime.h lays out EventCounter:
/// 128-byte aligned so no two stages share a line.
struct alignas(128) BarrierEvent {
  unsigned long long arrivals;
  unsigned long long epoch;
  unsigned long long pad[14];
};

__global__ __launch_bounds__(kThreads) void GridBarrierKernel(
    BarrierEvent* events, int iters) {
  // One event per iteration, as the megakernel has one per stage: the line is
  // written once and then never touched again.  This was written to test the
  // hypothesis that a cold barrier line explains the megakernel's residual
  // per-stage cost; the hypothesis is FALSIFIED -- the per-barrier cold line
  // measures 1045.9 ns at 128 CTAs against 1.03 us for a single reused hot
  // line, i.e. no difference beyond run-to-run spread.
  // The layout is kept because it is the one the megakernel actually has; the
  // residual is recorded as unexplained in COST_MODEL/result.md.
  for (int i = 0; i < iters; ++i) {
    unsigned long long needed = static_cast<unsigned long long>(gridDim.x);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned long long ticket = atomicAdd(&events[i].arrivals, 1ull);
      if (ticket + 1ull == needed) {
        __threadfence();
        atomicExch(&events[i].epoch, 1ull);
      } else {
        while (atomicAdd(&events[i].epoch, 0ull) < 1ull) __nanosleep(64);
      }
    }
    __syncthreads();
    __threadfence();
  }
}

int ResidentBlocks(TargetSpec const& spec, void const* kernel, int block_size) {
  int per_sm = 0;
  CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel,
                                                          block_size, 0),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  if (per_sm < 1) per_sm = 1;
  return spec.res.num_sms * per_sm;
}

}  // namespace

namespace {

/// Runs the FFMA kernel until the clock has settled.  Without this the first
/// group measured runs at the idle boost clock and the last at the sustained
/// one, which showed up as a 10% spread between otherwise identical runs of
/// the tool.  Everything after this point sees the sustained clock, which is
/// also what the workloads being modelled see.
void WarmUpClocks(TargetSpec const& spec, std::ostream& log) {
  float* sink = nullptr;
  CheckCuda(cudaMalloc(&sink, 1 << 20), "cudaMalloc");
  int const blocks =
      ResidentBlocks(spec, reinterpret_cast<void const*>(FfmaKernel), kThreads);
  auto started = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       started)
             .count() < 1.0) {
    FfmaKernel<<<blocks, kThreads>>>(sink, 65536);
    CheckCuda(cudaDeviceSynchronize(), "warm-up");
  }
  CheckCuda(cudaFree(sink), "cudaFree");
  log << "  clock warm-up: 1.0 s\n";
}

}  // namespace

void MeasurePipelines(TargetSpec& spec, Options const& options,
                      std::ostream& log) {
  float* sink_f = nullptr;
  int* sink_i = nullptr;
  CheckCuda(cudaMalloc(&sink_f, 1 << 20), "cudaMalloc");
  CheckCuda(cudaMalloc(&sink_i, 1 << 20), "cudaMalloc");

  int const blocks =
      ResidentBlocks(spec, reinterpret_cast<void const*>(FfmaKernel), kThreads);
  double const threads = static_cast<double>(blocks) * kThreads;
  int const iters = 4096;

  {
    Stat stat = TimeMs(options.repeats, [&] {
      FfmaKernel<<<blocks, kThreads>>>(sink_f, iters);
    });
    double flops = 2.0 * kChains * iters * threads;
    spec.calib.cuda_fp32_gflops = flops / (stat.median * 1e6);
    Record(spec, "cuda_fp32_gflops", spec.calib.cuda_fp32_gflops, "GFLOP/s",
           stat,
           "8 independent FFMA chains x 4096 iterations at full occupancy; "
           "2 flops per FFMA");
    log << "  cuda_fp32_gflops    = " << spec.calib.cuda_fp32_gflops << '\n';
  }
  {
    Stat stat = TimeMs(options.repeats, [&] {
      ImadKernel<<<blocks, kThreads>>>(sink_i, iters);
    });
    double ops = static_cast<double>(kChains) * iters * threads;
    spec.calib.cuda_int32_gops = ops / (stat.median * 1e6);
    Record(spec, "cuda_int32_gops", spec.calib.cuda_int32_gops, "GIMAD/s", stat,
           "8 independent IMAD chains x 4096 iterations at full occupancy");
    log << "  cuda_int32_gops     = " << spec.calib.cuda_int32_gops << '\n';
  }
  {
    Stat stat = TimeMs(options.repeats, [&] {
      SfuKernel<0><<<blocks, kThreads>>>(sink_f, iters);
    });
    double ops = static_cast<double>(kChains) * iters * threads;
    spec.calib.sfu_exp2_gops = ops / (stat.median * 1e6);
    Record(spec, "sfu_exp2_gops", spec.calib.sfu_exp2_gops, "Gop/s", stat,
           "inline ex2.approx.f32, 8 chains x 4096 iterations");
    log << "  sfu_exp2_gops       = " << spec.calib.sfu_exp2_gops << '\n';
  }
  {
    Stat stat = TimeMs(options.repeats, [&] {
      SfuKernel<1><<<blocks, kThreads>>>(sink_f, iters);
    });
    double ops = static_cast<double>(kChains) * iters * threads;
    spec.calib.sfu_rsqrt_gops = ops / (stat.median * 1e6);
    Record(spec, "sfu_rsqrt_gops", spec.calib.sfu_rsqrt_gops, "Gop/s", stat,
           "inline rsqrt.approx.f32, 8 chains x 4096 iterations");
    log << "  sfu_rsqrt_gops      = " << spec.calib.sfu_rsqrt_gops << '\n';
  }
  {
    int const mma_blocks = ResidentBlocks(
        spec, reinterpret_cast<void const*>(MmaKernel), kThreads);
    Stat stat = TimeMs(options.repeats, [&] {
      MmaKernel<<<mma_blocks, kThreads>>>(sink_f, iters);
    });
    double warps = static_cast<double>(mma_blocks) * (kThreads / 32);
    double flops = 2.0 * 16 * 8 * 16 * 4 * iters * warps;
    spec.calib.tc_fp16_gflops = flops / (stat.median * 1e6);
    Record(spec, "tc_fp16_gflops", spec.calib.tc_fp16_gflops, "GFLOP/s", stat,
           "4 independent mma.sync.m16n8k16.f32.f16.f16.f32 chains x 4096 "
           "iterations; 4096 flops per instruction");
    log << "  tc_fp16_gflops      = " << spec.calib.tc_fp16_gflops << '\n';
  }

  // Shared memory: the conflict-free rate, then the slowdown per conflict way.
  {
    int const smem_blocks = ResidentBlocks(
        spec, reinterpret_cast<void const*>(SmemKernel<1>), kThreads);
    double const smem_threads = static_cast<double>(smem_blocks) * kThreads;
    double const bytes = 4.0 * kChains * iters * smem_threads;
    Stat base = TimeMs(options.repeats, [&] {
      SmemKernel<1><<<smem_blocks, kThreads>>>(sink_f, iters);
    });
    spec.calib.smem_gbps = bytes / (base.median * 1e6);
    Record(spec, "smem_gbps", spec.calib.smem_gbps, "GB/s", base,
           "conflict-free ld.shared, 8 loads per iteration x 4096 iterations");
    log << "  smem_gbps           = " << spec.calib.smem_gbps << '\n';

    // The multiplier is fitted as max(1, slope * ways) rather than as a
    // per-way increment: on this pipeline a 2-way conflict is free (the
    // scalar path does not saturate the banks) and every way beyond that
    // costs its full share, which a straight line through 1.0 cannot express.
    double fit_num = 0.0, fit_den = 0.0, worst_rsd = 0.0;
    int slope_points = 0;
    auto add = [&](int ways, Stat const& stat) {
      double ratio = stat.median / base.median;
      if (ratio > 1.05) {
        fit_num += static_cast<double>(ways) * ratio;
        fit_den += static_cast<double>(ways) * static_cast<double>(ways);
        ++slope_points;
      }
      worst_rsd = std::max(worst_rsd, stat.rel_stddev);
      Stat one = stat;
      Record(spec, ("smem_conflict_ratio_w" + std::to_string(ways)).c_str(),
             ratio, "x conflict-free", one,
             "elapsed of the same loop with a `ways`-way bank conflict, "
             "relative to the conflict-free loop");
      log << "  smem " << ways << "-way conflict -> " << ratio << "x\n";
    };
    add(2, TimeMs(options.repeats, [&] {
          SmemKernel<2><<<smem_blocks, kThreads>>>(sink_f, iters);
        }));
    add(4, TimeMs(options.repeats, [&] {
          SmemKernel<4><<<smem_blocks, kThreads>>>(sink_f, iters);
        }));
    add(8, TimeMs(options.repeats, [&] {
          SmemKernel<8><<<smem_blocks, kThreads>>>(sink_f, iters);
        }));
    add(16, TimeMs(options.repeats, [&] {
          SmemKernel<16><<<smem_blocks, kThreads>>>(sink_f, iters);
        }));
    add(32, TimeMs(options.repeats, [&] {
          SmemKernel<32><<<smem_blocks, kThreads>>>(sink_f, iters);
        }));
    spec.calib.smem_conflict_slope = fit_den > 0.0 ? fit_num / fit_den : 0.0;
    Stat conflict_stat;
    conflict_stat.samples = options.repeats * slope_points;
    conflict_stat.rel_stddev = worst_rsd;
    Record(spec, "smem_conflict_slope", spec.calib.smem_conflict_slope,
           "multiplier per way", conflict_stat,
           "least squares through the origin of t(w)/t(1) against w over the "
           "ways whose ratio exceeds 1.05; the cost model applies "
           "max(1, slope * ways)");
    log << "  smem_conflict_slope = " << spec.calib.smem_conflict_slope << '\n';

    // The same conflict-free loop at a fixed occupancy instead of the
    // resident maximum.  The SIMT f32 mainloop is bound by this pipeline and
    // runs at one or two CTAs per SM, where eight warps cannot cover the
    // ld.shared latency: the measured rate there is a third of the plateau,
    // and using the plateau makes every small tile look three times cheaper
    // than it is.
    double occupancy_rsd = 0.0;
    for (int per_sm : {1, 2, 3, 4, 6}) {
      if (per_sm * kThreads > spec.res.max_threads_per_sm) break;
      int grid = spec.res.num_sms * per_sm;
      double loop_threads = static_cast<double>(grid) * kThreads;
      Stat stat = TimeMs(options.repeats, [&] {
        SmemKernel<1><<<grid, kThreads>>>(sink_f, iters);
      });
      double gbps = 4.0 * kChains * iters * loop_threads / (stat.median * 1e6);
      spec.calib.smem_occupancy_ctas.push_back(per_sm);
      spec.calib.smem_occupancy_gbps.push_back(gbps);
      occupancy_rsd = std::max(occupancy_rsd, stat.rel_stddev);
      log << "  smem " << per_sm << " CTA/SM -> " << gbps << " GB/s\n";
    }
    Stat occupancy_stat;
    occupancy_stat.samples =
        options.repeats *
        static_cast<int>(spec.calib.smem_occupancy_ctas.size());
    occupancy_stat.rel_stddev = occupancy_rsd;
    Record(spec, "smem_occupancy_gbps",
           spec.calib.smem_occupancy_gbps.empty()
               ? 0.0
               : spec.calib.smem_occupancy_gbps.front(),
           "GB/s at 1 CTA/SM", occupancy_stat,
           "the conflict-free ld.shared loop at a fixed CTAs-per-SM instead "
           "of the resident maximum; the recorded value is the 1 CTA/SM point "
           "and the curve carries the rest");
  }

  // Dependent-load latency at three levels of the hierarchy.
  {
    struct Level {
      char const* name;
      std::size_t bytes;   ///< working set
      int hops;            ///< dependent loads per launch
      int repeats;
      double* field;
    };
    // The DRAM point needs a touched-line footprint past the L2 capacity, so
    // it walks 2M lines (256 MiB) of a 512 MiB buffer: fewer hops and the
    // whole path stays L2-resident across repeats and reports the L2 number.
    Level levels[] = {
        {"l1", 16u << 10, 4096, options.repeats, &spec.calib.l1_latency_ns},
        {"l2", 8u << 20, 65536, options.repeats, &spec.calib.l2_latency_ns},
        {"dram", 512u << 20, 2000000, 3, &spec.calib.dram_latency_ns}};
    std::size_t const stride = 32;  ///< 128 B: one cache line per hop
    unsigned* chase = nullptr;
    CheckCuda(cudaMalloc(&chase, levels[2].bytes), "cudaMalloc(chase)");
    std::vector<unsigned> host(levels[2].bytes / sizeof(unsigned), 0u);
    std::vector<unsigned> order;
    std::mt19937 rng(20260903u);
    for (Level const& level : levels) {
      std::size_t const steps = level.bytes / sizeof(unsigned) / stride;
      order.resize(steps);
      for (std::size_t i = 0; i < steps; ++i)
        order[i] = static_cast<unsigned>(i);
      std::shuffle(order.begin(), order.end(), rng);
      // A single cycle through a random permutation: no stride for the
      // prefetcher to lock onto, and every line is visited exactly once.
      for (std::size_t i = 0; i < steps; ++i) {
        host[static_cast<std::size_t>(order[i]) * stride] =
            static_cast<unsigned>(order[(i + 1) % steps] * stride);
      }
      CheckCuda(cudaMemcpy(chase, host.data(), level.bytes,
                           cudaMemcpyHostToDevice),
                "cudaMemcpy(chase)");
      Stat stat = TimeMs(level.repeats, [&] {
        ChaseKernel<<<1, 32>>>(chase, level.hops,
                               reinterpret_cast<unsigned*>(sink_i));
      });
      Stat empty = TimeMs(level.repeats, [&] {
        ChaseKernel<<<1, 32>>>(chase, 0,
                               reinterpret_cast<unsigned*>(sink_i));
      });
      *level.field = (stat.median - empty.median) * 1e6 / level.hops;
      std::string const name = std::string(level.name) + "_latency_ns";
      std::string const method =
          "single-thread pointer chase, random single cycle over the " +
          std::to_string(level.bytes >> 20) + " MiB working set at one " +
          "cache line per hop, " + std::to_string(level.hops) +
          " hops, empty launch subtracted";
      Record(spec, name.c_str(), *level.field, "ns", stat, method.c_str());
      log << "  " << name << std::string(20 - name.size(), ' ') << "= "
          << *level.field << '\n';
    }
    CheckCuda(cudaFree(chase), "cudaFree");
  }

  // L2 / DRAM: bandwidth against working-set size.  The knee is the capacity.
  {
    // The sweep runs past the DRAM floor on purpose: at 512 MiB a 72 MiB L2
    // still answers a seventh of the requests, and dividing requested traffic
    // by elapsed time then reports a DRAM rate above the pin bandwidth.
    std::vector<double> megabytes = {1,  2,  4,  8,   16,  24,  32,   48,   64,
                                     72, 80, 96, 128, 192, 256, 512, 1024, 2048};
    std::size_t const largest =
        static_cast<std::size_t>(megabytes.back()) * (1u << 20);
    float4* data = nullptr;
    CheckCuda(cudaMalloc(&data, largest), "cudaMalloc(stream buffer)");
    int const stream_blocks = ResidentBlocks(
        spec, reinterpret_cast<void const*>(StreamReadKernel), kThreads);
    FillKernel<<<stream_blocks, kThreads>>>(data, largest / sizeof(float4));
    CheckCuda(cudaDeviceSynchronize(), "FillKernel");
    double const traffic_target = 2.0e9;  // ~2 GB per point
    double worst_rsd = 0.0;
    for (double mb : megabytes) {
      std::size_t bytes = static_cast<std::size_t>(mb) * (1u << 20);
      std::size_t elements = bytes / sizeof(float4);
      int passes = static_cast<int>(traffic_target / static_cast<double>(bytes));
      if (passes < 2) passes = 2;
      Stat stat = TimeMs(options.repeats, [&] {
        StreamReadKernel<<<stream_blocks, kThreads>>>(data, elements, passes,
                                                      977, sink_f);
      });
      double moved = static_cast<double>(bytes) * passes;
      double gbps = moved / (stat.median * 1e6);
      spec.calib.l2_curve_bytes.push_back(static_cast<double>(bytes));
      spec.calib.l2_curve_gbps.push_back(gbps);
      worst_rsd = std::max(worst_rsd, stat.rel_stddev);
      log << "  stream " << mb << " MiB -> " << gbps << " GB/s\n";
    }
    CheckCuda(cudaFree(data), "cudaFree");

    // The plateau is the sweep's maximum -- the smallest working sets sit
    // below it for want of parallelism, not for want of cache, so a median
    // over the head would understate it.  The knee is then the largest
    // working set still holding 95% of that plateau; taking the midpoint of
    // the fall instead names a size the cache never had.
    std::vector<double>& curve = spec.calib.l2_curve_gbps;
    std::size_t peak =
        std::max_element(curve.begin(), curve.end()) - curve.begin();
    spec.calib.l2_gbps = curve[peak];
    std::vector<double> tail(curve.end() - 3, curve.end());
    std::sort(tail.begin(), tail.end());
    spec.calib.dram_gbps = tail[tail.size() / 2];
    spec.calib.l2_knee_bytes = spec.calib.l2_curve_bytes[peak];
    for (std::size_t i = peak + 1; i < curve.size(); ++i) {
      if (curve[i] < 0.95 * spec.calib.l2_gbps) break;
      spec.calib.l2_knee_bytes = spec.calib.l2_curve_bytes[i];
    }
    Stat curve_stat;
    curve_stat.samples = options.repeats * static_cast<int>(megabytes.size());
    curve_stat.rel_stddev = worst_rsd;
    Record(spec, "l2_gbps", spec.calib.l2_gbps, "GB/s", curve_stat,
           "ld.global.cg streaming read, plateau (maximum) of the "
           "working-set sweep");
    Record(spec, "dram_gbps", spec.calib.dram_gbps, "GB/s", curve_stat,
           "same sweep, median of the 512-2048 MiB working sets (achieved, "
           "not peak)");
    Record(spec, "l2_knee_bytes", spec.calib.l2_knee_bytes, "bytes", curve_stat,
           "largest working set of the sweep still within 95% of the "
           "plateau bandwidth");
    log << "  l2_gbps             = " << spec.calib.l2_gbps << '\n'
        << "  dram_gbps           = " << spec.calib.dram_gbps << '\n'
        << "  l2_knee_bytes       = " << spec.calib.l2_knee_bytes << '\n';
  }

  CheckCuda(cudaFree(sink_f), "cudaFree");
  CheckCuda(cudaFree(sink_i), "cudaFree");
}

void MeasureSync(TargetSpec& spec, Options const& options, std::ostream& log) {
  unsigned long long* counter = nullptr;
  CheckCuda(cudaMalloc(&counter, 2 * sizeof(unsigned long long)), "cudaMalloc");
  CheckCuda(cudaMemset(counter, 0, 2 * sizeof(unsigned long long)), "cudaMemset");

  int const iters = 2048;
  {
    Stat stat = TimeMs(options.repeats,
                       [&] { AtomicKernel<<<1, 32>>>(counter, iters); });
    spec.calib.atomic_uncontended_ns = stat.median * 1e6 / iters;
    Record(spec, "atomic_uncontended_ns", spec.calib.atomic_uncontended_ns, "ns",
           stat,
           "one CTA, dependent chain of 2048 global atomicAdd; the return "
           "value feeds the next address so this is latency, not issue rate");
    log << "  atomic_uncontended_ns = " << spec.calib.atomic_uncontended_ns
        << '\n';
  }
  {
    double worst_rsd = 0.0;
    for (int blocks : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}) {
      Stat stat = TimeMs(options.repeats,
                         [&] { AtomicKernel<<<blocks, 32>>>(counter, iters); });
      double per_op = stat.median * 1e6 / (static_cast<double>(blocks) * iters);
      spec.calib.atomic_contention_ctas.push_back(blocks);
      spec.calib.atomic_contention_ns.push_back(per_op);
      worst_rsd = std::max(worst_rsd, stat.rel_stddev);
      log << "  atomic " << blocks << " CTAs -> " << per_op << " ns/op\n";
    }
    Stat curve_stat;
    curve_stat.samples = options.repeats * 10;
    curve_stat.rel_stddev = worst_rsd;
    Record(spec, "atomic_contention_ns", spec.calib.atomic_contention_ns.back(),
           "ns per completed atomic", curve_stat,
           "N CTAs hammering one address; the recorded value is inverse "
           "throughput (elapsed / N / iters), which is what "
           "T_sync = |image(C_kappa)| * latency multiplies");
  }

  unsigned* scratch = nullptr;
  int const barrier_blocks = ResidentBlocks(
      spec, reinterpret_cast<void const*>(BarrierKernel<0>), kThreads);
  CheckCuda(cudaMalloc(&scratch, static_cast<std::size_t>(barrier_blocks) *
                                     kThreads * sizeof(unsigned)),
            "cudaMalloc");
  Stat base = TimeMs(options.repeats, [&] {
    BarrierKernel<0><<<barrier_blocks, kThreads>>>(scratch, iters);
  });
  auto differential = [&](Stat const& stat) {
    return (stat.median - base.median) * 1e6 / iters;
  };
  {
    Stat stat = TimeMs(options.repeats, [&] {
      BarrierKernel<1><<<barrier_blocks, kThreads>>>(scratch, iters);
    });
    spec.calib.threadfence_ns = differential(stat);
    Record(spec, "threadfence_ns", spec.calib.threadfence_ns, "ns", stat,
           "store + __threadfence loop at full occupancy, minus the same loop "
           "without the fence");
    log << "  threadfence_ns      = " << spec.calib.threadfence_ns << '\n';
  }
  {
    Stat stat = TimeMs(options.repeats, [&] {
      BarrierKernel<2><<<barrier_blocks, kThreads>>>(scratch, iters);
    });
    spec.calib.syncthreads_ns = differential(stat);
    Record(spec, "syncthreads_ns", spec.calib.syncthreads_ns, "ns", stat,
           "store + __syncthreads loop at full occupancy, minus the same loop "
           "without the barrier; 256 threads per CTA");
    log << "  syncthreads_ns      = " << spec.calib.syncthreads_ns << '\n';
  }
  {
    Stat stat = TimeMs(options.repeats, [&] {
      BarrierKernel<3><<<barrier_blocks, kThreads>>>(scratch, iters);
    });
    spec.calib.named_barrier_ns = differential(stat);
    Record(spec, "named_barrier_ns", spec.calib.named_barrier_ns, "ns", stat,
           "store + bar.sync 1 loop at full occupancy, minus the same loop "
           "without the barrier; 256 threads per CTA");
    log << "  named_barrier_ns    = " << spec.calib.named_barrier_ns << '\n';
  }

  // The composite the megakernel actually pays once per stage.
  {
    int const barrier_iters = 512;
    BarrierEvent* events = nullptr;
    std::size_t const events_bytes = barrier_iters * sizeof(BarrierEvent);
    CheckCuda(cudaMalloc(&events, events_bytes), "cudaMalloc");
    int const resident = ResidentBlocks(
        spec, reinterpret_cast<void const*>(GridBarrierKernel), kThreads);
    int const per_sm = resident / spec.res.num_sms;
    double worst_rsd = 0.0;
    for (int step = 1; step <= per_sm; ++step) {
      int grid = spec.res.num_sms * step;
      Stat stat = TimeMs(options.repeats, [&] {
        // Zeroed per launch: each event is consumed once, so a second launch
        // needs a fresh set.
        cudaMemsetAsync(events, 0, events_bytes);
        GridBarrierKernel<<<grid, kThreads>>>(events, barrier_iters);
      });
      double per_barrier = stat.median * 1e6 / barrier_iters;
      spec.calib.grid_barrier_ctas.push_back(grid);
      spec.calib.grid_barrier_ns.push_back(per_barrier);
      worst_rsd = std::max(worst_rsd, stat.rel_stddev);
      log << "  grid barrier " << grid << " CTAs -> " << per_barrier << " ns\n";
    }
    Stat curve_stat;
    curve_stat.samples =
        options.repeats * static_cast<int>(spec.calib.grid_barrier_ctas.size());
    curve_stat.rel_stddev = worst_rsd;
    Record(spec, "grid_barrier_ns",
           spec.calib.grid_barrier_ns.empty() ? 0.0
                                              : spec.calib.grid_barrier_ns.back(),
           "ns per barrier", curve_stat,
           "512 back-to-back ModelHarness GridBarriers over a resident grid, "
           "one 128-byte-aligned event per barrier, swept over CTAs per SM; "
           "the recorded value is the widest grid and the curve carries the "
           "rest");
    CheckCuda(cudaFree(events), "cudaFree");
  }

  // cluster.sync() needs a cluster launch, which this target may not have.
  // Leaving the field at zero with `cluster_sync_calibrated = false` is the
  // honest answer; a plausible-looking number here would be a fabrication.
  spec.calib.cluster_sync_calibrated = false;
  spec.calib.cluster_sync_ns = 0.0;
  if (!spec.caps.cluster) {
    log << "  cluster_sync_ns     = uncalibrated (target has no clusters)\n";
  } else {
    log << "  cluster_sync_ns     = uncalibrated (needs "
           "docs/experiments/CLUSTER/run_on_h100.sh on the cluster target)\n";
  }

  CheckCuda(cudaFree(scratch), "cudaFree");
  CheckCuda(cudaFree(counter), "cudaFree");
}

/// Was this GPU ours alone for the run?
///
/// A rented GPU is often shared, and a co-tenant in another container is
/// invisible: nvidia-smi reports 100% utilisation, 425 W and "no running
/// processes found".  Two runs out of five on this machine recorded half the
/// DRAM bandwidth, half the tensor-core rate and a third of the L2 capacity
/// while the FMA rate did not move at all -- a neighbour competing for memory,
/// not a clock drop.  Constants measured under a neighbour describe the
/// neighbour as much as the target, so the run refuses to call itself
/// calibrated instead of shipping them.
///
/// Two checks, because they fail on different neighbours:
///  - achieved DRAM against the pin bandwidth the device reports, which catches
///    a neighbour that was there the whole time and so left no drift behind;
///  - the same point re-measured at the end against its value from the start,
///    which catches one that arrived or left partway through.
bool MachineWasIdle(TargetSpec& spec, Options const& options,
                    std::ostream& log) {
  int clock_khz = 0, bus_bits = 0;
  CheckCuda(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate,
                                   options.device),
            "cudaDevAttrMemoryClockRate");
  CheckCuda(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth,
                                   options.device),
            "cudaDevAttrGlobalMemoryBusWidth");
  double const peak_gbps =
      2.0 * static_cast<double>(clock_khz) * 1e3 * (bus_bits / 8.0) / 1e9;
  double const fraction =
      peak_gbps > 0.0 ? spec.calib.dram_gbps / peak_gbps : 0.0;

  // Re-measure the largest working set of the sweep, the one that reads at the
  // DRAM rate, and compare with what the sweep got for it.
  std::size_t const bytes = static_cast<std::size_t>(
      spec.calib.l2_curve_bytes.empty() ? 0 : spec.calib.l2_curve_bytes.back());
  double again_gbps = 0.0;
  if (bytes > 0) {
    float4* data = nullptr;
    float* sink = nullptr;
    CheckCuda(cudaMalloc(&data, bytes), "cudaMalloc(guard)");
    CheckCuda(cudaMalloc(&sink, sizeof(float) * 4), "cudaMalloc(guard sink)");
    int const blocks = ResidentBlocks(
        spec, reinterpret_cast<void const*>(StreamReadKernel), kThreads);
    std::size_t elements = bytes / sizeof(float4);
    FillKernel<<<blocks, kThreads>>>(data, elements);
    CheckCuda(cudaDeviceSynchronize(), "FillKernel(guard)");
    Stat stat = TimeMs(options.repeats, [&] {
      StreamReadKernel<<<blocks, kThreads>>>(data, elements, 2, 977, sink);
    });
    again_gbps = static_cast<double>(bytes) * 2 / (stat.median * 1e6);
    CheckCuda(cudaFree(data), "cudaFree");
    CheckCuda(cudaFree(sink), "cudaFree");
  }
  double const drift =
      spec.calib.l2_curve_gbps.empty()
          ? 0.0
          : std::fabs(again_gbps - spec.calib.l2_curve_gbps.back()) /
                spec.calib.l2_curve_gbps.back();

  Record(spec, "dram_peak_gbps", peak_gbps, "GB/s", Stat{peak_gbps, 0.0, 1}, 
         "2 * memory clock * bus width from cudaDeviceGetAttribute; the pin "
         "rate, not an achieved one");
  Record(spec, "dram_fraction_of_peak", fraction, "ratio", Stat{fraction, 0.0, 1},
         "achieved dram_gbps over the pin rate; the run is rejected below "
         "kMinFractionOfPeak");
  Record(spec, "dram_drift", drift, "ratio", Stat{drift, 0.0, options.repeats},
         "largest working set re-measured after every other group, against "
         "its value from the sweep; the run is rejected above kMaxDrift");

  // A clean run on this machine reads at 97% of the pin rate and the two
  // measurements of the same point agree to 0.02%.  A contended one read at
  // 46%.  The thresholds sit between, wide enough for an HBM part that
  // achieves less of its pin rate than a GDDR one.
  double const kMinFractionOfPeak = 0.70;
  double const kMaxDrift = 0.05;
  bool const idle = fraction >= kMinFractionOfPeak && drift <= kMaxDrift;
  log << "  dram peak           = " << peak_gbps << " GB/s ("
      << fraction * 100.0 << "% achieved)\n"
      << "  dram drift          = " << drift * 100.0 << "%\n";
  if (!idle) {
    log << "  [!] this GPU was not idle for the run: "
        << (fraction < kMinFractionOfPeak
                ? "the achieved DRAM rate is too far below the pin rate"
                : "the same working set read at two different rates")
        << ".\n      The profile is written but marked uncalibrated; re-run "
           "when the device is free.\n";
  }
  return idle;
}

void Run(TargetSpec& spec, Options const& options, std::ostream& log) {
  cudaDeviceProp properties{};
  CheckCuda(cudaGetDeviceProperties(&properties, options.device),
            "cudaGetDeviceProperties");
  CheckCuda(cudaSetDevice(options.device), "cudaSetDevice");
  auto started = std::chrono::steady_clock::now();

  log << "pipelines\n";
  WarmUpClocks(spec, log);
  MeasurePipelines(spec, options, log);
  log << "sync\n";
  MeasureSync(spec, options, log);
  if (options.skip_streamk) {
    log << "streamk + interference: skipped (--skip-streamk)\n";
  } else {
    log << "streamk\n";
    MeasureStreamK(spec, options, log);
    log << "interference\n";
    // Again, five seconds after the first: the Stream-K stage is host-bound
    // between short kernels, so the clock falls back towards its 210 MHz idle
    // and the interference ratio came out anywhere from 0.90 to 2.50 depending
    // on where in the ramp the two halves of a pair landed.  Warmed here it is
    // 1.50 +- 0.007 over six consecutive runs.
    WarmUpClocks(spec, log);
    MeasureInterference(spec, options, log);
  }

  std::chrono::duration<double> elapsed =
      std::chrono::steady_clock::now() - started;
  spec.calib.wall_seconds = elapsed.count();
  spec.calib.device = properties.name;
  std::time_t now = std::time(nullptr);
  char stamp[32] = {};
  std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  spec.calib.measured_at = stamp;
  // `calibrated` means every group this target *can* run produced a value
  // and the machine was ours while it did.  The Stream-K fit is part of that,
  // so a --skip-streamk run stays uncalibrated rather than shipping a
  // half-filled file.
  spec.calib.calibrated = !options.skip_streamk && MachineWasIdle(spec, options, log);
}

}  // namespace tilemega::calib
