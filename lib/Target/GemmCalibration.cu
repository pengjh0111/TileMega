// SPDX-License-Identifier: BSD-3-Clause
//
// P4.1(c) and P4.1(d): the Stream-K per-CTA coefficients and the concurrency
// interference ratio.
//
// Both are measured on the *real* collective the megakernel dispatches to --
// backend::GemmCandidate's SIMT f32 TN mainloop and epilogue -- because a
// proxy kernel would calibrate a pipeline that TileMega never runs.
//
// The problem shapes here are deliberately disjoint from the two reference
// models (N in {384, 768}, K in {256..1024} against the models' N in
// {256,512,1024}, K in {512,1024}).  The 2154 measured points are the cost
// model's validation set; fitting coefficients on them and then reporting a
// rank correlation against them would be circular.
#include <tilemega/Target/Calibration.h>

#include <tilemega/Backend/CutlassGemmCandidate.h>

#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
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

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values.empty() ? 0.0 : values[values.size() / 2];
}

double RelStddev(std::vector<double> const& values) {
  if (values.empty()) return 0.0;
  double mean = 0.0;
  for (double v : values) mean += v;
  mean /= static_cast<double>(values.size());
  double variance = 0.0;
  for (double v : values) variance += (v - mean) * (v - mean);
  variance /= static_cast<double>(values.size());
  return mean != 0.0 ? std::sqrt(variance) / std::fabs(mean) : 0.0;
}

struct Timed {
  double median_ms = 0.0;
  double rel_stddev = 0.0;
  int samples = 0;
};

template <class Launch>
Timed TimeMs(int repeats, Launch&& launch) {
  cudaEvent_t start, stop;
  CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
  CheckCuda(cudaEventCreate(&stop), "cudaEventCreate");
  for (int i = 0; i < 3; ++i) launch();
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
  Timed timed;
  timed.median_ms = Median(samples);
  timed.rel_stddev = RelStddev(samples);
  timed.samples = repeats;
  return timed;
}

void Record(TargetSpec& spec, std::string name, double value, char const* unit,
            int samples, double rel_stddev, std::string method) {
  TargetSpec::Measurement record;
  record.name = std::move(name);
  record.value = value;
  record.unit = unit;
  record.samples = samples;
  record.rel_stddev = rel_stddev;
  record.method = std::move(method);
  spec.calib.measurements.push_back(std::move(record));
}

/// One split-K chunk, exactly as ModelHarness::Create builds it: A and B are
/// the same matrices seen from a K offset, and only chunk 0 applies beta*C.
template <class Candidate>
struct Invocation {
  typename Candidate::Mainloop::Params mainloop;
  typename Candidate::Epilogue::Params epilogue;
  int m = 0, n = 0, k = 0;
};

template <class Candidate>
__global__ __launch_bounds__(Candidate::kThreads, 1) void CalibGemmKernel(
    Invocation<Candidate> const* table, int tiles_m, int tiles_n) {
  using namespace cute;
  extern __shared__ char shared[];
  int const tiles = tiles_m * tiles_n;
  int const chunk = static_cast<int>(blockIdx.x) / tiles;
  int const local = static_cast<int>(blockIdx.x) - chunk * tiles;
  int const tile_n = local % tiles_n;
  int const tile_m = local / tiles_n;
  Invocation<Candidate> const& invocation = table[chunk];

  constexpr auto tile_shape = typename Candidate::Mainloop::TileShape{};
  int const M = invocation.m, N = invocation.n, K = invocation.k;
  Tensor matrix_a = make_tensor(make_gmem_ptr(invocation.mainloop.ptr_A),
                                make_shape(M, K, 1), invocation.mainloop.dA);
  Tensor matrix_b = make_tensor(make_gmem_ptr(invocation.mainloop.ptr_B),
                                make_shape(N, K, 1), invocation.mainloop.dB);
  auto block_coord = make_coord(tile_m, tile_n, _, 0);
  Tensor gA = local_tile(matrix_a(_, _, 0), tile_shape, take<0, 3>(block_coord),
                         Step<_1, X, _1>{});
  Tensor gB = local_tile(matrix_b(_, _, 0), tile_shape, take<0, 3>(block_coord),
                         Step<X, _1, _1>{});
  auto residue = make_tuple(M - size<0>(gA) * tile_m, N - size<0>(gB) * tile_n,
                            K - size<1>(gA) * size<2>(gA));
  typename Candidate::Mainloop::TiledMma tiled_mma;
  Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(tile_shape));
  clear(accum);
  auto k_iter = make_coord_iterator(shape<2>(gA));
  typename Candidate::Mainloop mainloop;
  mainloop(accum, gA, gB, accum, k_iter, size<2>(gA), residue,
           static_cast<int>(threadIdx.x), shared);
  typename Candidate::Epilogue epilogue(invocation.epilogue);
  epilogue(cute::Shape<int, int, int, int>{M, N, K, 1}, tile_shape,
           make_coord(tile_m, tile_n, 0, 0), accum, tiled_mma, residue,
           static_cast<int>(threadIdx.x), shared);
}

/// The reduction half of §2.4's Split, byte for byte the GemmCombineTaskBody
/// loop the harness runs.
__global__ __launch_bounds__(256) void CalibCombineKernel(float const* partials,
                                                          float* out, int count,
                                                          int chunks) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    float sum = 0.0f;
    for (int c = 0; c < chunks; ++c) sum += partials[c * count + i];
    out[i] = sum;
  }
}

/// Empty kernel at the same launch configuration.  Its elapsed time is the
/// launch overhead, which is subtracted from every fit below: inside the
/// persistent megakernel a stage is a task body, not a launch, so leaving
/// ~4 us of launch in `a` would be a constant the cost model must never see.
__global__ void NullKernel() {}

/// A DRAM-bound neighbour for the interference measurement, run at one CTA
/// per SM so a GEMM CTA can always co-reside: a hog that fills the machine
/// measures serialization, not interference.
///
/// The pass count is bounded rather than a host-cleared spin flag.  A
/// resident spin kernel deadlocks against every full-device wait in the
/// process -- cudaDeviceSynchronize, a pageable cudaMemcpy, anything on a
/// blocking stream -- and the flag write is itself such a wait, so it queues
/// behind the very kernel it exists to stop.  Bounding the work removes the
/// class; the caller then checks that the neighbour really did outlast the
/// window it was meant to disturb.
__global__ __launch_bounds__(256) void MemoryHogKernel(float* data,
                                                       std::size_t elements,
                                                       int passes) {
  std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
  for (int pass = 0; pass < passes; ++pass) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
         i < elements; i += stride) {
      data[i] = data[i] * 1.000001f + 1.0f;
    }
  }
}

/// Per-element hashes in [1, 2).  A zero-filled buffer is compressible, and
/// the reduction then reads it at 33 TB/s -- six times the L2's own rate.
__global__ void FillKernel(float* v, std::size_t elements) {
  for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
       i < elements; i += (std::size_t)gridDim.x * blockDim.x) {
    unsigned h = static_cast<unsigned>(i) * 2654435761u;
    h = h * 1664525u + 1013904223u;
    v[i] = __uint_as_float((h & 0x007fffffu) | 0x3f800000u);
  }
}

/// The neighbour stopped before the window it was meant to disturb ended, so
/// part of that window ran clean and the ratio would understate.
struct ShortNeighbour : std::runtime_error {
  ShortNeighbour(double neighbour_ms, double window_ms)
      : std::runtime_error("interference: the neighbour ran " +
                           std::to_string(neighbour_ms) +
                           " ms but the GEMM window was " +
                           std::to_string(window_ms) + " ms"),
        factor(2.0 * window_ms / neighbour_ms) {}
  double factor;
};

/// Least squares fit of y = intercept + slope * x, with the coefficient of
/// determination.
struct LineFit {
  double intercept = 0.0, slope = 0.0, r2 = 0.0;
};

LineFit FitLine(std::vector<double> const& x, std::vector<double> const& y) {
  LineFit fit;
  std::size_t n = x.size();
  if (n < 2) return fit;
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < n; ++i) {
    sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i];
  }
  double denominator = n * sxx - sx * sx;
  if (denominator == 0.0) return fit;
  fit.slope = (n * sxy - sx * sy) / denominator;
  fit.intercept = (sy - fit.slope * sx) / n;
  double mean = sy / n, ss_total = 0, ss_residual = 0;
  for (std::size_t i = 0; i < n; ++i) {
    double predicted = fit.intercept + fit.slope * x[i];
    ss_total += (y[i] - mean) * (y[i] - mean);
    ss_residual += (y[i] - predicted) * (y[i] - predicted);
  }
  fit.r2 = ss_total > 0.0 ? 1.0 - ss_residual / ss_total : 1.0;
  return fit;
}

/// The reduction stage's cost, fitted once for the whole target.
///
/// CalibCombineKernel does not depend on the GEMM tile shape, so fitting b
/// and d per shape only re-measured the same two numbers at that shape's
/// single output width.  At M=4, N=64 the whole reduction moves 32 KB, which
/// is under the timing noise floor, and the first attempt duly produced
/// negative d with r^2 as low as 0.05.  Sweeping the width over four decades
/// lifts the traffic term above the floor.
///
///   combine_ns = fixed_ns + base_ns_per_elem * count
///                         + d_ns_per_peer_elem * (chunks - 1) * count
///
/// The Stream-K b of a given shape is then fixed_ns + base_ns_per_elem *
/// count(shape): the part of the reduction that does not grow with the peer
/// count, which is what b means.
struct CombineFit {
  double fixed_ns = 0.0;
  double base_ns_per_elem = 0.0;
  double d_ns_per_peer_elem = 0.0;
  double dram_d_ns_per_peer_elem = 0.0;  ///< the same slope past the L2 knee
  double peer_r2 = 0.0;   ///< of d, across widths
  double width_r2 = 0.0;  ///< of fixed_ns and base_ns_per_elem
  double worst_rsd = 0.0;
  bool valid = false;
};

/// Device state shared by every shape: A, B, C, D and the split partials.
struct Buffers {
  float* a = nullptr;
  float* b = nullptr;
  float* c = nullptr;
  float* d = nullptr;
  float* partials = nullptr;
  void* table = nullptr;
};

constexpr int kCalibM = 4;      ///< the reference models' token count
constexpr int kCalibTilesN = 6; ///< one wave: 6 CTAs unsplit, 48 at chunks=8
constexpr int kMaxK = 2048;
constexpr int kMaxN = 1536;
constexpr int kMaxChunks = 32;

/// Build the per-chunk invocations for one (shape, n, k, chunks) point and
/// upload them, mirroring ModelHarness::Create's K-offset construction.
template <class Candidate>
void BuildTable(Buffers const& buffers, int n, int k, int chunks,
                std::vector<Invocation<Candidate>>& host) {
  using Mainloop = typename Candidate::Mainloop;
  using Epilogue = typename Candidate::Epilogue;
  int const tile_k = Candidate::Traits().tile_k;
  int const k_tiles = (k + tile_k - 1) / tile_k;
  host.clear();
  for (int chunk = 0; chunk < chunks; ++chunk) {
    int k_begin = chunk * k_tiles / chunks * tile_k;
    int k_end = (chunk + 1) * k_tiles / chunks * tile_k;
    if (k_end > k) k_end = k;
    cute::Shape<int, int, int, int> problem{kCalibM, n, k_end - k_begin, 1};
    auto stride_a = cutlass::make_cute_packed_stride(
        typename Mainloop::StrideA{}, cute::make_shape(kCalibM, k, 1));
    auto stride_b = cutlass::make_cute_packed_stride(
        typename Mainloop::StrideB{}, cute::make_shape(n, k, 1));
    auto stride_c = cutlass::make_cute_packed_stride(
        typename Epilogue::StrideC{}, cute::make_shape(kCalibM, n, 1));
    auto stride_d = cutlass::make_cute_packed_stride(
        typename Epilogue::StrideD{}, cute::make_shape(kCalibM, n, 1));
    typename Mainloop::Arguments main_args{buffers.a + k_begin, stride_a,
                                           buffers.b + k_begin, stride_b};
    float* destination =
        chunks > 1
            ? buffers.partials + static_cast<std::size_t>(chunk) * kCalibM * n
            : buffers.d;
    typename Epilogue::Arguments epilogue_args{
        {1.0f, chunk == 0 ? 1.0f : 0.0f}, buffers.c, stride_c, destination,
        stride_d};
    Invocation<Candidate> invocation;
    invocation.mainloop =
        Mainloop::to_underlying_arguments(problem, main_args, nullptr);
    invocation.epilogue =
        Epilogue::to_underlying_arguments(problem, epilogue_args, nullptr);
    invocation.m = kCalibM;
    invocation.n = n;
    invocation.k = k_end - k_begin;
    host.push_back(invocation);
  }
  CheckCuda(cudaMemcpy(buffers.table, host.data(),
                       host.size() * sizeof(Invocation<Candidate>),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(invocation table)");
}

/// Fit one tile shape's Stream-K coefficients.
///
///   a, c  come from the unsplit sweep over k: per-CTA time is linear in the
///         MAC-loop iteration count, `a` is its intercept (launch, prologue
///         fill, output tile writeback) and `c` its slope.
///   b, d  come from the split sweep at fixed total work.  In TileMega the
///         fixup is not an atomic reduction inside the GEMM but the separate
///         GemmCombine stage, so `b` is the fixed price of producing a partial
/// Fit the reduction stage over a (width, peers) grid.
///
/// The sweep stays inside one level of the hierarchy.  Per-peer cost is not
/// one number across all widths -- measured per width it is 0.00057 ns/elem
/// at a 8 MB working set and 0.0045 ns/elem at 128 MB, which are the L2 and
/// the DRAM rate for a 4-byte read, a factor of eight apart.  The reference
/// models reduce 4 x N outputs with N <= 4096 over at most 16 chunks, so at
/// most 1 MB: the L2 regime.  d is fitted there, and the DRAM-regime value is
/// recorded beside it so the two are never silently averaged.
CombineFit MeasureCombine(TargetSpec& spec, Options const& options,
                          std::ostream& log) {
  CombineFit fit;
  constexpr int kWidest = 1048576;   ///< 128 MB of partials: past the L2 knee
  constexpr int kL2Widest = 262144;  ///< 32 MB: still inside the 72 MB L2
  constexpr int kBaseWidest = 4194304;  ///< 32 MB at two elements per output
  float *partials = nullptr, *out = nullptr;
  CheckCuda(cudaMalloc(&partials, static_cast<std::size_t>(kWidest) *
                                      kMaxChunks * sizeof(float)),
            "cudaMalloc(combine partials)");
  CheckCuda(cudaMalloc(&out, kBaseWidest * sizeof(float)),
            "cudaMalloc(combine D)");
  std::size_t const partial_elements =
      static_cast<std::size_t>(kWidest) * kMaxChunks;
  FillKernel<<<1024, 256>>>(partials, partial_elements);
  CheckCuda(cudaDeviceSynchronize(), "FillKernel");

  // The peer-independent cost is measured at chunks = 1 rather than read off
  // the fit's intercept.  Extrapolated, it is the difference of two numbers a
  // few hundred ns apart at widths where the whole kernel is latency bound,
  // and it came back negative with r^2 = 0.03.
  // One fixed, persistent-sized grid for every point.  Sizing the grid to
  // the width made the subtracted launch scale with the width too -- and
  // block dispatch overlaps the memory traffic rather than adding to it, so
  // the subtraction removed most of the signal and the reduction appeared to
  // read at 38 TB/s.  A fixed grid also matches how the megakernel runs the
  // stage: a resident grid draining a task queue, not a launch per width.
  int const blocks = spec.res.num_sms * 2;
  double const launch_ns =
      TimeMs(options.repeats, [&] { NullKernel<<<blocks, 256>>>(); })
          .median_ms * 1e6;

  auto sweep_single = [&](int count) {
    Timed timed = TimeMs(options.repeats, [&] {
      CalibCombineKernel<<<blocks, 256>>>(partials, out, count, 1);
    });
    fit.worst_rsd = std::max(fit.worst_rsd, timed.rel_stddev);
    double ns = timed.median_ms * 1e6 - launch_ns;
    log << "  combine single count=" << count << ": " << ns << "ns ("
        << ns / count << "ns/elem)\n";
    return ns;
  };

  auto sweep_peers = [&](int count) {
    std::vector<double> peers, combine_ns;
    for (int chunks : {2, 4, 8, 16, 32}) {
      Timed timed = TimeMs(options.repeats, [&] {
        CalibCombineKernel<<<blocks, 256>>>(partials, out, count, chunks);
      });
      fit.worst_rsd = std::max(fit.worst_rsd, timed.rel_stddev);
      peers.push_back(static_cast<double>(chunks - 1));
      combine_ns.push_back(timed.median_ms * 1e6 - launch_ns);
    }
    LineFit per_width = FitLine(peers, combine_ns);
    log << "  combine count=" << count << ": per-peer=" << per_width.slope
        << "ns (" << per_width.slope / count << "ns/elem) r2=" << per_width.r2
        << '\n';
    return per_width;
  };

  std::vector<double> widths, peer_slopes;
  for (int count : {1024, 4096, 16384, 65536, kL2Widest}) {
    LineFit per_width = sweep_peers(count);
    widths.push_back(count);
    peer_slopes.push_back(per_width.slope);
  }
  // A separate, wider sweep: at chunks=1 the working set is two elements per
  // output, so widths that are past the L2 knee for d are still inside it
  // here -- and below 2^16 the kernel sits on its launch latency floor, where
  // the slope is unmeasurable.
  std::vector<double> base_widths, singles;
  for (int count : {65536, 262144, 1048576, kBaseWidest}) {
    base_widths.push_back(count);
    singles.push_back(sweep_single(count));
  }
  LineFit dram = sweep_peers(kWidest);
  CheckCuda(cudaFree(partials), "cudaFree");
  CheckCuda(cudaFree(out), "cudaFree");
  if (widths.size() < 2) return fit;

  // Through the origin: the per-peer slope is proportional to the width, and
  // a free intercept would only absorb the smallest width's noise.
  double sxy = 0.0, sxx = 0.0, mean = 0.0;
  for (std::size_t i = 0; i < widths.size(); ++i) {
    sxy += widths[i] * peer_slopes[i];
    sxx += widths[i] * widths[i];
    mean += peer_slopes[i];
  }
  fit.d_ns_per_peer_elem = sxx > 0.0 ? sxy / sxx : 0.0;
  mean /= static_cast<double>(widths.size());
  double ss_total = 0.0, ss_residual = 0.0;
  for (std::size_t i = 0; i < widths.size(); ++i) {
    double predicted = fit.d_ns_per_peer_elem * widths[i];
    ss_total += (peer_slopes[i] - mean) * (peer_slopes[i] - mean);
    ss_residual += (peer_slopes[i] - predicted) * (peer_slopes[i] - predicted);
  }
  fit.peer_r2 = ss_total > 0.0 ? 1.0 - ss_residual / ss_total : 1.0;

  LineFit base = FitLine(base_widths, singles);
  fit.fixed_ns = base.intercept;
  fit.base_ns_per_elem = base.slope;
  fit.width_r2 = base.r2;
  fit.dram_d_ns_per_peer_elem = dram.slope / kWidest;
  fit.valid = true;
  // Clamped, not because the fit is inconvenient but because it is
  // unresolved: the intercept is ~100 ns with a 150% spread over five runs and
  // comes out either sign, which is what an intercept below the launch
  // latency looks like.  The fitted value stays in `measurements` as evidence;
  // what the cost model gets is the honest floor rather than a negative fixed
  // cost it would subtract from every reduction.
  spec.calib.combine_fixed_ns = std::max(0.0, fit.fixed_ns);
  spec.calib.combine_d_dram_ns = fit.dram_d_ns_per_peer_elem;

  int const samples = options.repeats * 25;
  Record(spec, "streamk_combine_d_ns", fit.d_ns_per_peer_elem,
         "ns per peer per element", samples, fit.worst_rsd,
         "GemmCombine duration against (peers-1)*elements over widths "
         "{1024,4096,16384,65536,262144} x chunks {2,4,8,16,32}, launch "
         "subtracted; every point L2-resident, as the reference models are");
  Record(spec, "streamk_combine_d_dram_ns", fit.dram_d_ns_per_peer_elem,
         "ns per peer per element", options.repeats * 5, fit.worst_rsd,
         "the same slope at a 128 MB working set, past the L2 knee: the "
         "regime the reference models do not reach, recorded so the two are "
         "not averaged");
  Record(spec, "streamk_combine_base_ns", fit.base_ns_per_elem,
         "ns per element", options.repeats * 5, fit.worst_rsd,
         "slope of the chunks=1 reduction against width: the output store and "
         "the single partial read, measured rather than extrapolated");
  Record(spec, "streamk_combine_fixed_ns", fit.fixed_ns, "ns",
         options.repeats * 5, fit.worst_rsd,
         "intercept of the same chunks=1 fit; unresolved -- |value| < 300 ns "
         "with a 150% spread over five runs and either sign, so the profile "
         "carries max(0, fit)");
  Record(spec, "streamk_combine_peer_r2", fit.peer_r2, "", samples, 0.0,
         "fit quality of d across widths");
  Record(spec, "streamk_combine_width_r2", fit.width_r2, "", samples, 0.0,
         "fit quality of the peer-independent part across widths");
  log << "  combine: d=" << fit.d_ns_per_peer_elem << "ns/peer/elem (r2="
      << fit.peer_r2 << ") base=" << fit.base_ns_per_elem << "ns/elem fixed="
      << fit.fixed_ns << "ns (r2=" << fit.width_r2 << ") dram-d="
      << fit.dram_d_ns_per_peer_elem << "ns/peer/elem\n";
  return fit;
}

///         at all (partial store plus the combine's launch) and `d` is the
///         marginal price of one more peer (one more read per output element).
template <int M, int N, int K, int S>
bool FitShape(TargetSpec& spec, Options const& options, Buffers const& buffers,
              CombineFit const& combine, std::ostream& log) {
  using Candidate = backend::GemmCandidate<M, N, K, S>;
  if constexpr (!Candidate::kShapeLegal) {
    log << "  tile " << M << "x" << N << "x" << K << "s" << S
        << ": illegal shape, skipped\n";
    return false;
  } else {
    if (Candidate::kSmemBytes > spec.res.max_dynamic_smem_per_cta) {
      log << "  tile " << M << "x" << N << "x" << K << "s" << S
          << ": smem " << Candidate::kSmemBytes << " over budget, skipped\n";
      return false;
    }
    CheckCuda(cudaFuncSetAttribute(
                  reinterpret_cast<void const*>(CalibGemmKernel<Candidate>),
                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                  Candidate::kSmemBytes),
              "cudaFuncSetAttribute");

    int const n = N * kCalibTilesN;
    int const tiles_m = (kCalibM + M - 1) / M;
    int const tiles_n = kCalibTilesN;
    std::vector<Invocation<Candidate>> host;
    auto* table = static_cast<Invocation<Candidate>*>(buffers.table);

    std::vector<double> iteration_counts, per_cta_ns;
    double worst_rsd = 0.0;
    // Down to a single MAC iteration: `a` is the prologue and epilogue, and
    // anchored only at 4 iterations it is a long extrapolation from y values
    // of 7-111 us.  It swung between -355 ns and +2882 ns across runs.
    for (int iters : {1, 2, 4, 8, 16, 32, 64}) {
      int k = iters * K;
      if (k > kMaxK) continue;
      BuildTable<Candidate>(buffers, n, k, 1, host);
      Timed timed = TimeMs(options.repeats, [&] {
        CalibGemmKernel<Candidate>
            <<<tiles_m * tiles_n, Candidate::kThreads, Candidate::kSmemBytes>>>(
                table, tiles_m, tiles_n);
      });
      // Paired with the point it is subtracted from.  Measured once before the
      // sweep instead, the launch estimate's own jitter lands undiluted on the
      // intercept -- it shifts every point by the same amount, which is exactly
      // what an intercept is -- and `a` swung +-350 ns run to run, with three
      // shapes going negative.  Pairing also cancels the clock drift across the
      // seconds the sweep takes.
      double launch_ns =
          TimeMs(options.repeats, [&] {
            NullKernel<<<tiles_m * tiles_n, Candidate::kThreads>>>();
          }).median_ms * 1e6;
      // One wave: every CTA is resident at once, so the kernel's duration is
      // one CTA's duration.
      iteration_counts.push_back(iters);
      per_cta_ns.push_back(timed.median_ms * 1e6 - launch_ns);
      worst_rsd = std::max(worst_rsd, timed.rel_stddev);
    }
    if (iteration_counts.size() < 2) return false;
    LineFit linear = FitLine(iteration_counts, per_cta_ns);

    TargetSpec::StreamKPoint point;
    point.tile_m = M; point.tile_n = N; point.tile_k = K; point.stages = S;
    point.a_ns = linear.intercept;
    point.c_ns = linear.slope;
    point.b_ns = combine.base_ns_per_elem;
    point.d_ns = combine.d_ns_per_peer_elem;
    point.fit_r2 = std::min(linear.r2, std::min(combine.peer_r2,
                                                combine.width_r2));
    point.ac_r2 = linear.r2;
    spec.calib.streamk.push_back(point);

    std::string tag = std::to_string(M) + "x" + std::to_string(N) + "x" +
                      std::to_string(K) + "s" + std::to_string(S);
    Record(spec, "streamk_a_ns[" + tag + "]", point.a_ns, "ns",
           options.repeats * static_cast<int>(iteration_counts.size()),
           worst_rsd,
           "intercept of per-CTA time vs MAC iterations over {1,2,4,8,16,32,"
           "64}, unsplit single-wave GEMM at M=4, N=" + std::to_string(n) +
           ", launch time subtracted; r2=" + std::to_string(linear.r2));
    Record(spec, "streamk_c_ns[" + tag + "]", point.c_ns, "ns per iteration",
           options.repeats * static_cast<int>(iteration_counts.size()),
           worst_rsd, "slope of the same fit");
    Record(spec, "streamk_b_ns[" + tag + "]", point.b_ns, "ns per element",
           options.repeats * 4, combine.worst_rsd,
           "shape-independent: see streamk_combine_base_ns");
    Record(spec, "streamk_d_ns[" + tag + "]", point.d_ns,
           "ns per peer per element", options.repeats * 25, combine.worst_rsd,
           "shape-independent: see streamk_combine_d_ns");

    log << "  tile " << tag << ": a=" << point.a_ns << "ns c=" << point.c_ns
        << "ns/iter b=" << point.b_ns << "ns/elem d=" << point.d_ns
        << "ns/peer/elem r2=" << point.fit_r2 << '\n';
    return true;
  }
}

}  // namespace

std::vector<TargetSpec::StreamKPoint> StreamKShapes() {
  return {{128, 128, 16, 3, 0, 0, 0, 0, 0},
          {64, 64, 16, 3, 0, 0, 0, 0, 0},
          {32, 32, 32, 3, 0, 0, 0, 0, 0},
          {16, 64, 32, 3, 0, 0, 0, 0, 0},
          {16, 64, 16, 2, 0, 0, 0, 0, 0},
          {256, 128, 16, 3, 0, 0, 0, 0, 0}};
}

void MeasureStreamK(TargetSpec& spec, Options const& options,
                    std::ostream& log) {
  Buffers buffers;
  std::size_t const ab_elements =
      static_cast<std::size_t>(kMaxN) * kCalibTilesN * kMaxK;
  std::size_t const cd_elements =
      static_cast<std::size_t>(kCalibM) * kMaxN * kCalibTilesN;
  CheckCuda(cudaMalloc(&buffers.a, static_cast<std::size_t>(kCalibM) * kMaxK *
                                       sizeof(float)),
            "cudaMalloc(A)");
  CheckCuda(cudaMalloc(&buffers.b, ab_elements * sizeof(float)), "cudaMalloc(B)");
  CheckCuda(cudaMalloc(&buffers.c, cd_elements * sizeof(float)), "cudaMalloc(C)");
  CheckCuda(cudaMalloc(&buffers.d, cd_elements * sizeof(float)), "cudaMalloc(D)");
  CheckCuda(cudaMalloc(&buffers.partials,
                       cd_elements * kMaxChunks * sizeof(float)),
            "cudaMalloc(partials)");
  CheckCuda(cudaMalloc(&buffers.table, 4096), "cudaMalloc(table)");
  CheckCuda(cudaMemset(buffers.a, 0,
                       static_cast<std::size_t>(kCalibM) * kMaxK * sizeof(float)),
            "cudaMemset(A)");
  CheckCuda(cudaMemset(buffers.b, 0, ab_elements * sizeof(float)),
            "cudaMemset(B)");
  CheckCuda(cudaMemset(buffers.c, 0, cd_elements * sizeof(float)),
            "cudaMemset(C)");

  CombineFit combine = MeasureCombine(spec, options, log);

  FitShape<128, 128, 16, 3>(spec, options, buffers, combine, log);
  FitShape<64, 64, 16, 3>(spec, options, buffers, combine, log);
  FitShape<32, 32, 32, 3>(spec, options, buffers, combine, log);
  FitShape<16, 64, 32, 3>(spec, options, buffers, combine, log);
  FitShape<16, 64, 16, 2>(spec, options, buffers, combine, log);
  FitShape<256, 128, 16, 3>(spec, options, buffers, combine, log);

  CheckCuda(cudaFree(buffers.a), "cudaFree");
  CheckCuda(cudaFree(buffers.b), "cudaFree");
  CheckCuda(cudaFree(buffers.c), "cudaFree");
  CheckCuda(cudaFree(buffers.d), "cudaFree");
  CheckCuda(cudaFree(buffers.partials), "cudaFree");
  CheckCuda(cudaFree(buffers.table), "cudaFree");
}

void MeasureInterference(TargetSpec& spec, Options const& options,
                         std::ostream& log) {
  // The operating point the oracle actually found: a 16x64 tile split 16 ways
  // so the GEMM fills the grid, which is exactly when it has to share the
  // machine with everything else in the megakernel.
  using Candidate = backend::GemmCandidate<16, 64, 32, 3>;
  static_assert(Candidate::kShapeLegal);

  Buffers buffers;
  // Big enough that the GEMM's own duration dominates.  At N=512, K=512 the
  // 16 chunks each ran a single MAC iteration and the 5 us per launch was
  // mostly launch, so the ratio was measuring interference on the launch path.
  int const n = kMaxN;
  int const k = kMaxK;
  int const chunks = 16;
  int const tiles_m = 1;
  int const tiles_n = (n + 64 - 1) / 64;
  std::size_t const cd_elements = static_cast<std::size_t>(kCalibM) * n;
  CheckCuda(cudaMalloc(&buffers.a, static_cast<std::size_t>(kCalibM) * k *
                                       sizeof(float)),
            "cudaMalloc(A)");
  CheckCuda(cudaMalloc(&buffers.b,
                       static_cast<std::size_t>(n) * k * sizeof(float)),
            "cudaMalloc(B)");
  CheckCuda(cudaMalloc(&buffers.c, cd_elements * sizeof(float)), "cudaMalloc(C)");
  CheckCuda(cudaMalloc(&buffers.d, cd_elements * sizeof(float)), "cudaMalloc(D)");
  CheckCuda(cudaMalloc(&buffers.partials,
                       cd_elements * chunks * sizeof(float)),
            "cudaMalloc(partials)");
  CheckCuda(cudaMalloc(&buffers.table,
                       chunks * sizeof(Invocation<Candidate>)),
            "cudaMalloc(table)");
  CheckCuda(cudaMemset(buffers.a, 0,
                       static_cast<std::size_t>(kCalibM) * k * sizeof(float)),
            "cudaMemset");
  CheckCuda(cudaMemset(buffers.b, 0,
                       static_cast<std::size_t>(n) * k * sizeof(float)),
            "cudaMemset");
  CheckCuda(cudaMemset(buffers.c, 0, cd_elements * sizeof(float)), "cudaMemset");

  std::vector<Invocation<Candidate>> host;
  BuildTable<Candidate>(buffers, n, k, chunks, host);
  auto* table = static_cast<Invocation<Candidate>*>(buffers.table);
  CheckCuda(cudaFuncSetAttribute(
                reinterpret_cast<void const*>(CalibGemmKernel<Candidate>),
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                Candidate::kSmemBytes),
            "cudaFuncSetAttribute");

  std::size_t const hog_elements = 128u * 1024u * 1024u / sizeof(float) * 2u;
  float* hog = nullptr;
  CheckCuda(cudaMalloc(&hog, hog_elements * sizeof(float)), "cudaMalloc(hog)");
  CheckCuda(cudaMemset(hog, 0, hog_elements * sizeof(float)), "cudaMemset(hog)");

  cudaStream_t gemm_stream, hog_stream;
  CheckCuda(cudaStreamCreateWithFlags(&gemm_stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
  CheckCuda(cudaStreamCreateWithFlags(&hog_stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");

  auto launch_gemm = [&](cudaStream_t stream) {
    CalibGemmKernel<Candidate><<<tiles_m * tiles_n * chunks, Candidate::kThreads,
                                 Candidate::kSmemBytes, stream>>>(table, tiles_m,
                                                                  tiles_n);
  };

  cudaEvent_t start, stop_event, hog_start, hog_stop;
  CheckCuda(cudaEventCreate(&start), "cudaEventCreate");
  CheckCuda(cudaEventCreate(&stop_event), "cudaEventCreate");
  CheckCuda(cudaEventCreate(&hog_start), "cudaEventCreate");
  CheckCuda(cudaEventCreate(&hog_stop), "cudaEventCreate");

  // The neighbour's own bandwidth, so "memory bound" is measured, not
  // asserted.  It also sets the pass count: the hog must outlast the whole
  // GEMM timing window below, and this is how long one pass takes.
  int const passes = 4;
  double hog_ms = 0.0;
  {
    Timed timed = TimeMs(options.repeats, [&] {
      MemoryHogKernel<<<spec.res.num_sms, 256>>>(hog, hog_elements, passes);
    });
    hog_ms = timed.median_ms;
    double moved = 2.0 * static_cast<double>(hog_elements) * sizeof(float) *
                   passes;  // read + write
    double gbps = moved / (timed.median_ms * 1e6);
    Record(spec, "interference_neighbour_gbps", gbps, "GB/s", timed.samples,
           timed.rel_stddev,
           "read-modify-write over 256 MiB at one CTA per SM, four passes; "
           "the neighbour interference_ratio is measured against");
    log << "  neighbour bandwidth = " << gbps << " GB/s over " << hog_ms
        << " ms\n";
  }

  int const iterations = options.repeats + 2;
  double window_ms = 0.0;
  auto time_gemm = [&](bool with_neighbour, int hog_passes) {
    if (with_neighbour) {
      CheckCuda(cudaEventRecord(hog_start, hog_stream), "cudaEventRecord");
      MemoryHogKernel<<<spec.res.num_sms, 256, 0, hog_stream>>>(
          hog, hog_elements, hog_passes);
      CheckCuda(cudaEventRecord(hog_stop, hog_stream), "cudaEventRecord");
    }
    std::vector<double> samples;
    double total = 0.0;
    for (int i = 0; i < iterations; ++i) {
      CheckCuda(cudaEventRecord(start, gemm_stream), "cudaEventRecord");
      launch_gemm(gemm_stream);
      CheckCuda(cudaEventRecord(stop_event, gemm_stream), "cudaEventRecord");
      CheckCuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize");
      float ms = 0.0f;
      CheckCuda(cudaEventElapsedTime(&ms, start, stop_event),
                "cudaEventElapsedTime");
      total += ms;
      if (i >= 2) samples.push_back(static_cast<double>(ms));
    }
    window_ms = total;
    if (with_neighbour) {
      CheckCuda(cudaStreamSynchronize(hog_stream), "cudaStreamSynchronize");
      float elapsed = 0.0f;
      CheckCuda(cudaEventElapsedTime(&elapsed, hog_start, hog_stop),
                "cudaEventElapsedTime");
      if (elapsed < total) throw ShortNeighbour{elapsed, total};
    }
    return samples;
  };

  // Paired: each ratio is an undisturbed and a disturbed window measured back
  // to back.  The GEMM's own duration drifts ~12% between runs seconds apart,
  // which is larger than the effect being measured.
  time_gemm(false, 0);
  // Sized from the undisturbed window, then raised until it actually covers
  // the disturbed one: the disturbance is exactly what makes the window grow,
  // so the first estimate is always short.
  int hog_passes = static_cast<int>(
      std::ceil(4.0 * window_ms / (hog_ms / passes))) + passes;
  std::vector<double> ratios, shared, alone;
  for (int i = 0; i < options.repeats; ++i) {
    double a = Median(time_gemm(false, 0));
    std::vector<double> disturbed;
    for (int attempt = 0;; ++attempt) {
      try {
        disturbed = time_gemm(true, hog_passes);
        break;
      } catch (ShortNeighbour const& shortfall) {
        if (attempt == 4) throw std::runtime_error(shortfall.what());
        hog_passes = static_cast<int>(std::ceil(hog_passes * shortfall.factor));
      }
    }
    double b = Median(disturbed);
    alone.push_back(a);
    shared.push_back(b);
    ratios.push_back(b / a);
  }
  double alone_ms = Median(alone);
  double shared_ms = Median(shared);
  spec.calib.interference_ratio = Median(ratios);
  Record(spec, "interference_ratio", spec.calib.interference_ratio, "ratio",
         options.repeats,
         RelStddev(ratios),
         "median of paired ratios; each pair is the duration of a 16x64x32s3 "
         "GEMM (M=4, N=1536, K=2048, split 16) on its own stream while a "
         "256 MiB streaming read-modify-write runs at one CTA per SM on "
         "another stream, over the same GEMM measured immediately before");
  log << "  gemm alone          = " << alone_ms * 1e3 << " us\n"
      << "  gemm with neighbour = " << shared_ms * 1e3 << " us\n"
      << "  interference_ratio  = " << spec.calib.interference_ratio << '\n';

  CheckCuda(cudaEventDestroy(start), "cudaEventDestroy");
  CheckCuda(cudaEventDestroy(stop_event), "cudaEventDestroy");
  CheckCuda(cudaEventDestroy(hog_start), "cudaEventDestroy");
  CheckCuda(cudaEventDestroy(hog_stop), "cudaEventDestroy");
  CheckCuda(cudaStreamDestroy(gemm_stream), "cudaStreamDestroy");
  CheckCuda(cudaStreamDestroy(hog_stream), "cudaStreamDestroy");
  CheckCuda(cudaFree(hog), "cudaFree");
  CheckCuda(cudaFree(buffers.a), "cudaFree");
  CheckCuda(cudaFree(buffers.b), "cudaFree");
  CheckCuda(cudaFree(buffers.c), "cudaFree");
  CheckCuda(cudaFree(buffers.d), "cudaFree");
  CheckCuda(cudaFree(buffers.partials), "cudaFree");
  CheckCuda(cudaFree(buffers.table), "cudaFree");
}

}  // namespace tilemega::calib
