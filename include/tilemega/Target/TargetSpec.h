// SPDX-License-Identifier: BSD-3-Clause
//
// TileMega -- Target/TargetSpec.h
//
// Single source of truth for *runtime-visible* hardware properties.
//
// Design rule (see task spec §2, skeleton §8.7):
//   Business code NEVER compares architecture version numbers and NEVER
//   hard-codes a hardware constant.  It asks TargetSpec.
//
// Architecture versions and resource literals are never policy inputs.
// Business code queries `target.caps` and `target.res` instead.
//
// The device-side compile-time counterpart and its compiler-macro dispatch
// live exclusively in Target/ArchDispatch.h.
//
// Skeleton refs: §3.4 (hardware capabilities), §8.7 (co-residency),
//                §4.4 (cost model calibration constants).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tilemega {

/// Runtime description of a compilation / execution target.
///
/// Two ways to obtain one:
///   * TargetSpec::Probe()          -- query the GPU present in this machine.
///   * TargetSpec::FromJson(path)   -- load configs/targets/sm_XX.json, which
///                                     works even without the hardware present
///                                     (cross compilation, cost modelling).
struct TargetSpec {
  /// "sm_80" | "sm_89" | "sm_90" | "sm_120"
  std::string arch_tag;
  int sm_major = 0;
  int sm_minor = 0;

  /// Capability switches.  Business code queries these booleans; it must not
  /// derive them from (sm_major, sm_minor) at the use site.
  struct Caps {
    bool cluster          = false;  ///< Thread Block Cluster + distributed smem
    bool tma              = false;  ///< Tensor Memory Accelerator (cp.async.bulk)
    bool warp_specialized = false;  ///< CUTLASS warp-specialized GEMM collectives
    bool tcgen05          = false;  ///< 5th-gen tensor core / TMEM (sm_100 only)
    bool l1_5             = false;  ///< Blackwell L1.5 / LRC layer (sm_100 only)
    bool net              = false;  ///< inter-GPU fabric in the execution domain
    bool cp_async         = false;  ///< cp.async (sm_80+)
    bool mbarrier         = false;  ///< PTX mbarrier primitives
  } caps;

  /// Resource budgets.  All of these differ between consumer and datacenter
  /// parts; max_smem_per_sm in particular differs by ~2x, which directly
  /// bounds TaskBody `Stages` (skeleton §5.3).
  struct Res {
    int num_sms                  = 0;
    int max_smem_per_sm          = 0;  ///< bytes
    int max_dynamic_smem_per_cta = 0;  ///< bytes, after opt-in
    int regs_per_sm              = 0;
    int max_cluster_size         = 1;  ///< 1 when caps.cluster == false
    int max_threads_per_sm       = 0;
    int warp_size                = 32;
  } res;

  /// One calibrated quantity together with the evidence behind it.  The cost
  /// model reads `value`; the rest travels with it so a constant can never be
  /// cited without its method, sample count and dispersion (§0 work
  /// discipline).
  struct Measurement {
    std::string name;
    double value = 0.0;
    std::string unit;
    int samples = 0;
    double rel_stddev = 0.0;  ///< stddev / mean over `samples` repeats
    std::string method;
  };

  /// Stream-K's per-CTA four-parameter model (arXiv 2301.03598 A.1),
  ///   time_CTA = a + b*[peers>1] + c*iters + d*(peers-1).
  ///
  /// `a` and `c` are per CTA and fitted per tile shape.  `b` and `d` are the
  /// reduction, which in TileMega is the separate GemmCombine stage and does
  /// not depend on the tile shape at all, so they are held per output element
  /// and the per-CTA form is
  ///   time_CTA = a + c*iters + (b + d*(peers-1)) * live_outputs(tile)
  ///            + calib.combine_fixed_ns / ctas_in_the_stage.
  /// Keeping them per element is what makes them measurable: fitted at one
  /// shape's single output width the reduction moves 32 KB, below the timer's
  /// noise floor, and the fit returned negative d at r^2 = 0.05.
  struct StreamKPoint {
    int tile_m = 0, tile_n = 0, tile_k = 0, stages = 0;
    double a_ns = 0.0, b_ns = 0.0, c_ns = 0.0, d_ns = 0.0;
    double fit_r2 = 0.0;  ///< the worse of the two fits behind this point
    double ac_r2 = 0.0;   ///< of the per-CTA a/c fit alone
    /// `a` and `c` re-fitted at a full-width grid of `occ_per_sm` CTAs per SM.
    /// The sweep above runs six CTAs on 128 SMs, which is one CTA alone on an
    /// idle SM and an idle memory system -- the regime the megakernel is never
    /// in.  A tile whose mainloop is latency-bound gets a second CTA per SM
    /// almost free; one that saturates a pipeline pays for it in full, and
    /// only a measurement separates the two.
    std::vector<double> occ_per_sm;
    std::vector<double> occ_a_ns;
    std::vector<double> occ_c_ns;
  };

  /// Cost-model calibration, filled by tools/tilemega-calibrate (skeleton
  /// §4.4).  `calibrated` gates its use: every field is zero until a real
  /// measurement wrote it, and a fabricated constant is worse than none.
  ///
  /// Rates are per nanosecond throughout, so a resource-vector component is
  /// work / rate with no unit conversion (1 GB/s == 1 byte/ns).
  struct Calib {
    // (a) pipeline rates -- the denominators of u(o)'s components.
    double tc_fp16_gflops   = 0.0;  ///< mma.m16n8k16 f16 in, f32 accumulate
    double cuda_fp32_gflops = 0.0;  ///< FFMA
    double cuda_int32_gops  = 0.0;  ///< IMAD
    double sfu_exp2_gops    = 0.0;
    double sfu_rsqrt_gops   = 0.0;
    double smem_gbps        = 0.0;  ///< conflict-free ld.shared
    /// Bank-conflict cost multiplier: max(1, smem_conflict_slope * ways).
    /// Not an increment per way -- on a scalar ld.shared pipeline the first
    /// conflicting way can be free, so a line anchored at 1.0 misfits.
    double smem_conflict_slope = 0.0;
    /// Scalar ld.shared throughput against resident CTAs per SM.
    /// `smem_gbps` above is the full-occupancy plateau, which the SIMT GEMM
    /// mainloop never sees: it runs at one or two CTAs per SM, where the same
    /// loop reaches a third of that rate.  The cost model interpolates this
    /// curve at the configuration's own occupancy instead.
    std::vector<double> smem_occupancy_ctas;
    std::vector<double> smem_occupancy_gbps;
    /// Dependent-load round trip, from a pointer chase whose stride defeats
    /// the prefetcher.  Every non-GEMM stage in the reference models owns a
    /// few hundred elements per CTA: its cost is a handful of these latencies,
    /// not a bandwidth, and a model built only on rates predicts zero for it.
    double l1_latency_ns    = 0.0;  ///< working set inside one SM's L1
    double l2_latency_ns    = 0.0;  ///< inside the L2, past L1
    double dram_latency_ns  = 0.0;  ///< past the L2 knee
    double l2_gbps          = 0.0;  ///< plateau below the capacity knee
    double l2_knee_bytes    = 0.0;  ///< measured working-set knee
    double dram_gbps        = 0.0;  ///< achieved above the knee, not peak
    std::vector<double> l2_curve_bytes;  ///< working-set sweep, x axis
    std::vector<double> l2_curve_gbps;   ///< working-set sweep, y axis

    // (b) synchronization latencies, nanoseconds.
    double atomic_uncontended_ns = 0.0;
    std::vector<double> atomic_contention_ctas;  ///< contending CTA count
    std::vector<double> atomic_contention_ns;    ///< ns per completed atomic
    double threadfence_ns   = 0.0;
    double syncthreads_ns   = 0.0;
    double named_barrier_ns = 0.0;
    double cluster_sync_ns  = 0.0;
    bool   cluster_sync_calibrated = false;  ///< false on targets without clusters
    /// The megakernel's own stage barrier (§8.2: release fence, CTA barrier,
    /// one monotonic arrival, epoch publish, backoff poll) against the
    /// resident CTA count.  It is a composite of the primitives above, but
    /// not their sum: the poll is a contended global atomic whose cost is set
    /// by the grid, so the cost model reads it as one measured quantity.
    std::vector<double> grid_barrier_ctas;
    std::vector<double> grid_barrier_ns;

    // (c) Stream-K coefficients, one entry per calibrated tile shape.
    std::vector<StreamKPoint> streamk;
    /// Width-independent part of the reduction stage, launch excluded.
    double combine_fixed_ns = 0.0;
    /// The per-peer coefficient past the L2 knee.  `StreamKPoint::d_ns` is
    /// the L2-resident value, which is the regime the reference models
    /// reduce in; these differ by 5.6x on sm_89 and must not be averaged.
    double combine_d_dram_ns = 0.0;

    // (d) concurrency interference: a GEMM's duration next to a memory-bound
    // task, relative to the same GEMM alone.  1.0 means the independent
    // duration assumption holds; skeleton §4.4 calls >1.3 its failure point.
    double interference_ratio = 0.0;

    bool calibrated = false;
    std::string device;      ///< the GPU name the numbers came from
    std::string measured_at; ///< ISO-8601 UTC
    double wall_seconds = 0.0;
    std::vector<Measurement> measurements;

    /// The Stream-K entry for a tile shape, or nullptr when that shape was
    /// not calibrated.  Callers must not interpolate silently -- ask, and
    /// handle the absence.
    StreamKPoint const* FindStreamK(int m, int n, int k, int stages) const;
  } calib;

  /// Probe the GPU at `device_ordinal`.  Throws std::runtime_error when no
  /// CUDA device is available.
  static TargetSpec Probe(int device_ordinal = 0);

  /// Load from a configs/targets/*.json file.  No hardware required.
  static TargetSpec FromJson(std::string const& path);

  /// Serialize back to the same JSON schema (used by tilemega-calibrate).
  std::string ToJson() const;

  /// Serialize directly to `path`, replacing an existing calibration file.
  void ToJson(std::string const& path) const;

  /// "sm_89" -> the value nvcc wants for -arch=
  std::string NvccArch() const { return arch_tag; }

  /// Compile-time-ish helper used by TaskBody instantiation: how many pipeline
  /// stages fit in the dynamic smem budget for a given per-stage byte count.
  /// Never call with a hard-coded budget -- pass res.max_dynamic_smem_per_cta.
  static int ComputeStages(int smem_budget_bytes,
                           int bytes_per_stage,
                           int fixed_overhead_bytes = 0,
                           int max_stages = 16);

  /// Query the CUDA occupancy calculator for a concrete kernel.  Occupancy is
  /// kernel-specific and therefore cannot be populated by Probe() alone; this
  /// companion keeps the calculation attached to the probed target (§8.7).
  int ActiveBlocksPerSM(void const* kernel,
                        int block_size,
                        std::size_t dynamic_smem_bytes = 0) const;

  /// Maximum simultaneously resident grid for a non-cluster kernel. Cluster
  /// launches must instead use cudaOccupancyMaxActiveClusters (see V-C).
  int ResidentGridLimit(void const* kernel,
                        int block_size,
                        std::size_t dynamic_smem_bytes = 0) const;

  /// Human-readable one-liner for logs / result.md tables.
  std::string Summary() const;
};

}  // namespace tilemega
