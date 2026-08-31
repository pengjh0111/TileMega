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
//     BAD :  if (sm_major >= 9)            // sm_120 breaks this
//     BAD :  constexpr int kNumSMs = 128;
//     GOOD:  if (target.caps.cluster)
//     GOOD:  int grid = target.res.num_sms * occupancy;
//
// The *device-side* compile-time counterpart lives in Target/ArchDispatch.h,
// which is the only file in the project allowed to test __CUDA_ARCH__.
//
// Skeleton refs: §3.4 (hardware capabilities), §8.7 (co-residency),
//                §4.4 (cost model calibration constants).

#pragma once

#include <string>
#include <cstdint>

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
    bool cp_async         = false;  ///< cp.async (sm_80+)
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

  /// Cost-model calibration.  Filled by tools/tilemega-calibrate (Phase 4,
  /// skeleton §4.4).  Placeholder for now; `calibrated` gates its use.
  struct Calib {
    double atomic_latency_ns       = 0.0;
    double cluster_sync_latency_ns = 0.0;
    double named_barrier_ns        = 0.0;
    double hbm_bandwidth_gbps      = 0.0;
    bool   calibrated              = false;
  } calib;

  /// Probe the GPU at `device_ordinal`.  Throws std::runtime_error when no
  /// CUDA device is available.
  static TargetSpec Probe(int device_ordinal = 0);

  /// Load from a configs/targets/*.json file.  No hardware required.
  static TargetSpec FromJson(std::string const& path);

  /// Serialize back to the same JSON schema (used by tilemega-calibrate).
  std::string ToJson() const;

  /// "sm_89" -> the value nvcc wants for -arch=
  std::string NvccArch() const { return arch_tag; }

  /// Compile-time-ish helper used by TaskBody instantiation: how many pipeline
  /// stages fit in the dynamic smem budget for a given per-stage byte count.
  /// Never call with a hard-coded budget -- pass res.max_dynamic_smem_per_cta.
  static int ComputeStages(int smem_budget_bytes,
                           int bytes_per_stage,
                           int fixed_overhead_bytes = 0,
                           int max_stages = 16);

  /// Human-readable one-liner for logs / result.md tables.
  std::string Summary() const;
};

}  // namespace tilemega
