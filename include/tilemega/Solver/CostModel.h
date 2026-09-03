// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2.2 (hierarchical tile-pipeline model), §2.3 (Stream-K
//                split), §4.4 (calibrated backend cost model).
//
// §2.1's roofline form is not used, and not because it is inelegant: ranked
// against the oracle's 1077 measured configurations it reaches Spearman 0.44
// and places none of its top three inside the measured top 3%, against 0.945
// for the model here.  The ablation in docs/experiments/COST_MODEL/result.md
// carries that number and the rest of the ladder.
//
// §2.2(b)'s fill depth d = stages * resident_tiles_per_SM - 1 defaults off:
// the calibration measures a line in `iters`, whose intercept and d*c are not
// separately identifiable, so recovering a per-CTA setup constant by adding
// d*c back injects c's occupancy scaling into it (483 ns rms -> 1449 ns) and
// costs rank.  The prologue and epilogue terms are always charged.
//
// Every coefficient here comes from TargetSpec::Calib.  Two shape-generalizing
// scalars (the per-instruction cost of a scalar ld.shared, and the kernel's
// fixed setup) are least-squares fits over the *calibrated* Stream-K table at
// construction, never over measured end-to-end latencies: the validation set
// and the model must not share a number.
#pragma once

#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Target/TargetSpec.h>

#include <vector>

namespace tilemega::solver {

/// §2.2(a) u(o): nanoseconds of occupancy each pipe owes for one unit of work.
/// The steady state is the largest component, not the sum.
struct ResourceVector {
  double tensor_core = 0.0;
  double cuda_core = 0.0;
  double sfu = 0.0;
  double smem = 0.0;
  double l2 = 0.0;
  double dram = 0.0;

  double Bottleneck() const;
  char const* BottleneckName() const;
};

/// One GEMM operator's implementation choice -- the DP's state minus the
/// coarsening factor, which P4.6 decides separately.
struct GemmConfig {
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 0;
  int stages = 0;
  int split_k = 1;
};

/// What no closed form supplies (§4.3): the megakernel's resident CTA count is
/// a global maximum over every TaskBody's register demand and the shared-memory
/// union, so it arrives from a tier-3 compile and is a property of the whole
/// kernel rather than of one operator.
struct Residency {
  int ctas_per_sm = 1;
};

struct CostBreakdown {
  double total_ns = 0.0;
  double gemm_ns = 0.0;
  double combine_ns = 0.0;
  double other_ns = 0.0;
  double barrier_ns = 0.0;
  int stage_count = 0;  ///< after the split rewrite, so one barrier each
};

/// Ablation switches, one per layer of §2.2.  The default is every layer on;
/// the validation tool walks them to produce §2.4's ladder.
struct CostModelOptions {
  bool resource_lanes = true;     ///< §2.2(a): all lanes, not the smem lane alone
  /// §2.2(b)'s fill depth `d = stages * resident_tiles_per_SM - 1`.  Off by
  /// default: `d` is not identifiable from the calibration on this target --
  /// see the class comment and COST_MODEL/result.md.  The prologue and
  /// epilogue terms of the envelope are always charged.
  bool pipeline_envelope = false;
  bool wave_tail = true;          ///< §2.2(d): tail wave at its own active-SM count
  bool cache_model = true;        ///< §2.2(e): SDCM hit probability feeds the DRAM lane
  bool split_k = true;            ///< §2.3
  bool non_gemm = true;           ///< the non-GEMM stages' latency model
  bool sync = true;               ///< §2.2(f)
};

class CostModel {
 public:
  /// The two scalars the model fits from `target.calib.streamk` at
  /// construction, kept with their residuals so neither can be quoted alone.
  struct Fit {
    double lds_ns = 0.0;        ///< per scalar ld.shared warp-instruction
    double lds_rel_rms = 0.0;
    double setup_ns = 0.0;      ///< per-CTA kernel setup, traffic excluded
    double setup_rms_ns = 0.0;  ///< absolute, because `setup` crosses zero
    int points = 0;             ///< (shape, CTAs/SM) pairs behind both fits
  };

  explicit CostModel(TargetSpec const& target, CostModelOptions options = {});

  /// `configs` holds one entry per model GEMM, so a per-operator solution and
  /// a uniform one evaluate through the same path.
  CostBreakdown Evaluate(ModelDescription const& model,
                         std::vector<GemmConfig> const& configs,
                         Residency residency) const;

  /// Uniform `g`: every GEMM gets the same configuration.
  CostBreakdown Evaluate(ModelDescription const& model, GemmConfig config,
                         Residency residency) const;

  /// The pieces, exposed because the chain DP scores one operator at a time.
  double GemmStageNs(GemmOp const& gemm, GemmConfig const& config,
                     Residency residency, ModelDescription const& model,
                     int* chunks_out = nullptr) const;
  double CombineStageNs(GemmOp const& gemm, int chunks,
                        ModelDims const& dims) const;
  double NonGemmStageNs(ModelStage const& stage, ModelDims const& dims,
                        Residency residency) const;
  /// §2.2(f): one stage barrier, at this grid width.
  double BarrierNs(Residency residency) const;

  /// The steady-state resource vector of one mainloop iteration for `o`
  /// resident CTAs, before the envelope is applied.
  ResourceVector Steady(GemmConfig const& config, double ctas_per_sm,
                        double dram_fraction) const;

  /// §2.2(e): probability that a line of a `footprint`-byte working set is
  /// still in the L2 when it is next touched, by the SDCM Gaussian form.
  double CacheHitProbability(double footprint_bytes) const;

  /// Number of K chunks this configuration actually splits into.
  int Chunks(GemmOp const& gemm, GemmConfig const& config) const;

  Fit const& fit() const { return fit_; }
  CostModelOptions const& options() const { return options_; }
  TargetSpec const& target() const { return *target_; }

 private:
  double WavesNs(double per_sm_work_count, Residency residency,
                 GemmConfig const& config, double iters,
                 double dram_fraction) const;

  TargetSpec const* target_ = nullptr;
  CostModelOptions options_;
  Fit fit_;
  double l2_bytes_per_ns_per_sm_ = 0.0;
  double dram_bytes_per_ns_per_sm_ = 0.0;
  double cuda_flops_per_ns_per_sm_ = 0.0;
  double sfu_ops_per_ns_per_sm_ = 0.0;
  double tc_flops_per_ns_per_sm_ = 0.0;
};

}  // namespace tilemega::solver
