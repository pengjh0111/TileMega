// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §4.2 candidate enumeration, P4.2 legality pruning.
#pragma once

#include <tilemega/Solver/BackendCostQuery.h>

#include <vector>
#include <memory>

namespace tilemega::solver {

struct TileCandidate {
  int m = 0;
  int n = 0;
  int k = 0;
  int stages = 0;
};

/// One operator's GEMM extents, in the CUTLASS coordinate contract: A is
/// (M,K), B is (N,K).
struct GemmProblem {
  int m = 0, n = 0, k = 0;
};
struct AscendGemmEstimate {
  double service_makespan_ns = 0;
  int groups = 0;
  std::string provenance;
};
class AscendGemmPriceProvider {
 public:
  virtual ~AscendGemmPriceProvider() = default;
  virtual AscendGemmEstimate Estimate(BackendCandidate const&,GemmProblem const&) const = 0;
};

/// Tier 2: a legacy FFMA estimate or a native Ascend estimate in nanoseconds.
/// Structural on purpose -- it uses only budgets TargetSpec actually carries,
/// never an uncalibrated bandwidth or clock (calib.calibrated is false until
/// P4.4 lands).
struct AnalyticalCost {
  int ctas = 0;
  int co_resident = 0;    ///< CTAs of this shape resident per SM
  int waves = 0;
  int k_iters = 0;
  double cta_cycles = 0;  ///< per CTA per K-tile: tile_m * tile_n * tile_k / threads
  double cycles = 0;      ///< waves * k_iters * cta_cycles
  std::optional<AscendGemmEstimate> ascend;
};

/// The search envelope, i.e. how far an axis may be doubled before the walk
/// gives up on it even if shared memory still allows it.
struct SearchEnvelope {
  int max_tile_mn = 256;
  int max_tile_k = 32;
  int max_stages = 5;
};

/// What the walk touched, so the saving over a Cartesian enumeration is a
/// measured number and not a claim.
struct EnumerationStats {
  int touched = 0;         ///< shapes whose traits were evaluated
  int wall_pruned = 0;     ///< subtrees cut at the shared-memory wall
  int shape_rejected = 0;  ///< touched but rejected by a shape or budget rule
  int legal = 0;
  int cartesian = 0;       ///< |envelope|, what a product enumeration costs
};

/// §4.2's enumeration. The walk expands the native MMA tile along each axis
/// and stops at the resource wall, so the candidate set is the legal region
/// itself rather than a Cartesian product filtered afterwards.
class CandidateGenerator {
 public:
  using Envelope = SearchEnvelope;
  using Stats = EnumerationStats;

  explicit CandidateGenerator(TargetSpec target, Envelope envelope = Envelope());
  CandidateGenerator(TargetSpec target, ScalarType dtype,
                     Envelope envelope = Envelope());

  std::vector<BackendCandidate> Enumerate(Stats* stats = nullptr) const;
  void SetAscendDomain(std::vector<BackendCandidate>,std::shared_ptr<AscendGemmPriceProvider const>);

  AnalyticalCost Analyze(BackendCandidate const& candidate,
                         GemmProblem const& problem) const;
  /// Total native service nanoseconds (or legacy cycles), never mixed units.
  double RankKey(BackendCandidate const& candidate,
                 std::vector<GemmProblem> const& problems) const;
  /// The `k` cheapest candidates, ties broken towards less shared memory
  /// (co-residency is what the other task kinds of the megakernel compete for).
  std::vector<BackendCandidate> TopK(std::vector<BackendCandidate> const& candidates,
                                     std::vector<GemmProblem> const& problems,
                                     int k) const;

  std::vector<TileCandidate> Generate() const;

  TargetSpec const& target() const { return target_; }
  ScalarType dtype() const { return dtype_; }
  char const* backendName() const {
    if(target_.arch_tag=="ascend_3510")return "catlass.ascend_3510.bf16_f32acc";
    return dtype_ == ScalarType::kBF16 ? kTensorBF16Backend : kSimtF32Backend;
  }

 private:
  TargetSpec target_;
  Envelope envelope_;
  ScalarType dtype_ = ScalarType::kF32;
  std::vector<BackendCandidate> ascend_candidates_;
  std::shared_ptr<AscendGemmPriceProvider const> ascend_prices_;
};

}  // namespace tilemega::solver
