// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §4.1 chain dynamic programming, §4.3 the global part of the
//                state (resident CTAs per SM).
//
// DP[i][s] = min_{s'} { DP[i-1][s'] + Cost_i(s) + Interface(s', s) } over the
// GEMM operators in the order the megakernel executes them.  Execution order
// is the right chain: L1 puts a grid barrier after every stage, so the DAG's
// forks (gate/up) are already linearised by the generator and a chain over the
// emitted stage list is exact rather than an approximation of the DAG.
//
// The one quantity that is not per-operator is `resident_tiles_per_SM`: it
// follows from the *union* of every TaskBody's shared memory and the *maximum*
// of their register counts, so one operator's tile shape can cost every other
// operator an SM slot (§4.3).  The solver handles it by fixing the residency
// in an outer loop and admitting only states that reach it, which keeps the
// inner problem exact instead of approximating a global constraint locally.
#pragma once

#include <tilemega/Solver/CostModel.h>

#include <vector>

namespace tilemega::solver {

/// One point of the per-operator search space, with the two tier-3 numbers no
/// closed form supplies.  `registers` comes from a ptxas log; a candidate
/// without one cannot be placed because its occupancy is unknown.
struct DpCandidate {
  GemmConfig config;
  int registers = 0;
  int smem_bytes = 0;
};

/// How much of the state each operator is allowed to choose for itself.
enum class DpMode {
  /// One configuration for the whole model -- the oracle's search space, and
  /// therefore what acceptance (a) compares against.
  kUniform,
  /// One tile shape for the whole model, but a per-GEMM split factor.  This is
  /// the part of a per-operator plan that costs nothing to build: the split is
  /// a host-side loop bound in ModelHarness, not a template argument, so it
  /// varies per GEMM in the megakernel that exists today.
  kPerOperatorSplit,
  /// The full per-operator state, which needs one compiled GemmImpl variant
  /// per distinct tile shape.
  kPerOperator,
};

struct ChainDpOptions {
  DpMode mode = DpMode::kPerOperator;
  /// The pair-dependent part of §4.1's Interface term -- the L2 carry.  The
  /// stage sequence between two GEMMs is charged either way, because it is a
  /// constant of the model rather than a coupling; this switch measures what
  /// the coupling itself is worth.
  bool interface_term = true;
  /// Run the full O(L*|C|^2) transition loop even when the interface term is
  /// separable.  Off takes the per-operator minimum directly; both must agree.
  bool general_transitions = true;
  int max_ctas_per_sm = 8;
  /// Tier-2 admissibility (§3.2): for each GEMM in model order, the global
  /// candidate indices alignment propagation left standing.  Empty -- the
  /// default -- admits every candidate at every operator, which is what the
  /// tier-1 solver did before propagation existed.
  std::vector<std::vector<int>> per_operator_candidates;
};

struct ChainDpStats {
  int candidates = 0;
  int operators = 0;
  int residency_levels = 0;       ///< levels that admitted at least one state
  long long transitions = 0;      ///< s' -> s pairs actually relaxed
  double solve_ms = 0.0;
  /// |DP objective - CostModel::Evaluate(chosen)|.  The decomposition into
  /// prefix + sum of per-operator costs + interfaces + suffix has to reproduce
  /// the whole-model evaluation exactly, or the DP is minimizing something
  /// other than the cost model.
  double decomposition_error_ns = 0.0;
  /// max over (i, s) of the spread of Interface(., s) across s'.  Zero means
  /// the chain separates and the DP degenerates to |R| independent minima.
  double interface_spread_ns = 0.0;
};

struct ChainDpSolution {
  bool feasible = false;
  std::vector<GemmConfig> configs;  ///< one per GEMM, in model order
  Residency residency;
  CostBreakdown cost;
  int max_smem_bytes = 0;
  int max_registers = 0;
};

class ChainDP {
 public:
  ChainDP(CostModel const& model, std::vector<DpCandidate> candidates);

  ChainDpSolution Solve(ModelDescription const& model, ChainDpOptions options,
                        ChainDpStats* stats = nullptr) const;

  /// Everything the megakernel executes between GEMM `from` and GEMM `to`,
  /// plus what the boundary itself costs.  `from < 0` is the model prefix and
  /// `to < 0` the suffix.
  double Interface(ModelDescription const& model, int from, int to,
                   GemmConfig const& from_config, GemmConfig const& to_config,
                   Residency residency) const;

  /// F-40, the occupancy closed form, verified 1075/1077 on this target.
  int CtasPerSm(int smem_bytes, int registers) const;

  std::vector<DpCandidate> const& candidates() const { return candidates_; }

 private:
  /// Stage indices of the GEMMs, in execution order.
  std::vector<int> GemmStages(ModelDescription const& model) const;

  /// The stages the megakernel runs strictly between two GEMMs, with their
  /// barriers.  Independent of both configurations.
  double BetweenNs(ModelDescription const& model,
                   std::vector<int> const& gemm_stages, int from, int to,
                   Residency residency) const;

  /// What the boundary itself costs: the producer's output re-fetched from
  /// DRAM instead of the L2, for whatever `miss` fraction of it did not
  /// survive.  The miss fraction is a property of the model, so the caller
  /// computes it once rather than once per relaxed transition.
  double CarryNs(ModelDescription const& model,
                 std::vector<int> const& gemm_stages, int from, int to,
                 double miss, GemmConfig const& from_config,
                 GemmConfig const& to_config) const;

  CostModel const* cost_ = nullptr;
  std::vector<DpCandidate> candidates_;
};

}  // namespace tilemega::solver
