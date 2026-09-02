// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §1.2 principle 2 (the solver/backend contract), §4.3
//                implementation selection, P4.3 contract verification.
#pragma once

#include <tilemega/Analysis/AccessRelation.h>
#include <tilemega/Solver/CandidateGenerator.h>

#include <optional>
#include <string>
#include <vector>

namespace tilemega::solver {

/// One operand as the *implementation* declares it: for each tensor axis, the
/// task coordinate that advances along it and how many elements one task
/// touches there.
///
/// The axis order is the operand's logical order in the backend's own
/// coordinate contract, which for CUTLASS means B is (N,K) and not (K,N)
/// (F-17). Declaring it is the whole point: a transposed declaration and a
/// transposed kernel are indistinguishable at the task-graph level, and the
/// resulting wrong *access range* is masked by L1's global barrier -- it
/// surfaces only probabilistically once L2 removes it.
struct OperandContract {
  std::string name;
  /// Per tensor axis, the driving task coordinate; empty means "whole axis".
  std::vector<std::string> coordinate;
  /// Per tensor axis, elements touched per task.
  std::vector<int> span;
};

/// `@impl_qkv_17`: one concrete backend instantiation bound to one task space.
struct ImplementationContract {
  std::string name;
  std::string task;
  std::string backend;
  int tile_m = 0, tile_n = 0, tile_k = 0;
  ClusterShape cluster;
  int stages = 0;
  int threads = 0;
  int smem_bytes = 0;
  /// Empty until a tier-3 ptxas log fills it; see BackendCostQuery.h.
  std::optional<int> regs_est;
  AlignmentRequirement alignment;
  int arch_required = 0;
  std::vector<OperandContract> operands;

  BackendTraits DeclaredTraits() const;
};

/// The declared traits must be the backend's own answer for the declared
/// tile: an implementation may choose a shape, never restate its cost.
bool VerifyTraits(ImplementationContract const& impl, std::string* error);

/// The implementation must fit the target it claims.
bool VerifyTarget(ImplementationContract const& impl, TargetSpec const& target,
                  std::string* error);

/// Reduce the task space's own indexing map to the contract's vocabulary:
/// per tensor axis, the driving coordinate and the element span under
/// (theta, g). Fails rather than approximating a packed or unevaluable index.
bool LowerAccess(std::vector<analysis::AccessRelation> const& derived,
                 analysis::ParamBinding const& theta,
                 analysis::ParamBinding const& g,
                 std::vector<OperandContract>* out, std::string* error);

/// The one comparison §5 exists for; both front doors below reach it.
bool VerifyAccessAgainst(ImplementationContract const& impl,
                         std::vector<OperandContract> const& derived,
                         std::string* error);

/// The declared access pattern against the task space's own indexing map.
/// `derived` is BuildWriteMap / BuildReadMap output for the same operator, in
/// the same operand order.
bool VerifyAccess(ImplementationContract const& impl,
                  std::vector<analysis::AccessRelation> const& derived,
                  analysis::ParamBinding const& theta,
                  analysis::ParamBinding const& g, std::string* error);

bool VerifyContract(ImplementationContract const& impl, TargetSpec const& target,
                    std::vector<analysis::AccessRelation> const& derived,
                    analysis::ParamBinding const& theta,
                    analysis::ParamBinding const& g, std::string* error);

/// Which `g` symbol each GEMM tile axis is written as in the derived maps.
struct TileSymbols {
  std::string m = "Tm", n = "Tn", k = "Tk";
};

/// §4.3: the solver picks a granularity and an implementation together. The
/// granularity is what binds the tile symbols the derived access maps are
/// written in, so the emitted contract's spans are the ones the chosen `g`
/// actually induces -- not a second, independently authored copy.
/// Returns false when no candidate is legal on the generator's target.
bool SelectImplementation(CandidateGenerator const& generator,
                          GemmProblem const& problem, std::string name,
                          std::string task,
                          std::vector<analysis::AccessRelation> const& derived,
                          analysis::ParamBinding const& theta,
                          TileSymbols const& symbols, TileCandidate* chosen_g,
                          ImplementationContract* chosen_impl,
                          analysis::ParamBinding* chosen_binding = nullptr);

}  // namespace tilemega::solver
