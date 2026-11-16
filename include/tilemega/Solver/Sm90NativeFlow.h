// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/Sm90NativePrices.h>
#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <tilemega/Analysis/CouplingCache.h>
#include <tilemega/Solver/Sm90NativeCtg.h>
#include <string>
#include <vector>

namespace tilemega::solver {

// P2: one candidate's whole decode step as a FlowProblem, so Eq. (8)'s
// PoolScore is the shared evaluator's makespan over it. The boundaries carry
// appendix A's sync kinds, which is what makes the score sensitive to the
// Realize and Coarsen choices rather than only to the tile geometry.
enum class Sm90SyncKind {kAggregate, kGroup, kHandoff, kCounted, kData};

char const* Sm90SyncKindName(Sm90SyncKind kind);

struct Sm90NativeStageNode {
  Sm90NativeStagePrice price;
  // The boundary ahead of this stage: how its producer releases it.
  Sm90SyncKind sync = Sm90SyncKind::kAggregate;
  // Event grouping: 1 is per task, >1 coarsens the producer's completions
  // into groups (appendix A's Coarsen row).
  int kappa = 1;
  int instances = 1;
};

struct Sm90NativeFlowResult {
  double pool_score_ns = 0;      // Eq. (8) over the complete decode step
  double step_ns = 0;
  FlowProblem problem;
  json::Value approximation;
  FlowResult evaluated;
  std::vector<Sm90NativeStageNode> nodes;
  std::vector<double> stage_ns;  // per node, from FlowSpaceResult
  std::size_t cache_hits = 0, cache_misses = 0;
  double analysis_ms=0,release_ms=0,pricing_ms=0,evaluation_ms=0;
  std::size_t derivations=0,enumerated_tasks=0,enumerated_edges=0;
};

// Exact value encoding, including floating-point representations and optional
// tables. It intentionally excludes pointers, padding, and measured CPU time.
std::string Sm90NativeFlowIdentity(FlowProblem const&);

// The stages of one decode step at these coordinates, with each boundary's
// sync kind decided by the coordinate (group_wait, expert_fuse, embed_la, tp).
std::vector<Sm90NativeStageNode> Sm90NativeStages(Sm90DenseWorkload const& work,
                                                 Sm90DenseCoordinates const& coordinates,
                                                 Sm90DenseCostModel const& cost,
                                                 bool isolated = false);

double Sm90NativeOperatorSumScore(Sm90DenseWorkload const& work,
    Sm90DenseCoordinates const& coordinates,Sm90DenseCostModel const& cost);

// Eq. (8): build the FlowProblem from those stages and evaluate it.
Sm90NativeFlowResult Sm90NativePoolScore(Sm90DenseWorkload const& work,
                                         Sm90DenseCoordinates const& coordinates,
                                         Sm90DenseCostModel const& cost,
                                         analysis::CouplingCache* cache = nullptr);

Sm90NativeFlowResult Sm90NativeCtgScore(frontend::ImportedSemantics const&,
    Sm90DenseWorkload const&,Sm90DenseCoordinates const&,Sm90DenseCostModel const&,
    mlir::MLIRContext&,Sm90NativeCtgCache&,bool isolated=false);
Sm90NativeFlowResult Sm90NativeAnalyze(frontend::ImportedSemantics const&,
    Sm90DenseWorkload const&,Sm90DenseCoordinates const&,Sm90DenseCostModel const&,
    mlir::MLIRContext&,Sm90NativeCtgCache&,std::string const& mode,bool isolated=false);
}  // namespace tilemega::solver
