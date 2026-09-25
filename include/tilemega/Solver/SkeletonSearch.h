// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/CompilerSearch.h>
#include <tilemega/Solver/SkeletonPlacement.h>
#include <tilemega/Solver/VariantResourceCache.h>
#include <tilemega/Solver/FlowPreparation.h>

namespace tilemega::solver {
struct SkeletonSearchOptions {
  CompilerSearchOptions common;
  GemmConfig seed;
  int kappa=1,k_base=8,passes=3,jobs=1;
  bool all_workers=false;
  VariantResourceCache::Probe variant_probe;
  std::string artifact_prefix,fixture;
  int seed_residency=1,top_m=8;
  bool pure_template=false,search_only=false;
};
struct SkeletonCandidate {
  std::string key,error;
  std::vector<GemmConfig> config;
  double score=std::numeric_limits<double>::infinity();
  int residency=0,estimated_limit=0,actual_limit=0,kappa=1;
  SkeletonPlacementStats placement;
};
struct SkeletonSearchResult {
  CompilerSearchResult compiled;
  std::vector<OperatorClass> classes;
  std::vector<SkeletonCandidate> evaluated,top;
  int rounds=0;
};
struct SkeletonSolvedPoint {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  SymbolicProblem problem;
  PlanSkeleton skeleton;
  EftSchedule schedule;
  SkeletonCandidate candidate;
  std::optional<PreparedFlow> flow;
};
// This is the sole bridge to the explicit graph. Called only after top-K.
CompilerSearchResult::ShortlistEntry FinalizeSkeletonPoint(SkeletonSolvedPoint&& point,
    SkeletonSearchOptions const& options,std::string const& prefix);
SkeletonSearchResult SolveSkeletonExport(std::string const& path,mlir::MLIRContext& context,
    SkeletonSearchOptions const& options,frontend::ImportSummary* summary,std::ostream& evidence);
}
