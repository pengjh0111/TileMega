// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/CompilerSearch.h>
#include <tilemega/Solver/SkeletonPlacement.h>
#include <tilemega/Solver/VariantResourceCache.h>
#include <tilemega/Solver/FlowPreparation.h>

namespace tilemega::solver {
struct SkeletonEvaluationCase {
  std::vector<GemmConfig> config;
  int kappa=1,residency=1;
};
struct SkeletonSearchOptions {
  CompilerSearchOptions common;
  GemmConfig seed;
  int kappa=1,k_base=8,passes=3,jobs=1;
  bool all_workers=false;
  VariantResourceCache::Probe variant_probe;
  std::string artifact_prefix,fixture;
  int seed_residency=1,top_m=8,measure_top=6;
  // Bound only the outer Level 1 scan. The caller reserves the remainder of
  // its plan budget for top-M materialization, compilation and measurement.
  int search_budget_ms=0;
  // Serving decode uses one physical plan throughout this closed past range.
  int serving_past_lo=-1,serving_past_hi=-1;
  bool pure_template=false,search_only=false,incremental_prepare=true;
  // Verification arm: keep only the R-1 legality filter.
  bool serving_pruning=true;
  bool pg_pages=false;
  // An edge class is a coordinate: bit 0 selects access-proved normalization
  // recompute, bit 1 selects access-proved attention last-arriver reduction.
  bool handoff_auto=false;
  int page_bytes=8192;
  std::vector<int> page_choices;
  std::vector<int> lookahead_choices{0,65536,131072};
  // A previous batch's winning serving plan is the second coordinate-descent
  // start. Entries are indexed by GEMM instance, then folded by SemSig class.
  std::vector<GemmConfig> serving_warm_gemms;
  std::vector<GemmConfig> paged_seed_gemms;
  int serving_warm_kappa=1,serving_warm_residency=1;
  int serving_warm_kv_block=0,serving_warm_query_rows=0;
  std::vector<SkeletonEvaluationCase> evaluation_cases;
};
struct SkeletonCandidate {
  std::string key,error;
  std::vector<GemmConfig> config;
  double score=std::numeric_limits<double>::infinity();
  int residency=0,estimated_limit=0,actual_limit=0,kappa=1;
  int shared_bytes=0;
  std::uint64_t task_count=0;
  // Serving attention coordinates. Decode searches KV extent; prefill
  // searches the query-row block. Both are literal in the ISL map.
  int attention_kv_block=0,attention_query_rows=0;
  int page_bytes=0;
  int lookahead_bytes=0;
  unsigned handoff_mask=0;
  SkeletonPlacementStats placement;
};
struct SkeletonSearchResult {
  CompilerSearchResult compiled;
  std::vector<OperatorClass> classes;
  std::vector<SkeletonCandidate> evaluated,top;
  int rounds=0;
  std::string seed_key;
  std::vector<std::string> split1_seed_keys;
  std::vector<std::string> fill_seed_keys;
};
struct SkeletonSolvedPoint {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  SymbolicProblem problem;
  PlanSkeleton skeleton;
  EftSchedule schedule;
  SkeletonCandidate candidate;
  std::optional<PreparedFlow> flow;
  std::vector<std::pair<int,PreparedFlow>> interval_flows;
};
// This is the sole bridge to the explicit graph. Called only after top-K.
CompilerSearchResult::ShortlistEntry FinalizeSkeletonPoint(SkeletonSolvedPoint&& point,
    SkeletonSearchOptions const& options,std::string const& prefix);
// Reuse the same imported semantics while changing theta bindings.
SkeletonSearchResult SolveSkeletonImported(frontend::ImportedSemantics const& imported,
    mlir::MLIRContext& context,SkeletonSearchOptions const& options,
    frontend::ImportSummary* summary,std::ostream& evidence);
SkeletonSearchResult SolveSkeletonExport(std::string const& path,mlir::MLIRContext& context,
    SkeletonSearchOptions const& options,frontend::ImportSummary* summary,std::ostream& evidence);
}
