// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/DnnModelPlan.h>
#include <tilemega/Solver/SkeletonSearch.h>

namespace tilemega::solver {
struct DnnStructureChoice {
  std::set<unsigned> fused_gemms;
  std::set<frontend::DeferredLayerNormEdge> deferred_edges;
  std::string reuse;
  unsigned small_channels=8;
  int reduction_mask=-1;
  std::string Key() const;
};
struct DnnStructureEvaluation {
  DnnStructureChoice choice;
  std::string error;
  double score=std::numeric_limits<double>::infinity();
  std::vector<OperatorClass> classes;
  std::vector<SkeletonCandidate> top;
};
struct DnnStructureSearchResult {
  std::vector<DnnStructureEvaluation> evaluated;
  std::size_t winner=0;
  bool budget_exhausted=false;
  int rounds=0;
};
// Geometry, residency and page coordinates are rescored on each rebuilt graph.
// The budget is soft at frontend/import boundaries, as in the inner search.
DnnStructureSearchResult SearchDnnStructures(frontend::ImportedSemantics const&,
    frontend::DnnPlanOptions const&,std::vector<std::string> const& reuse_choices,
    mlir::MLIRContext&,SkeletonSearchOptions const&,std::ostream&);
frontend::ImportedSemantics RebuildDnnStructure(frontend::ImportedSemantics const&,
    frontend::DnnPlanOptions,DnnStructureChoice const&);
SkeletonSearchResult SolveDnnStructures(frontend::ImportedSemantics const&,
    frontend::DnnPlanOptions const&,std::vector<std::string> const& reuse_choices,
    mlir::MLIRContext&,SkeletonSearchOptions const&,
    frontend::ImportSummary*,std::ostream&);
}
