// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/SkeletonSearch.h>

namespace tilemega::solver {
struct MoeBindingChoice {
  bool grouped=false;
  unsigned block_rows=1;
  std::string Key() const;
};
struct MoeStructureEvaluation {
  MoeBindingChoice choice;
  double score=std::numeric_limits<double>::infinity();
  std::string error;
  std::vector<SkeletonCandidate> top;
};
struct MoeStructureSearchResult {
  std::vector<MoeStructureEvaluation> evaluated;
  std::size_t winner=0;
  bool budget_exhausted=false;
};
frontend::ImportedSemantics RebuildMoeStructure(frontend::ImportedSemantics const&,
    MoeBindingChoice const&);
MoeStructureSearchResult SearchMoeStructures(frontend::ImportedSemantics const&,
    std::vector<MoeBindingChoice> const&,mlir::MLIRContext&,
    SkeletonSearchOptions const&,std::ostream&);
SkeletonSearchResult SolveMoeStructures(frontend::ImportedSemantics const&,
    std::vector<MoeBindingChoice> const&,mlir::MLIRContext&,
    SkeletonSearchOptions const&,frontend::ImportSummary*,std::ostream&);
}
