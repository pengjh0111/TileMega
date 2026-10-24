// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <tilemega/Solver/Sm90DenseSearchSpace.h>

namespace tilemega::solver {
struct Sm90DenseSearchOptions {
  int top=4;
  // Per-family alternatives combined around the per-family optimum.
  int alternatives=3;
};
struct Sm90DenseCandidate {
  Sm90DenseCoordinates coordinates;
  Sm90DensePrediction prediction;
  std::string origin;
};
struct Sm90DenseSearchResult {
  std::vector<Sm90DenseCandidate> ranked;
  std::size_t evaluated=0;
  double elapsed_ms=0;
};
Sm90DenseSearchResult SearchSm90Dense(Sm90DenseSearchSpace const&,
    Sm90DenseWorkload const&,Sm90DenseCostModel const&,Sm90DenseSearchOptions const&);
}  // namespace tilemega::solver
