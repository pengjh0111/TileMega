// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/TaskModel.h>
namespace tilemega::solver {
struct PricePiece {
  analysis::CouplingRelation domain;
  analysis::QuasiPolynomial count;
  analysis::ParamBinding representative;
  TaskPriceParts parts;
};
struct PiecePrices {
  std::vector<PricePiece> pieces;
  bool coordinate_varying=false;
  double total_isolated_ns=0;
};
struct PiecePriceCache {
  std::map<std::string,PiecePrices> entries;
  std::uint64_t hits=0,misses=0;
};
PiecePrices PriceBoundaryPieces(CostModel const& cost,DerivedTaskInput const& input,
    ModelTaskSemantics const& semantic,BackendTraits const& traits,Residency residency,
    ModelDescription const& model,int chunks,PiecePriceCache* cache=nullptr);
} // namespace tilemega::solver
