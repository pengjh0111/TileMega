// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Solver/BackendCostQuery.h>
#include <tilemega/Solver/CostModel.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace tilemega::solver {

struct ServingPruneContext {
  int m = 0, n = 0, k = 0;
  int gate_interleave_u = 0;
  int max_worker_count = 0;
  TargetSpec const* target = nullptr;
};

inline int ServingNextPow2(int value) {
  if (value <= 1) return 1;
  unsigned result = 1;
  while (result < static_cast<unsigned>(value)) result <<= 1;
  return static_cast<int>(result);
}

// R-1: legality and resources, independent of the model score.
inline bool PruneServingR1(GemmConfig const& g,
                           ServingPruneContext const& context) {
  if (!context.target || context.m < 1 || context.n < 1 || context.k < 1 ||
      g.split_k < 1 || !ServingBF16ShapeLegal(
          g.tile_m, g.tile_n, g.tile_k, g.stages)) return true;
  if (ServingBF16SmemBytes(g.tile_m, g.tile_n, g.tile_k, g.stages) >
      context.target->res.max_dynamic_smem_per_cta) return true;
  if (context.gate_interleave_u > 0 &&
      g.tile_n % (2 * context.gate_interleave_u)) return true;
  // Each split must have at least one full K tile, including the shortest.
  return context.k / g.split_k < g.tile_k;
}

// R-2: larger tiles or pipeline depth can only add inactive padding.
inline bool PruneServingR2(GemmConfig const& g,
                           ServingPruneContext const& context) {
  if (g.tile_m > std::max(16, ServingNextPow2(context.m)) ||
      g.tile_n > std::max(32, ServingNextPow2(context.n))) return true;
  int const k_tiles = (context.k + g.split_k * g.tile_k - 1) /
                      (g.split_k * g.tile_k);
  return g.stages > k_tiles;
}

// R-3 is a model-level dominance hypothesis. It is admitted only after the
// unpruned-search equivalence experiment; the caller controls enablement.
inline bool PruneServingR3(GemmConfig const& g,
                           ServingPruneContext const& context,
                           bool enabled) {
  if (!enabled || g.split_k == 1) return false;
  int const output_tiles =
      ((context.m + g.tile_m - 1) / g.tile_m) *
      ((context.n + g.tile_n - 1) / g.tile_n);
  return (context.max_worker_count > 0 &&
          output_tiles >= context.max_worker_count) ||
         context.k / (g.split_k * g.tile_k) < 2;
}

// R-4: class coordinates start with the minimum event coarsening and the
// highest legal residency; the two global coordinates are scanned afterward.
struct ServingSearchOrderR4 {
  int initial_kappa = 1;
  int initial_residency = 0;
  std::vector<int> kappa_scan{1, 2, 4};
  std::vector<int> residency_scan;
};

inline ServingSearchOrderR4 MakeServingSearchOrderR4(int resident_limit) {
  ServingSearchOrderR4 result;
  result.initial_residency = resident_limit;
  for (int value = 1; value <= resident_limit; ++value)
    result.residency_scan.push_back(value);
  return result;
}

}  // namespace tilemega::solver
