// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ServingPruning.h>

#include <cassert>
#include <iostream>

int main() {
  using namespace tilemega::solver;
  tilemega::TargetSpec target;
  GemmConfig g{16, 128, 64, 2, 1};
  target.res.max_dynamic_smem_per_cta =
      ServingBF16SmemBytes(g.tile_m, g.tile_n, g.tile_k, g.stages);
  ServingPruneContext decode{1, 3072, 2048, 0, 64, &target};
  assert(!PruneServingR1(g, decode));
  assert(!PruneServingR2(g, decode));
  assert(PruneServingR2(GemmConfig{32, 128, 64, 2, 1}, decode));
  assert(PruneServingR2(GemmConfig{16, 128, 64, 33, 1}, decode));
  assert(PruneServingR1(GemmConfig{16, 128, 64, 2, 64}, decode));
  ServingPruneContext gate = decode;
  gate.gate_interleave_u = 16;
  assert(!PruneServingR1(g, gate));
  assert(PruneServingR1(GemmConfig{16, 16, 64, 2, 1}, gate));
  target.res.max_dynamic_smem_per_cta = 1024;
  assert(PruneServingR1(g, decode));
  target.res.max_dynamic_smem_per_cta =
      ServingBF16SmemBytes(g.tile_m, g.tile_n, g.tile_k, g.stages);
  GemmConfig split{16, 32, 64, 2, 4};
  assert(!PruneServingR3(split, decode, false));
  assert(PruneServingR3(split, decode, true));
  auto order = MakeServingSearchOrderR4(3);
  assert(order.initial_kappa == 1 && order.initial_residency == 3);
  assert((order.kappa_scan == std::vector<int>{1, 2, 4}));
  assert((order.residency_scan == std::vector<int>{1, 2, 3}));
  std::cout << "SERVING_PRUNING_R1_R4 PASS\n";
}
