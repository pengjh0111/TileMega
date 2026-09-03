// SPDX-License-Identifier: BSD-3-Clause
// P4.3: the tier-2 constraint has to come out of the model tables, so the test
// builds a model whose reader granularity is the only thing that could produce
// the answer, and checks the answer changes when that granularity does.
#include <tilemega/Solver/AlignmentPropagation.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace tilemega::solver;

#define REQUIRE(condition)                                                 \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

namespace {

/// One GEMM writing buffer 3, read by a head-wise stage of width `head`, with
/// its own input written by an RMSNorm of width 512.
ModelDescription HeadModel(int head) {
  ModelDescription model;
  model.name = "head";
  model.dims = {4, 3, 7};
  model.gemms = {{512, 512, 2, 3}};
  ModelStage norm;
  norm.kind = StageKind::kRMSNorm;
  norm.width = 512;
  norm.operands = {0, 11, 2};
  ModelStage gemm;
  gemm.kind = StageKind::kGemm;
  gemm.gemm = 0;
  ModelStage rope;
  rope.kind = StageKind::kRoPE;
  rope.extent = 4;
  rope.width = head;
  rope.operands = {3, 6, 20};
  model.stages = {norm, gemm, rope};
  return model;
}

std::vector<TileAxes> Axes() {
  std::vector<TileAxes> out;
  for (int n = 16; n <= 128; n += 16) out.push_back({n, 16});
  return out;
}

bool Kept(std::vector<int> const& kept, std::vector<TileAxes> const& axes,
          int tile_n) {
  for (int i : kept) {
    if (axes[i].tile_n == tile_n) return true;
  }
  return false;
}

}  // namespace

int main() {
  // The closed form itself, on the two cases that decide everything else.
  REQUIRE(WaitInflation(128, 64, 512) == 0);
  REQUIRE(WaitInflation(128, 48, 512) == 1);
  REQUIRE(WaitTiles(0, 128, 48) == 3);
  REQUIRE(WaitTiles(1, 128, 48) == 4);
  // A single task shorter than one producer tile still waits for exactly one.
  REQUIRE(WaitInflation(64, 128, 512) == 0);

  std::vector<TileAxes> const axes = Axes();

  ModelDescription const model = HeadModel(128);
  std::vector<AlignmentConstraint> const constraints =
      DeriveAlignmentConstraints(model);
  REQUIRE(constraints.size() == 1);
  // Nothing here names 128: it is read out of the stage that consumes the
  // GEMM's destination buffer.
  REQUIRE(constraints[0].consumer_stage == 2);
  REQUIRE(constraints[0].consumer_kind == StageKind::kRoPE);
  REQUIRE(constraints[0].consumer_tile == 128);
  REQUIRE(constraints[0].producer_stage == 0);
  REQUIRE(constraints[0].producer_tile == 512);

  AlignmentStats stats;
  std::vector<std::vector<int>> const kept =
      PropagateAlignment(model, axes, &stats);
  REQUIRE(stats.operators == 1);
  REQUIRE(stats.before == static_cast<int>(axes.size()));
  REQUIRE(stats.after_max < stats.before);
  REQUIRE(Kept(kept[0], axes, 64) && Kept(kept[0], axes, 128));
  REQUIRE(!Kept(kept[0], axes, 48) && !Kept(kept[0], axes, 80));

  // Move the head dimension and the surviving set has to move with it, which
  // is what "derived, not hard-coded" means operationally.
  // 48 and 80 straddle a 128-wide head but not a 96-wide one; 112 and 128
  // straddle a 96-wide head but not a 128-wide one.  Neither set is a subset
  // of the other, so nothing but the granularity could be producing them.
  ModelDescription const wide = HeadModel(96);
  std::vector<std::vector<int>> const wide_kept = PropagateAlignment(wide, axes);
  REQUIRE(Kept(wide_kept[0], axes, 48) && Kept(wide_kept[0], axes, 80));
  REQUIRE(!Kept(wide_kept[0], axes, 112) && !Kept(wide_kept[0], axes, 128));

  // A GEMM reading a GEMM couples two free variables; the single-variable form
  // must refuse rather than answer the wrong question.
  ModelDescription chained = HeadModel(128);
  chained.gemms.push_back({256, 512, 3, 8});
  ModelStage second;
  second.kind = StageKind::kGemm;
  second.gemm = 1;
  chained.stages.insert(chained.stages.begin() + 2, second);
  bool threw = false;
  try {
    DeriveAlignmentConstraints(chained);
  } catch (std::exception const&) {
    threw = true;
  }
  REQUIRE(threw);

  std::printf("alignment_propagation: ok\n");
  return 0;
}
