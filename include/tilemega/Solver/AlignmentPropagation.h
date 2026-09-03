// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §P4.3 tier-2 alignment propagation.
//
// Tier 1 prunes a GEMM's tile shapes against the hardware alone and leaves 224
// of them (§P4.2).  Tier 2 prunes against the *neighbours*: a producer tile
// that straddles the boundary of the task that reads it forces that reader to
// wait on one more producer tile than its shape needs, and the extra wait is
// pure serialisation behind a grid barrier.
//
// The reader's granularity is never assumed here.  It is read out of the
// generated stage table -- RoPE and KVAppend carry the head dimension in
// `width`, the elementwise tail its row length in `extent` -- so the constraint
// that QKV columns align to `d` is derived from the model that was compiled,
// not written down.
#pragma once

#include <tilemega/Solver/ModelDescription.h>

#include <string>
#include <vector>

namespace tilemega::solver {

/// How many producer tiles of width `producer_tile` the `m`-th consumer task of
/// width `consumer_tile` has to wait for: ceil((m+1)Tm/Tr) - floor(m Tm/Tr).
int WaitTiles(int m, int consumer_tile, int producer_tile);

/// max_m WaitTiles(...) - ceil(consumer_tile / producer_tile) over the tasks
/// that exist in `extent` elements.  Zero means every consumer task waits on
/// the fewest producer tiles its own width allows; anything larger is the
/// inflation §P4.3 names.
int WaitInflation(int consumer_tile, int producer_tile, int extent);

/// Who reads one GEMM's output and who wrote its input, with the granularity
/// each of them works at.  The N axis is constrained by the consumer, the K
/// axis by the producer, and both are read off the generated table.
struct AlignmentConstraint {
  int gemm = -1;
  int consumer_stage = -1;    ///< -1 when nothing in the model reads the output
  int consumer_tile = 0;      ///< the reader's `Tr` along the GEMM's N
  StageKind consumer_kind = StageKind::kGemm;
  int producer_stage = -1;    ///< -1 when the input is a model entry buffer
  int producer_tile = 0;      ///< the writer's granularity along the GEMM's K
  StageKind producer_kind = StageKind::kGemm;
};

/// One entry per GEMM, in model order.  Derived by walking the stage list
/// backwards from the output end, which is the direction the constraint
/// travels: a GEMM's admissible column tiles depend on its consumer's, and a
/// consumer that is itself a GEMM passes its own tile on.
std::vector<AlignmentConstraint> DeriveAlignmentConstraints(
    ModelDescription const& model);

struct AlignmentStats {
  int operators = 0;
  int before = 0;              ///< candidates per operator entering tier 2
  int after_min = 0;
  int after_max = 0;
  double after_mean = 0.0;
  double joint_log10_before = 0.0;  ///< log10 of the product over operators
  double joint_log10_after = 0.0;
  int unconstrained = 0;       ///< operators whose output nothing reads
};

/// The (tile_n, tile_k) of each tier-1 candidate, in the caller's order.
struct TileAxes {
  int tile_n = 0;
  int tile_k = 0;
};

/// Keeps, for each GEMM, the candidates whose column tile costs its consumer
/// no extra wait and whose depth tile costs it no extra wait on its producer.
/// Returns one index list per GEMM into `axes`.
std::vector<std::vector<int>> PropagateAlignment(ModelDescription const& model,
                                                 std::vector<TileAxes> const& axes,
                                                 AlignmentStats* stats = nullptr);

/// A human-readable derivation trace, one line per GEMM, for the experiment
/// log: which stage constrains it, at what granularity, and what survived.
std::string ExplainAlignment(ModelDescription const& model,
                             std::vector<TileAxes> const& axes,
                             std::vector<std::vector<int>> const& kept);

}  // namespace tilemega::solver
