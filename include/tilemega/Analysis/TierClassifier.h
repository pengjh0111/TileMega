// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2.4 coupling analyzability tier (Part 4).
#pragma once
#include <tilemega/Analysis/CouplingDerivation.h>
namespace tilemega::analysis {

/// What isl can and cannot say about a coupling's Tier.
///
/// It cannot say what the Tier *is*. Tier is provenance, not geometry:
///   * Tier 1 means a shared injective layout cancels -- the layout id is
///     not in C, so isl cannot see it.
///   * Tier 2 means the task space has a run-time extent -- but a Tier 0
///     edge's relation carries workload parameters (S) just the same, so
///     "has a parameter" does not separate them.
///   * Tier 3 means the index came from a tensor -- a relaxed C looks
///     structurally identical whatever caused the relaxation.
///
/// A tempting shortcut is worth naming because it is wrong: classifying by
/// isl_map_is_single_valued (one producer per consumer point => Tier 0).
/// Measured against §2.7 it misclassifies four of the twenty-one derived
/// edges, including three the table itself tabulates as Tier 0
/// (attn_combine->wo, add1->rmsnorm2, silu->wdown). Those edges are
/// one-to-many simply because `wait > 1`, which is ordinary for an exact
/// affine edge and says nothing about analyzability. Deriving a Tier from
/// the relation alone is therefore not implemented, rather than implemented
/// wrongly; CouplingDerivation assigns it from the access-map context that
/// actually carries the information.
///
/// What isl *can* do is check a claimed Tier's consequences, which is what
/// this class is for. A relaxed edge claims to have widened C to the
/// producer's whole task space; that claim is checkable, and if it were
/// false the widening would not be I2-safe.
class TierClassifier {
 public:
  /// True when every consumer point in `C`'s domain is coupled to every
  /// producer task in `producer`'s task space -- the widening a relaxed edge
  /// claims. False means the claim was not established, so the relaxation
  /// must not be treated as I2-safe.
  ///
  /// Note this is per consumer point, not `range(C) == producer space`: an
  /// exact identity edge covers the producer's whole task space as its range
  /// too (it is a bijection onto it), so range coverage does not separate
  /// relaxed from exact. Reaching all of it from a *single* consumer point
  /// does.
  bool RelaxationCoversProducer(CouplingRelation const& C,
                                OperatorNode const& producer,
                                ParamBinding const& known) const;
};

}  // namespace tilemega::analysis
