// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 1-5 and §2.4 tiers.
#pragma once

#include <string>
#include <vector>

#include <tilemega/Analysis/AccessRelation.h>
#include <tilemega/Analysis/ClosedForm.h>
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <tilemega/Analysis/DerivedMetrics.h>
#include <tilemega/Analysis/TensorSpace.h>

namespace tilemega::analysis {

enum class Tier {
  kAffine = 0,
  kSharedInjectiveLayout = 1,
  kStructuredRagged = 2,
  kDataDependent = 3,
};

enum class SyncKind { kGlobal, kCluster, kLocal };

std::string ToString(Tier tier);

/// One derived coupling edge.  `event_shape` is the shape §2.3 asks for:
/// EventTensor(e) = image(C_kappa), which for kappa = 1 is the quotient of the
/// consumer task space by the fibers of C -- the product of the ranges of the
/// consumer coordinates that actually occur in C.
struct CouplingEdge {
  TaskSpaceId src;
  TaskSpaceId dst;
  CouplingRelation C;
  DerivedMetrics metrics;
  Tier tier = Tier::kAffine;
  SyncKind sync = SyncKind::kGlobal;
  std::vector<ClosedForm> event_shape;
  /// False when a projection had to be over-approximated.  An inexact edge is
  /// still correct by I2 (the relaxed C contains the exact one) but its tier is
  /// raised so nothing downstream mistakes it for a closed form.
  bool exact = true;
  /// Non-empty when the edge is only live on part of the consumer task space:
  /// the producer writes a strict sub-window of the tensor (an append), so the
  /// coupling exists exactly where the consumer's read meets that window.
  std::string guard;
  /// Why the projection had to be widened, for the P3 report.  Empty when the
  /// derivation was exact.
  std::string relaxation;

  std::string EventShapeString() const;
};

/// C_{p->c} = W_p^-1 o R_c (§2 Definition 3), built directly as an isl_map:
/// every producer's W is a tiling, so W^-1 is exactly floor(./g_p) per tiled
/// axis, and the projection of a read interval through it has three outcomes
/// (exact quotient, exact floordiv for a single element, or an explicit
/// relaxation) -- the same case analysis as before this migration, now
/// emitting isl map constraint text per producer axis instead of building an
/// AffineExpr/ProducerMap tree.  `known` supplies every symbol ("theta" or
/// "g") whose value is already fixed at derivation time; isl can only take a
/// literal floor/ceildiv divisor (confirmed empirically, see
/// docs/experiments/P3_ISL/result.md), so any coefficient/group/tile/origin
/// that ends up as a divisor must be in `known` or ToIslText throws. A
/// symbol left out of `known` survives as a genuine isl parameter of the
/// resulting relation (invariant I1: workload dimensions stay symbolic).
///
/// Case (c) (relax) never invents an inverse; it widens C to the axis' full
/// range, which I2 permits.
struct CouplingDetail {
  bool exact = true;
  std::string guard;
  std::string relaxation;
  /// Consumer task-coordinate names a producer-axis constraint actually
  /// referenced while building C (exact-divide and single-element cases
  /// only; a relaxed axis references none). This, not a query on the
  /// assembled isl_map, is what ComputeEventShape needs: every consumer
  /// coordinate ends up *bounded* in C (so wait/fanout stay finite, see
  /// CouplingRelation::IntersectDomain), and isl's only "does this
  /// dimension matter" query is syntactic -- it cannot tell a coordinate
  /// that merely bounds the domain from one whose value actually changes
  /// the result. Tracking this at construction time, where the answer is
  /// unambiguous, sidesteps that isl limitation entirely.
  std::vector<std::string> occurring;
};

CouplingRelation DeriveCoupling(AccessRelation const& W, AccessRelation const& R,
                                OperatorNode const& producer,
                                OperatorNode const& consumer,
                                ParamBinding const& known,
                                CouplingDetail* detail = nullptr);

/// wait / fanout / volume / count, all QuasiPolynomial(theta) (§2 Definition 4).
DerivedMetrics ComputeMetrics(CouplingRelation const& C, AccessRelation const& W,
                              AccessRelation const& R,
                              OperatorNode const& producer,
                              OperatorNode const& consumer,
                              ParamBinding const& known);

/// image(C_kappa) for kappa = 1: the extents of the consumer coordinates in
/// `occurring` (see CouplingDetail::occurring -- tracked during
/// DeriveCoupling, not re-derived from C, for reasons documented there).
std::vector<ClosedForm> ComputeEventShape(OperatorNode const& consumer,
                                          std::vector<std::string> const& occurring);

/// The producer's own task-space box as an isl set, named after `C`'s range
/// dimensions: `{ [p0,...] : 0 <= p_i < producer_extent_i }`. Exposed
/// because both the fanout count and the relaxation check need to talk
/// about "every producer task there is".
std::string ProducerTaskSpaceText(CouplingRelation const& C,
                                  OperatorNode const& producer,
                                  ParamBinding const& known);

/// I2, machine-checked via isl_map_is_subset: does `wide` contain `narrow` as
/// a set of (consumer, producer) coordinate pairs?  `true` means containment
/// was *established*; `false` means "not established", never "disproved".
bool Contains(CouplingRelation const& wide, CouplingRelation const& narrow);

class CouplingDerivation {
 public:
  /// Derive every edge of `graph`: one edge per (operator, operand) pair whose
  /// operand names a producer inside the graph. `known` is the combined
  /// theta/g binding available at derivation time (see DeriveCoupling); any
  /// symbol not in it stays a genuine isl parameter on every derived edge.
  std::vector<CouplingEdge> Derive(OperatorGraph const& graph,
                                   ParamBinding const& known) const;
};

}  // namespace tilemega::analysis
