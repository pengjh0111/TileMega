// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 3-5 and §2.4 tiers.
#pragma once

#include <string>
#include <vector>

#include <tilemega/Analysis/AccessRelation.h>
#include <tilemega/Analysis/AffineRelation.h>
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
  AffineRelation C;
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

/// C_{p->c} = W_p^-1 o R_c (§2 Definition 3).
///
/// Every producer's W is a tiling, so W^-1 is exactly floor(./g_p) per tiled
/// axis; the derivation therefore stays inside closed-form algebra and does not
/// need a Presburger solver for the exact cases.  Three outcomes per axis:
///   (a) the read base divides the tile exactly    -> exact plain coordinate or
///                                                    a quantified range;
///   (b) the read span is a single element         -> exact floordiv(base,tile);
///   (c) otherwise                                 -> relax to the whole axis,
///                                                    mark the edge inexact.
/// Case (c) never invents an inverse; it widens C, which I2 permits.
struct CouplingDetail {
  bool exact = true;
  std::string guard;
  std::string relaxation;
};

AffineRelation DeriveCoupling(AccessRelation const& W, AccessRelation const& R,
                              OperatorNode const& producer,
                              OperatorNode const& consumer,
                              CouplingDetail* detail = nullptr);

/// wait / fanout / volume / count, all ClosedForm(theta, g) (§2 Definition 4).
DerivedMetrics ComputeMetrics(AffineRelation const& C, AccessRelation const& W,
                              AccessRelation const& R,
                              OperatorNode const& producer,
                              OperatorNode const& consumer);

/// image(C_kappa) for kappa = 1.
std::vector<ClosedForm> ComputeEventShape(AffineRelation const& C,
                                          OperatorNode const& consumer);

class CouplingDerivation {
 public:
  /// Derive every edge of `graph`: one edge per (operator, operand) pair whose
  /// operand names a producer inside the graph.
  std::vector<CouplingEdge> Derive(OperatorGraph const& graph) const;
};

}  // namespace tilemega::analysis
