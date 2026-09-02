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

/// §2.4's Tier is a summary of five facts that vary independently, and only
/// the summary was ever recorded. That loses distinctions the scheduler
/// needs: a runtime extent and an over-approximated projection both print as
/// Tier 2, yet the first needs one prefix sum at launch and the second needs
/// nothing at all. Each attribute below names one observation the derivation
/// already makes; `DeriveTier` collapses them, so §2.7 keeps its tiers.

/// How the index mapping is expressed. Layout-mediated is §2.4's Tier 1
/// (the shared injective layout cancels, leaving a closed form in logical
/// space); data-dependent is Tier 3 (an index read out of a tensor).
enum class RelationKind { kAffine, kLayoutMediated, kDataDependent };

/// What fixes the loop bounds. `kSymbolicStatic` is invariant I1's normal
/// case -- an extent that is a theta symbol, whether or not the derivation's
/// `known` binding happens to give it a value. `kRuntimeDynamic` is a task
/// space whose own extent is only known once theta is instantiated.
enum class ExtentKind { kStaticLiteral, kSymbolicStatic, kRuntimeDynamic };

/// Whether the projection was carried exactly or widened under I2.
enum class Exactness { kExact, kRelaxed };

/// What the runtime must supply before the edge can be armed.
enum class RuntimeRequirement { kNone, kPrefixSum, kTensorValues };

/// The form the derived counts take. Orthogonal to the rest: a plainly
/// affine, exact, statically shaped edge still has a piecewise wait when the
/// two tilings are misaligned.
enum class Countability {
  kConstant,
  kQuasiPolynomial,
  kPiecewiseQuasiPolynomial,
  kUncountable,
};

std::string ToString(RelationKind kind);
std::string ToString(ExtentKind kind);
std::string ToString(Exactness exactness);
std::string ToString(RuntimeRequirement requirement);
std::string ToString(Countability countability);

struct CouplingAttributes {
  RelationKind relation_kind = RelationKind::kAffine;
  ExtentKind extent_kind = ExtentKind::kStaticLiteral;
  Exactness exactness = Exactness::kExact;
  RuntimeRequirement runtime_requirement = RuntimeRequirement::kNone;
  Countability countability = Countability::kConstant;

  /// `affine + symbolic_static + exact + none + piecewise_quasipoly`.
  std::string ToString() const;
};

/// The §2.4 tier as a function of the five attributes, and nothing else.
Tier DeriveTier(CouplingAttributes const& attributes);

/// Inverse of the ToString overloads, for the CG verifier: every field must
/// name a value this header defines, so IR cannot carry a classification the
/// derivation is unable to produce. Returns false and leaves `out` untouched
/// on the first unrecognized name.
bool ParseCouplingAttributes(std::string const& relation_kind,
                             std::string const& extent_kind,
                             std::string const& exactness,
                             std::string const& runtime_requirement,
                             std::string const& countability,
                             CouplingAttributes* out);

/// One derived coupling edge.  `event_shape` is the shape §2.3 asks for:
/// EventTensor(e) = image(C_kappa), which for kappa = 1 is the quotient of the
/// consumer task space by the fibers of C -- the product of the ranges of the
/// consumer coordinates that actually occur in C.
struct CouplingEdge {
  TaskSpaceId src;
  TaskSpaceId dst;
  CouplingRelation C;
  DerivedMetrics metrics;
  CouplingAttributes attributes;
  /// Always `DeriveTier(attributes)`; kept as a field because §2.7, the CG
  /// attributes and the solver all still speak in tiers.
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
