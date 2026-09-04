// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definition 3, §3.1 solving-authority migration.
//
// The isl-backed replacement for the deleted `AffineRelation` class. C is
// genuinely an isl_map now: `C = W^-1 o R` is
// `isl_map_apply_range(R, isl_map_reverse(W))`, containment (invariant I2)
// is `isl_map_is_subset`, and Coarsen (§2.3's C_kappa = floor(./kappa) o C)
// is isl_map composition with a floor map -- none of which `AffineRelation`
// could express (it had no image/preimage/composition operators at all,
// which is why Coarsen was unimplementable before this migration).
//
// Like QuasiPolynomial, this type stores only isl text: never a live isl
// object, so it stays a plain, comparable, hashable value usable as an MLIR
// attribute payload.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include <llvm/ADT/Hashing.h>

namespace tilemega::analysis {

class QuasiPolynomial;  // fwd, for Card()

/// C: consumer task coordinates -> producer task coordinates, in isl map
/// syntax, e.g. `[S,Tm] -> { [m,n] -> [i] : i = m and 0 <= m < ceild(S,Tm) }`.
class CouplingRelation {
 public:
  CouplingRelation() = default;  // empty: no producer maps this consumer

  /// `text` must be valid isl_map syntax; canonicalized by an isl parse/
  /// print round trip.
  static CouplingRelation FromIslText(std::string const& text);

  std::string const& ToString() const { return text_; }
  bool empty() const { return text_.empty(); }

  /// W^-1: reverses domain and range.
  CouplingRelation Reverse() const;
  /// `this` composed with `other`, matching isl_map_apply_range: the result
  /// maps `this`'s domain to `other`'s range through `this`'s range /
  /// `other`'s domain. `R.ApplyRange(W.Reverse())` is C = W^-1 o R.
  CouplingRelation ApplyRange(CouplingRelation const& other) const;
  /// Restrict the domain to `domain_set_text` (isl set syntax over the same
  /// domain tuple/parameters). DeriveCoupling uses this to bind every
  /// consumer coordinate to its own task-space extent before returning C
  /// (a coordinate this edge's own per-axis constraints never reference,
  /// e.g. a GEMM's "n" when only "m" ties back to its producer, would
  /// otherwise stay isl-unbounded -- ranging over every integer -- which
  /// makes Reverse().Card() (fanout) diverge instead of coming out as that
  /// coordinate's real extent).
  CouplingRelation IntersectDomain(std::string const& domain_set_text) const;
  /// The same restriction, applied to the range (producer-coordinate) tuple.
  /// DeriveCoupling uses this to bind every producer coordinate to its own
  /// task-space extent, for the same reason the domain is bound: without it,
  /// isl_map_card/isl_map_range's own piecewise decomposition of a
  /// quantified or boundary-derived output can retain "reachable in the
  /// unbounded case" pieces whose value happens to be 0 after a parameter is
  /// fixed, rather than genuinely excluding them -- confirmed empirically,
  /// see FanoutCard's comment.
  CouplingRelation IntersectRange(std::string const& range_set_text) const;
  /// The image of `this` under the elementwise map `floor(./kappa)` applied
  /// to every range (producer-coordinate) dimension -- §2.3's C_kappa.
  /// `kappa` values are per range dimension, in order; a value of 1 leaves
  /// that dimension unchanged.
  CouplingRelation Coarsen(std::vector<long> const& kappa) const;
  /// Invariant I2's substitutability check: does `this` (the narrow,
  /// exact relation) hold as a subset of `wide` (the relaxed one)? True
  /// means `wide ⊇ this` is machine-established; false means "not
  /// established", never "disproved" (isl_map_is_subset is exact for the
  /// Presburger relations this migration builds, so in practice false here
  /// does mean the containment does not hold -- the "not established"
  /// caveat is about relations outside what this codebase constructs, e.g.
  /// ones a future caller builds from unvalidated text).
  bool IsSubset(CouplingRelation const& wide) const;
  /// True when every consumer coordinate maps to exactly one producer
  /// coordinate (isl_map_is_single_valued). Note what this is *not*: it is
  /// not a Tier signal. A one-to-many C simply means `wait > 1`, which is
  /// ordinary for an exact affine edge -- §2.7's rows 7, 9 and 12 are all
  /// Tier 0 and all one-to-many. See TierClassifier.h.
  bool IsSingleValued() const;

  /// wait(x) = |this(x)|, a function of `this`'s domain (consumer)
  /// coordinates.
  QuasiPolynomial Card() const;
  /// fanout(y) = |this^-1(y)|, a function of `this`'s range (producer)
  /// coordinates, restricted to y actually in range(this). Not simply
  /// `Reverse().Card()`: the domain tuple is bound in DeriveCoupling
  /// (DomainBoxText), but the *range* (producer-coordinate) tuple never is,
  /// so isl_map_card's piecewise result for the reversed map covers a wider
  /// declared domain than range(this) actually spans -- with the formula
  /// correctly evaluating to 0 on the extra points, but "correctly 0" still
  /// breaks Eval's "is this the same value everywhere" check, since 0
  /// becomes a spurious extra value alongside the real, constant fanout
  /// (confirmed empirically: attn_combine->wo's fanout piece is a uniform
  /// 32 within the true range, plus a formula-evaluates-to-0 tail outside
  /// it, up to p0 = 126 + S). Restricting the reversed map's domain to
  /// range(this) first removes those points from consideration entirely,
  /// rather than merely zeroing them.
  QuasiPolynomial FanoutCard() const;

  /// True when *every* domain point of `this` is coupled to *every* point
  /// of `set_text` -- i.e. `domain(this) x set_text` is contained in `this`.
  ///
  /// This, not `range(this) == set_text`, is what a relaxation claims. An
  /// exact identity edge also has the producer's whole task space as its
  /// range (it is a bijection onto it); what makes a relaxed edge different
  /// is that a *single* consumer point reaches all of it.
  bool CouplesEveryDomainPointTo(std::string const& set_text) const;

  /// Every integer point pair of `this`, as (consumer, producer) coordinate
  /// vectors. Requires a relation with no free parameters and a bounded
  /// domain and range -- i.e. a fully instantiated model. Used to answer
  /// questions isl has no closed-form operator for, such as whether the wait
  /// set is contiguous under a given linearization of the two task spaces.
  std::vector<std::pair<std::vector<long>, std::vector<long>>> Points() const;

  /// Task-coordinate names on the domain (consumer) side, in order.
  std::vector<std::string> DomainDimNames() const;
  /// Task-coordinate names on the range (producer) side, in order.
  std::vector<std::string> RangeDimNames() const;

  friend bool operator==(CouplingRelation const& lhs,
                         CouplingRelation const& rhs) {
    return lhs.text_ == rhs.text_;
  }
  friend bool operator!=(CouplingRelation const& lhs,
                         CouplingRelation const& rhs) {
    return !(lhs == rhs);
  }
  friend llvm::hash_code hash_value(CouplingRelation const& value);

 private:
  explicit CouplingRelation(std::string text) : text_(std::move(text)) {}
  std::string text_;
};

}  // namespace tilemega::analysis
