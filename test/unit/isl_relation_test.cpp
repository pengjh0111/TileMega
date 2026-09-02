// SPDX-License-Identifier: BSD-3-Clause
// Unit tests for the isl-backed CouplingRelation/QuasiPolynomial primitives
// that replaced AffineRelation/ClosedForm as the solving authority (Part 3).
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/QuasiPolynomial.h>

#include <cassert>

using namespace tilemega::analysis;

int main() {
  // C = W^-1 o R, matching §2.7 edge 1's shape: a producer tiled by Tm rows
  // (W), a consumer that reads the whole matching row tile (R, an identity
  // on the row coordinate). Composing should recover the identity on m.
  // Tm is a literal (128): a tile size is a "g" quantity, known before any
  // isl object is built (see docs/experiments/P3_ISL/result.md -- isl_aff_div
  // rejects a parametric divisor/coefficient, so Tm cannot itself be an isl
  // parameter here). S stays a genuine isl parameter (a "theta" quantity,
  // symbolic through to the generated binary per invariant I1).
  CouplingRelation W = CouplingRelation::FromIslText(
      "[S] -> { [m] -> [row] : 128*m <= row < 128*m + 128 and 0 <= m < "
      "ceild(S,128) }");
  CouplingRelation R = CouplingRelation::FromIslText(
      "[S] -> { [m] -> [row] : 128*m <= row < 128*m + 128 and 0 <= m < "
      "ceild(S,128) }");
  CouplingRelation C = R.ApplyRange(W.Reverse());
  assert(!C.empty());
  QuasiPolynomial wait = C.Card();
  ParamBinding known;
  known.Bind("S", 512);
  assert(wait.Eval(known) == 1);  // one producer tile per consumer tile

  // fanout(y) = |C^-1(y)|: reverse and card, restricted to y in range(C).
  QuasiPolynomial fanout = C.FanoutCard();
  assert(fanout.Eval(known) == 1);

  // I2 containment: a relaxed relation (whole producer axis) must contain
  // the exact one.
  CouplingRelation relaxed = CouplingRelation::FromIslText(
      "[S] -> { [m] -> [row] : 0 <= row < 128*ceild(S,128) and 0 <= m < "
      "ceild(S,128) }");
  assert(C.IsSubset(relaxed));
  assert(!relaxed.IsSubset(C));  // not established the other way

  // Coarsen: C_kappa = floor(./kappa) o C. Coarsening the row-tiled identity
  // by kappa=Tm collapses back to the m coordinate itself (image shrinks).
  CouplingRelation coarse = C.Coarsen({128});
  QuasiPolynomial coarse_card = coarse.Card();
  assert(coarse_card.Eval(known) == 1);

  // A genuinely position-dependent wait: a triangular map. wait(x) should
  // NOT reduce to one scalar (Eval must throw), and its printed form should
  // still be exact -- the quasi-polynomial-necessity case §3.5(d) asks for.
  CouplingRelation triangular =
      CouplingRelation::FromIslText("{ [i] -> [j] : 0 <= i < 8 and 0 <= j < i }");
  QuasiPolynomial triangular_wait = triangular.Card();
  bool threw = false;
  try {
    (void)triangular_wait.Eval({});
  } catch (std::out_of_range const&) {
    threw = true;
  }
  assert(threw);

  // SemanticallyEqual: same function after substitution, spelled differently.
  QuasiPolynomial a = QuasiPolynomial::FromIslText("[S] -> { S : S > 0 }");
  QuasiPolynomial b = QuasiPolynomial::FromIslText("[S] -> { 2*S - S : S > 0 }");
  assert(a.SemanticallyEqual(b, {}));
  QuasiPolynomial c = QuasiPolynomial::FromIslText("[S] -> { S + 1 : S > 0 }");
  assert(!a.SemanticallyEqual(c, {}));

  return 0;
}
