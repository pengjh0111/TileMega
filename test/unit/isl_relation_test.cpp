// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
// Unit tests for the isl-backed CouplingRelation/QuasiPolynomial primitives
// that replaced AffineRelation/ClosedForm as the solving authority (Part 3).
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/QuasiPolynomial.h>

#include <cassert>
#include <iostream>
#include <set>
#include <sstream>

using namespace tilemega::analysis;

int main() {
  tilemega::analysis::IslContext isl_context;
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

  // Bind/project parameters on an already-derived relation, then retain only
  // one endpoint per domain point.  Window synthesis uses this path instead
  // of re-deriving and enumerating the entire fan-in at three sequence sizes.
  CouplingRelation fixed = C.BindParams(known);
  assert(fixed.LexMin().Points().size() == 4);
  assert(fixed.LexMax().Points().size() == 4);

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
  // Parameter substitution deliberately does not bind task coordinates.
  ParamBinding task_point;
  task_point.Bind("i",5);
  assert(triangular_wait.BindCoordinates(task_point).Eval({})==5);
  int refs=isl_context.ReferenceCount();
  threw=false;
  try { (void)triangular_wait.BindCoordinates({}); }
  catch (std::invalid_argument const&) { threw=true; }
  assert(threw && isl_context.ReferenceCount()==refs);
  assert(triangular_wait.SumDomain().Eval({}) == 28);
  assert(triangular.FanoutCard().SumDomain().Eval({}) == 28);
  assert(triangular.ImageCard().Eval({}) == 7);
  assert(triangular.Image().ImageCard().Eval({}) == 7);
  assert(triangular.Coarsen({2}).ImageCard().Eval({}) == 4);
  assert(triangular.AggregateImage().ImageCard().Eval({}) == 1);
  auto symbolic_sum = QuasiPolynomial::FromIslText(
      "[S] -> { [i] -> i : 0 <= i < S }").SumDomain();
  ParamBinding eight;
  eight.Bind("S", 8);
  assert(symbolic_sum.SubstituteParams(eight).Eval(eight) == 28);
  auto disjoint_sum = QuasiPolynomial::FromIslText("[S] -> { S : 0 <= S <= 4 }")
      .Add(QuasiPolynomial::FromIslText("[S] -> { 2*S : S >= 4 }"));
  for (int s : {0,3,4,8}) {
    ParamBinding theta; theta.Bind("S",s);
    assert(disjoint_sum.Eval(theta) == (s <= 4 ? s : 0) + (s >= 4 ? 2*s : 0));
  }
  auto lexical = QuasiPolynomial::FromIslText(
      "[S,SS] -> { [x] -> 31*S + SS + floor((S+3)/4) : 0 <= x < SS and S >= -2 }");
  ParamBinding partial; partial.Bind("S",-2);
  auto specialized = lexical.SubstituteParams(partial);
  partial.Bind("SS",3);
  assert(lexical.Eval(partial) == -59);
  assert(specialized.Eval(partial) == -59);
  assert(QuasiPolynomial::Sum({disjoint_sum,disjoint_sum.Scale(-1)}).Eval(eight) == 0);
  auto coordinate_sum = QuasiPolynomial::Sum({triangular_wait, triangular_wait});
  assert(coordinate_sum.SumDomain().Eval({}) == 56);
  assert(QuasiPolynomial::Sum({}).Eval({}) == 0);
  auto periodic = QuasiPolynomial::FromIslText(
      "[S] -> { floor((S+3)/4) + floor(S/7) : 0 <= S <= 63 }");
  for (int limit : {1,4,16,64}) {
    auto split_periods = periodic.SplitPeriods(limit);
    bool equivalent = periodic.SemanticallyEqual(split_periods, {});
    std::cerr << "PERIOD_CHECK limit=" << limit << " symbolic_equal=" << equivalent
              << " before=" << periodic.ToString() << " after=" << split_periods.ToString() << '\n';
    for (int s=0;s<=63;++s) {
      ParamBinding theta; theta.Bind("S",s);
      assert(split_periods.Eval(theta) == periodic.Eval(theta));
    }
    auto difference = periodic.Add(split_periods.Scale(-1));
    // is_zero is structural: it misses floor expressions confined to S=0.
    // Equal global min/max of the difference proves the whole domain, not
    // merely the finite point checks above; no numerical tolerance is used.
    std::cerr << "PERIOD_DIFFERENCE limit=" << limit << " expression="
              << difference.ToString() << " range_value=" << difference.Eval({}) << '\n';
    assert(difference.Eval({}) == 0);
  }
  int references = isl_context.ReferenceCount();
  bool rejected_period_limit = false;
  try { (void)periodic.SplitPeriods(0); }
  catch (std::invalid_argument const&) { rejected_period_limit = true; }
  assert(rejected_period_limit && isl_context.ReferenceCount() == references);
  std::cout << "PERIOD_ERROR rejected=" << rejected_period_limit
            << " before=" << references << " after=" << isl_context.ReferenceCount() << '\n';
  assert(triangular.Union(triangular).Card().SumDomain().Eval({}) == 28);
  assert(CouplingRelation().Union(triangular) == triangular);
  assert(triangular.Union(CouplingRelation()) == triangular);
  auto overlap = CouplingRelation::FromIslText(
      "{ [i] -> [j] : 0 <= i < 5 and j=i; [i] -> [j] : 2 <= i < 8 and j=i }");
  auto overlapping_points = overlap.Points();
  assert(overlapping_points.size() == 8);
  for (int i = 0; i < 8; ++i) {
    assert(overlapping_points[i].first == std::vector<long>{i});
    assert(overlapping_points[i].second == std::vector<long>{i});
  }
  for (int seed = 0; seed < 32; ++seed) {
    std::ostringstream text;
    std::set<std::pair<std::vector<long>, std::vector<long>>> expected;
    text << "{ ";
    for (int part = 0; part < 8; ++part) {
      int il = (seed+part)%5, ih = 5+(2*seed+part)%5;
      int jl = (3*seed+part)%5, jh = 5+(seed+3*part)%5;
      int offset = part%3;
      text << (part ? "; " : "") << "[i] -> [j] : " << il << " <= i < " << ih
           << " and " << jl << " <= j < " << jh << " and j <= i+" << offset;
      for (long i = il; i < ih; ++i)
        for (long j = jl; j < jh; ++j)
          if (j <= i+offset) expected.insert({{i},{j}});
    }
    text << " }";
    auto points = CouplingRelation::FromIslText(text.str()).Points();
    assert(points.size() == expected.size());
    assert((decltype(expected)(points.begin(),points.end()) == expected));
  }

  // SemanticallyEqual: same function after substitution, spelled differently.
  QuasiPolynomial a = QuasiPolynomial::FromIslText("[S] -> { S : S > 0 }");
  QuasiPolynomial b = QuasiPolynomial::FromIslText("[S] -> { 2*S - S : S > 0 }");
  assert(a.SemanticallyEqual(b, {}));
  QuasiPolynomial c = QuasiPolynomial::FromIslText("[S] -> { S + 1 : S > 0 }");
  assert(!a.SemanticallyEqual(c, {}));

  return 0;
}
