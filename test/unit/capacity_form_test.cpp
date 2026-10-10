// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ClosedForm.h>
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <algorithm>

namespace tilemega::tests::capacity_form_test {
int TestCapacityForm(int, char**) {
  using namespace analysis;
  IslContext context;
  auto t = ClosedForm::Symbol("T"), e = ClosedForm::Constant(128), k = ClosedForm::Constant(8);
  for (long block : {16, 32, 64, 128}) {
    auto capacity = (t * k).CeilDiv(ClosedForm::Constant(block)) + e.Min(t * k);
    auto roundtrip = ClosedForm::Parse(capacity.ToString());
    assert(roundtrip.ToString() == capacity.ToString() && roundtrip.HasPiecewise());
    auto polynomial = QuasiPolynomial::FromClosedForm(capacity);
    auto bound = CouplingRelation::FromIslText("[T] -> { [] -> [v] : 0 <= v < " + capacity.ToIslText() + " }");
    for (long tokens : {1, 2, 4, 8, 16, 17, 31, 64, 256, 4096}) {
      ParamBinding known; known.Bind("T", tokens);
      auto expected = (tokens * 8 + block - 1) / block + std::min(128L, tokens * 8);
      assert(capacity.Eval(known, {}) == expected && polynomial.Eval(known) == expected);
      assert(QuasiPolynomial::FromClosedForm(capacity, known).Eval({}) == expected);
      assert(bound.BindParams(known).ImageCard().Eval({}) == expected);
    }
  }
  auto a = t.Min(ClosedForm::Constant(7));
  auto b = (t * ClosedForm::Constant(2)).Min(ClosedForm::Constant(11));
  auto expression = ((a + b) * ClosedForm::Constant(3)).FloorDiv(ClosedForm::Constant(2));
  auto polynomial = QuasiPolynomial::FromClosedForm(expression);
  for (long tokens : {-5, 0, 1, 5, 7, 9, 100}) {
    ParamBinding known; known.Bind("T", tokens);
    auto expected = (3 * (std::min(tokens, 7L) + std::min(2 * tokens, 11L)));
    auto floor = expected / 2 - (expected % 2 < 0);
    assert(expression.Eval(known, {}) == floor && polynomial.Eval(known) == floor);
  }
  assert(ClosedForm::Parse("min_value").ToString() == "min_value");
  assert(!ClosedForm::Parse("(T + 2)").HasPiecewise());
  return 0;
}
}
