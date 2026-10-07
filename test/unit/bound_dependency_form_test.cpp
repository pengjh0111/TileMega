// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/BoundDependencyForm.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <cstdio>

namespace tilemega::tests::bound_dependency_form_test {
namespace {
using namespace analysis;
ClosedForm F(long x) { return ClosedForm::Constant(x); }
OperatorNode Task(char const* name, ClosedForm extent) {
  OperatorNode node; node.name = name; node.output = {"ownership", {{"task", extent}}}; node.tile = {F(1)}; return node;
}
void Check(char const* text, ClosedForm p, ClosedForm c, bool table, ParamBinding const& known = {}) {
  auto producer = Task("producer", p), consumer = Task("consumer", c);
  CouplingEdge edge; edge.C = CouplingRelation::FromIslText(text);
  auto encoded = BindExactTaskDependency(edge, producer, consumer, known);
  if ((encoded.encoding == BoundDependencyForm::Encoding::kTable) != table)
    std::fprintf(stderr, "dependency form mismatch: %s\n", text);
  assert((encoded.encoding == BoundDependencyForm::Encoding::kTable) == table);
  auto exact = LinearizeTaskCoupling(edge.C, producer, consumer, known);
  assert(Contains(encoded.encoded_relation, exact) && Contains(exact, encoded.encoded_relation));
  assert(encoded.table.has_value() == table);
}
}
int TestBoundDependencyForm(int, char**) {
  IslContext context;
  Check("{ [c] -> [p] : 0 <= c < 9 and p=c }", F(9), F(9), false);
  Check("{ [c] -> [p] : 0 <= c < 9 and 0 <= p < 9 and c-1 <= p <= c+1 }", F(9), F(9), false);
  Check("{ [c] -> [p] : 0 <= c < 9 and 0 <= p < 4 and floord(c,3) <= p <= floord(c,3)+1 }", F(4), F(9), false);
  Check("{ [c] -> [p] : 0 <= c < 9 and 0 <= p < 7 }", F(7), F(9), false);
  Check("{ [c] -> [p] : 0 <= c < 10 and 0 <= p < 7 and floord(c,3)-1 <= p < floord(c,3)+3 }", F(7), F(10), false);
  Check("{ [c] -> [p] : 0 <= c < 10 and 0 <= p < 10 and c-20 <= p < c+5 }", F(10), F(10), false);
  Check("{ [c] -> [p] : 0 <= c < 9 and 0 <= p < 12 and (p=c or p=c+3) }", F(12), F(9), true);
  Check("{ [c] -> [p] : false }", F(9), F(9), true);
  ParamBinding known; known.Bind("B", 3);
  Check("[B] -> { [c] -> [p] : 0 <= c < 3*B and p=c }", F(3)*ClosedForm::Symbol("B"), F(3)*ClosedForm::Symbol("B"), false, known);
  return 0;
}
}
