// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/CGAttrs.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/AsmParser/AsmParser.h>

#include <cassert>
#include <sstream>

using namespace tilemega;

namespace {
/// Print `attr` the way MLIR would inside a larger op, then reparse it
/// through MLIR's own generic attribute parser -- the actual round trip the
/// CG dialect verifier and IR text form depend on, not just the C++ getter.
template <typename AttrT>
AttrT RoundTripThroughMlirText(mlir::MLIRContext& context, AttrT attr) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  mlir::Attribute generic = attr;
  generic.print(stream);
  mlir::Attribute parsed = mlir::parseAttribute(text, &context);
  assert(parsed && "MLIR failed to reparse the printed attribute");
  auto typed = mlir::dyn_cast<AttrT>(parsed);
  assert(typed && "reparsed attribute has the wrong C++ type");
  return typed;
}
}  // namespace

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();

  // MetricAttr: a QuasiPolynomial (wait/fanout/volume/count's storage type)
  // round-trips through MLIR IR text with its value intact.
  auto original = analysis::QuasiPolynomial::FromIslText(
      "[S,heads] -> { (heads * ceild(S,3)) : S >= 0 }");
  auto metric = dialect::MetricAttr::get(&context, original);
  auto restoredMetric = RoundTripThroughMlirText(context, metric);
  analysis::ParamBinding known;
  known.Bind("S", 7).Bind("heads", 4);
  assert(original.Eval(known) == 12);  // heads * floor(7/3) = 4*2
  assert(restoredMetric.getValue().Eval(known) == original.Eval(known));
  assert(restoredMetric.getValue().ToString() == original.ToString());

  // CouplingMapAttr: a CouplingRelation (C's storage type) round-trips too,
  // and stays usable for the isl operations the verifier runs on it.
  auto relation = analysis::CouplingRelation::FromIslText(
      "[S] -> { [m] -> [p0 = m] : 0 <= m and 128*m < S }");
  auto couplingMap = dialect::CouplingMapAttr::get(&context, relation);
  auto restoredMap = RoundTripThroughMlirText(context, couplingMap);
  assert(restoredMap.getMap().ToString() == relation.ToString());
  assert(restoredMap.getMap().Card().Eval(known) == 1);

  return 0;
}
