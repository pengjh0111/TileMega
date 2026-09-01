// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/CGAttrs.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>

#include <mlir/IR/MLIRContext.h>

#include <cassert>

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<tilemega::dialect::CGDialect>();
  auto original = tilemega::analysis::ClosedForm::Symbol("S")
      .CeilDiv(tilemega::analysis::ClosedForm::Symbol("Tm")) *
      tilemega::analysis::ClosedForm::Symbol("heads");
  auto attr = tilemega::dialect::ClosedFormAttr::get(&context, original);
  auto restored = attr.getValue();
  tilemega::analysis::ParamBinding theta, g;
  theta.Bind("S", 7).Bind("heads", 4);
  g.Bind("Tm", 3);
  assert(original.Eval(theta, g) == 12);
  assert(restored.Eval(theta, g) == original.Eval(theta, g));
  assert(restored.ToString() == original.ToString());
  auto floor = tilemega::analysis::ClosedForm::Parse("s0//128");
  theta.Bind("s0", 129);
  assert(floor.Eval(theta, g) == 1);
  return 0;
}
