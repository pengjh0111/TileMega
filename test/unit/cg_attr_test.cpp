// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/CGAttrs.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/AsmParser/AsmParser.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>

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

  // CouplingAttributesAttr round-trips, and the coupling verifier recomputes
  // the tier from it rather than trusting the tier that was written down.
  auto attributes = dialect::CouplingAttributesAttr::get(
      &context, mlir::StringAttr::get(&context, "affine"),
      mlir::StringAttr::get(&context, "symbolic_static"),
      mlir::StringAttr::get(&context, "exact"),
      mlir::StringAttr::get(&context, "none"),
      mlir::StringAttr::get(&context, "piecewise_quasipoly"));
  auto restoredAttributes = RoundTripThroughMlirText(context, attributes);
  assert(restoredAttributes.getCountability().getValue() ==
         "piecewise_quasipoly");

  // The negative case: the same edge, relabelled Tier 2, must be rejected --
  // nothing about `affine + symbolic_static + exact + none` is ragged. A
  // misaligned tiling is exactly this edge (coupling_types_test), so the
  // check has to distinguish it from a genuinely runtime extent.
  auto module = [&](llvm::StringRef tier, llvm::StringRef extent) {
    std::string text =
        "module {\n"
        "  tilemega.task_space @a {fx_name = \"a\", granularity = {t = 1 : i64},"
        " kind = #tilemega.task_kind<\"elementwise\">,"
        " operator_name = \"aten.mul.Tensor\", stage = 0 : i64,"
        " write_map = #tilemega.access_map<{kind = \"elementwise\"}>}\n"
        "  tilemega.task_space @b {fx_name = \"b\", granularity = {t = 1 : i64},"
        " kind = #tilemega.task_kind<\"elementwise\">,"
        " operator_name = \"aten.mul.Tensor\", stage = 0 : i64,"
        " write_map = #tilemega.access_map<{kind = \"elementwise\"}>}\n"
        "  tilemega.event_tensor @e : tensor<1xi32> "
        "{extent = #tilemega.metric<\"{ 1 }\">}\n"
        "  tilemega.coupling @c from @a to @b {count = #tilemega.metric<\"{ 1 }\">,"
        " coupling_attrs = #tilemega.coupling_attrs<\"affine\", \"";
    text += extent.str();
    text +=
        "\", \"exact\", \"";
    text += extent == "runtime_dynamic" ? "prefix_sum" : "none";
    text +=
        "\", \"constant\">,"
        " event = @e, fanout = #tilemega.metric<\"{ 1 }\">,"
        " read_map = #tilemega.access_map<{kind = \"identity\"}>,"
        " relation = #tilemega.coupling_map<\"{ [0] -> [0] }\">,"
        " sync_kind = #tilemega.sync<\"global\">, tier = #tilemega.tier<";
    text += tier.str();
    text +=
        ">, volume = #tilemega.metric<\"{ 1 }\">,"
        " wait = #tilemega.metric<\"{ 1 }\">}\n}\n";
    return text;
  };
  assert(mlir::parseSourceString<mlir::ModuleOp>(
             module("0", "symbolic_static"), &context) &&
         "the consistent edge must verify");
  assert(!mlir::parseSourceString<mlir::ModuleOp>(
             module("2", "symbolic_static"), &context) &&
         "tier 2 does not follow from an exact static affine edge");
  // ... and the same tier is required once the extent really is runtime.
  assert(mlir::parseSourceString<mlir::ModuleOp>(
             module("2", "runtime_dynamic"), &context) &&
         "a runtime extent is Tier 2");
  assert(!mlir::parseSourceString<mlir::ModuleOp>(
             module("0", "runtime_dynamic"), &context) &&
         "a runtime extent cannot be relabelled Tier 0");

  return 0;
}
