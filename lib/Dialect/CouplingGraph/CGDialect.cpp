// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/CGAttrs.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <llvm/ADT/TypeSwitch.h>

#include <cctype>
#include <stdexcept>

using namespace mlir;
using namespace tilemega::dialect;

#include "tilemega/Dialect/CouplingGraph/CGDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tilemega/Dialect/CouplingGraph/CGAttrs.cpp.inc"

namespace tilemega::dialect {

void CGDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tilemega/Dialect/CouplingGraph/CGAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tilemega/Dialect/CouplingGraph/CGOps.cpp.inc"
      >();
}

mlir::FailureOr<analysis::QuasiPolynomial> parseMetric(mlir::AsmParser& parser) {
  std::string expression;
  if (failed(parser.parseString(&expression))) return failure();
  try {
    return analysis::QuasiPolynomial::FromIslText(expression);
  } catch (std::exception const& error) {
    parser.emitError(parser.getCurrentLocation(), error.what());
    return failure();
  }
}

void printMetric(mlir::AsmPrinter& printer,
                 analysis::QuasiPolynomial const& value) {
  printer.printString(value.ToString());
}

mlir::FailureOr<analysis::CouplingRelation> parseCouplingRelation(
    mlir::AsmParser& parser) {
  std::string text;
  if (failed(parser.parseString(&text))) return failure();
  try {
    return analysis::CouplingRelation::FromIslText(text);
  } catch (std::exception const& error) {
    parser.emitError(parser.getCurrentLocation(), error.what());
    return failure();
  }
}

void printCouplingRelation(mlir::AsmPrinter& printer,
                           analysis::CouplingRelation const& value) {
  printer.printString(value.ToString());
}

static analysis::ParamBinding readBinding(ModuleOp module, StringRef name) {
  analysis::ParamBinding result;
  if (auto dict = module->getAttrOfType<DictionaryAttr>(name)) {
    for (NamedAttribute item : dict) {
      if (auto integer = dyn_cast<IntegerAttr>(item.getValue()))
        result.Bind(item.getName().str(), integer.getInt());
    }
  }
  return result;
}

/// The combined binding available at verification time: every theta value
/// the module happens to have already fixed, plus every g (granularity)
/// value. A workload dimension the module leaves free (e.g. the sequence
/// length) stays out of this binding and therefore stays a genuine isl
/// parameter on every metric this verifier checks (invariant I1).
static analysis::ParamBinding combinedBinding(ModuleOp module) {
  analysis::ParamBinding result = readBinding(module, "tilemega.theta");
  analysis::ParamBinding granularity = readBinding(module, "tilemega.g");
  for (auto const& [name, value] : granularity.values) result.Bind(name, value);
  return result;
}

LogicalResult TaskSpaceOp::verify() {
  static constexpr StringLiteral known[] = {
      "gemm", "rmsnorm", "rope", "kvappend", "elementwise", "attention",
      "view", "transpose", "broadcast", "reduction", "slice", "concat",
      // `generic` is the degraded classification: one conservative task space
      // for an operator no rule covers.
      "frontend", "generic"};
  StringRef kind = getKind().getValue().getValue();
  if (llvm::none_of(known, [&](StringRef value) { return value == kind; }))
    return emitOpError() << "unknown task kind '" << kind << "'";
  if (getStage() < 0) return emitOpError("stage must be non-negative");
  return success();
}

LogicalResult EventTensorOp::verify() {
  auto tensor = dyn_cast<RankedTensorType>(getEventType());
  if (!tensor || tensor.getRank() != 1 || !tensor.getElementType().isInteger(32))
    return emitOpError("event type must be rank-1 tensor<i32>");
  return success();
}

LogicalResult CouplingOp::verify() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module) return emitOpError("must be nested in a module");
  if (!SymbolTable::lookupNearestSymbolFrom<TaskSpaceOp>(*this, getSrcAttr()))
    return emitOpError() << "unknown source task " << getSrc();
  if (!SymbolTable::lookupNearestSymbolFrom<TaskSpaceOp>(*this, getDstAttr()))
    return emitOpError() << "unknown destination task " << getDst();
  auto event = SymbolTable::lookupNearestSymbolFrom<EventTensorOp>(*this, getEventAttr());
  if (!event) return emitOpError() << "unknown event tensor " << getEvent();
  // No separate structure check: isl_map syntax is its own schema, already
  // validated by isl when the attribute was parsed (parseCouplingRelation).
  StringRef sync = getSyncKind().getValue().getValue();
  if (sync != "global" && sync != "cluster" && sync != "local")
    return emitOpError() << "invalid sync kind '" << sync << "'";
  if (getTier().getValue() == 3 && sync == "cluster")
    return emitOpError("Tier 3 coupling cannot use cluster synchronization");
  if (getTier().getValue() < 0 || getTier().getValue() > 3)
    return emitOpError("tier must be in [0,3]");

  try {
    analysis::ParamBinding known = combinedBinding(module);
    // wait(x) = |C(x)|, computed directly from the relation -- not read back
    // from a second, separately authored copy the way the pre-migration
    // DictionaryAttr's "fiber" field was. SemanticallyEqual compares the two
    // quasi-polynomials as functions after substituting `known`, not as
    // scalars: a genuinely position-dependent wait must match at every task
    // coordinate, not merely at whichever point a scalar comparison would
    // have implicitly picked.
    analysis::QuasiPolynomial expectedWait = getRelation().getMap().Card();
    if (!expectedWait.SemanticallyEqual(getWait().getValue(), known))
      return emitOpError() << "wait " << getWait().getValue().ToString()
                           << " does not match the relation's fiber "
                              "cardinality " << expectedWait.ToString();
    // image(C_kappa) itself is not re-derived from the relation here: doing
    // so needs "does producer coordinate depend on consumer coordinate X"
    // per domain dimension, and the only isl query available for that
    // (isl_map_involves_dims) is syntactic -- it also flags a domain
    // coordinate that is merely *bounded* (as every one now is, so wait/
    // fanout stay finite) but does not actually influence the output,
    // which silently overcounts image(C_kappa) (confirmed empirically:
    // a plain `j >= 0 and j < N` domain bound makes involves_dims report
    // `j` as involved even when the map's output never depends on it).
    // ComputeEventShape (lib/Analysis/CouplingDerivation.cpp) gets this
    // right because it tracks "which coordinate a producer-axis constraint
    // actually referenced" during construction, not by re-inspecting the
    // assembled map -- context this verifier does not have. So the event
    // extent is checked the same way wait/fanout/volume/count all were
    // before this migration: it must evaluate under `known`, not that it
    // matches a value re-derived from C.
    (void)event.getExtent().getValue().Eval(known);
  } catch (std::exception const& error) {
    return emitOpError() << "cannot evaluate coupling metric after theta/g "
                            "binding: " << error.what();
  }
  return success();
}

LogicalResult PlacementOp::verify() {
  if (getCluster() < 1) return emitOpError("cluster must be positive");
  if (getMap().empty()) return emitOpError("placement map cannot be empty");
  if (!SymbolTable::lookupNearestSymbolFrom<TaskSpaceOp>(*this, getTaskAttr()))
    return emitOpError() << "unknown task space " << getTask();
  return success();
}

}  // namespace tilemega::dialect

#define GET_OP_CLASSES
#include "tilemega/Dialect/CouplingGraph/CGOps.cpp.inc"
