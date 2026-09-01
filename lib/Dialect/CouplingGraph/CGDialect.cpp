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

mlir::FailureOr<analysis::ClosedForm> parseClosedForm(mlir::AsmParser& parser) {
  std::string expression;
  if (failed(parser.parseString(&expression))) return failure();
  try {
    return analysis::ClosedForm::Parse(expression);
  } catch (std::exception const& error) {
    parser.emitError(parser.getCurrentLocation(), error.what());
    return failure();
  }
}

void printClosedForm(mlir::AsmPrinter& printer,
                     analysis::ClosedForm const& value) {
  printer.printString(value.ToString());
}

LogicalResult CouplingMapAttr::verifyStructure(
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  DictionaryAttr fields = getFields();
  auto consumers = fields.getAs<ArrayAttr>("consumer");
  auto producers = fields.getAs<ArrayAttr>("producers");
  auto parameters = fields.getAs<ArrayAttr>("parameters");
  auto fiber = fields.getAs<ClosedFormAttr>("fiber");
  auto image = fields.getAs<ClosedFormAttr>("image");
  if (!consumers || !producers || !parameters || !fiber || !image) {
    return emitError() << "coupling_map requires consumer, producers, "
                          "parameters, fiber, and image fields";
  }
  for (Attribute coordinate : consumers) {
    if (!isa<StringAttr>(coordinate))
      return emitError() << "coupling_map consumer entries must be strings";
  }
  if (producers.empty())
    return emitError() << "coupling_map must contain at least one producer";
  for (Attribute item : producers) {
    auto producer = dyn_cast<DictionaryAttr>(item);
    if (!producer || !producer.getAs<StringAttr>("source") ||
        !producer.getAs<ArrayAttr>("coordinates") ||
        !producer.getAs<ArrayAttr>("ranges")) {
      return emitError() << "each producer requires source, coordinates, ranges";
    }
  }
  return success();
}

static analysis::AffineExpr parseAffineExpr(StringRef text) {
  std::string value = text.trim().str();
  bool identifier = !value.empty() &&
      (std::isalpha(static_cast<unsigned char>(value[0])) || value[0] == '_');
  for (char c : value) {
    identifier = identifier &&
        (std::isalnum(static_cast<unsigned char>(c)) || c == '_');
  }
  if (identifier) return analysis::AffineExpr::Variable(value);
  return analysis::AffineExpr::Constant(analysis::ClosedForm::Parse(value));
}

analysis::AffineRelation CouplingMapAttr::getRelation() const {
  std::vector<std::string> consumers;
  for (Attribute item : getFields().getAs<ArrayAttr>("consumer"))
    consumers.push_back(cast<StringAttr>(item).getValue().str());
  std::vector<std::string> parameters;
  for (Attribute item : getFields().getAs<ArrayAttr>("parameters"))
    parameters.push_back(cast<StringAttr>(item).getValue().str());
  std::vector<analysis::ProducerMap> producers;
  for (Attribute item : getFields().getAs<ArrayAttr>("producers")) {
    auto dict = cast<DictionaryAttr>(item);
    analysis::ProducerMap producer;
    producer.source.name = dict.getAs<StringAttr>("source").getValue().str();
    for (Attribute coordinate : dict.getAs<ArrayAttr>("coordinates"))
      producer.coordinates.push_back(
          parseAffineExpr(cast<StringAttr>(coordinate).getValue()));
    for (Attribute rangeItem : dict.getAs<ArrayAttr>("ranges")) {
      auto range = cast<DictionaryAttr>(rangeItem);
      producer.quantified.push_back({
          range.getAs<StringAttr>("name").getValue().str(),
          parseAffineExpr(range.getAs<StringAttr>("begin").getValue()),
          analysis::ClosedForm::Parse(
              range.getAs<StringAttr>("extent").getValue().str())});
    }
    producers.push_back(std::move(producer));
  }
  return analysis::AffineRelation(std::move(consumers), std::move(producers),
                                  std::move(parameters));
}

ClosedFormAttr CouplingMapAttr::getFiberCardinality() const {
  return getFields().getAs<ClosedFormAttr>("fiber");
}

ClosedFormAttr CouplingMapAttr::getImageCardinality() const {
  return getFields().getAs<ClosedFormAttr>("image");
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

LogicalResult TaskSpaceOp::verify() {
  static constexpr StringLiteral known[] = {
      "gemm", "rmsnorm", "rope", "kvappend", "elementwise", "attention",
      "view", "transpose", "broadcast", "reduction", "slice", "concat",
      "frontend"};
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
  if (failed(getRelation().verifyStructure([&] { return emitOpError(); })))
    return failure();
  StringRef sync = getSyncKind().getValue().getValue();
  if (sync != "global" && sync != "cluster" && sync != "local")
    return emitOpError() << "invalid sync kind '" << sync << "'";
  if (getTier().getValue() == 3 && sync == "cluster")
    return emitOpError("Tier 3 coupling cannot use cluster synchronization");
  if (getTier().getValue() < 0 || getTier().getValue() > 3)
    return emitOpError("tier must be in [0,3]");

  try {
    auto theta = readBinding(module, "tilemega.theta");
    auto granularity = readBinding(module, "tilemega.g");
    long expectedWait = getRelation().getFiberCardinality().getValue().Eval(theta, granularity);
    long actualWait = getWait().getValue().Eval(theta, granularity);
    if (expectedWait != actualWait)
      return emitOpError() << "wait evaluates to " << actualWait
                           << " but relation fiber cardinality is " << expectedWait;
    long image = getRelation().getImageCardinality().getValue().Eval(theta, granularity);
    long eventExtent = event.getExtent().getValue().Eval(theta, granularity);
    if (eventExtent != image)
      return emitOpError() << "event extent evaluates to " << eventExtent
                           << " but image(C_kappa) has " << image;
    auto type = cast<RankedTensorType>(event.getEventType());
    if (type.hasStaticShape() && type.getNumElements() != image)
      return emitOpError() << "event tensor has " << type.getNumElements()
                           << " elements but image(C_kappa) has " << image;
  } catch (std::exception const& error) {
    return emitOpError() << "cannot evaluate closed form after theta/g binding: "
                         << error.what();
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
