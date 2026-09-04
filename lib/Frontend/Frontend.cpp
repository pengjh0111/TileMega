// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/SymbolicShapeBridge.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/DependencyForm.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Dialect/CouplingGraph/CGAttrs.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>

#include <algorithm>
#include <regex>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace tilemega::frontend {
namespace {

std::pair<long, long> parseRange(std::string const& text) {
  static std::regex const pattern(R"(VR\[(-?[0-9]+),\s*(-?[0-9]+)\])");
  std::smatch match;
  if (!std::regex_match(text, match, pattern))
    throw std::invalid_argument("unsupported ShapeEnv range: " + text);
  return {std::stol(match[1].str()), std::stol(match[2].str())};
}

std::vector<std::string> extractGuardTerms(
    std::string const& side, std::vector<std::vector<std::string>> const& shapes) {
  static std::regex const reference(
      R"(L\['flat_args'\]\[([0-9]+)\]\.size\(\)\[([0-9]+)\])");
  std::vector<std::string> terms;
  for (auto it = std::sregex_iterator(side.begin(), side.end(), reference);
       it != std::sregex_iterator(); ++it) {
    std::size_t input = std::stoul((*it)[1].str());
    std::size_t dimension = std::stoul((*it)[2].str());
    if (input >= shapes.size() || dimension >= shapes[input].size())
      throw std::invalid_argument("guard references an unknown input dimension");
    terms.push_back(shapes[input][dimension]);
  }
  std::sort(terms.begin(), terms.end());
  static std::regex const scalar(R"(^\s*\(?\s*(-?[0-9]+)\s*\)?\s*$)");
  std::smatch literal;
  if (std::regex_match(side, literal, scalar)) terms.push_back(literal[1].str());
  return terms;
}

analysis::ClosedForm sumTerms(std::vector<std::string> const& terms) {
  auto result = analysis::ClosedForm::Constant(0);
  for (auto const& term : terms) result = result + analysis::ClosedForm::Parse(term);
  return result;
}

std::string findRoot(std::unordered_map<std::string, std::string>& parent,
                     std::string value) {
  auto found = parent.find(value);
  if (found == parent.end()) return parent.emplace(value, value).first->second;
  if (found->second != value) found->second = findRoot(parent, found->second);
  return found->second;
}

void unite(std::unordered_map<std::string, std::string>& parent,
           std::string a, std::string b) {
  a = findRoot(parent, a); b = findRoot(parent, b);
  if (a != b) parent[std::max(a, b)] = std::min(a, b);
}

std::vector<std::string> readStrings(llvm::json::Array const* array) {
  std::vector<std::string> result;
  if (!array) return result;
  for (auto const& item : *array)
    if (auto value = item.getAsString()) result.push_back(value->str());
  return result;
}

llvm::StringSet<> const& known();

/// The composite spelling and its Core ATen decomposition classify the same
/// way, so normalization does not move an operator between kinds.
std::string classify(llvm::StringRef target) {
  // An operator no rule covers is not guessed at: it becomes its own task
  // space with conservative semantics (§0.1 prefers degradation to refusal).
  if (!known().contains(target)) return "generic";
  if (target == "aten.linear.default" || target == "aten.matmul.default" ||
      target == "aten.outer.default" || target == "aten.mm.default" ||
      target == "aten.bmm.default") return "gemm";
  if (target.starts_with("aten.mean.") || target.starts_with("aten.softmax.") ||
      target.starts_with("aten._softmax.") || target.starts_with("aten.sum."))
    return "reduction";
  if (target.starts_with("aten.transpose.") || target == "aten.contiguous.default" ||
      target.starts_with("aten.permute.")) return "transpose";
  if (target == "aten.view.default" || target == "aten.clone.default" ||
      target == "aten.alias.default" || target == "aten._to_copy.default")
    return "view";
  if (target == "<built-in function getitem>" || target.starts_with("aten.chunk.") ||
      target.starts_with("aten.split_with_sizes.") ||
      target.starts_with("aten.split.") || target.starts_with("aten.slice."))
    return "slice";
  if (target.starts_with("aten.cat.")) return "concat";
  if (target.starts_with("aten.unsqueeze.") ||
      target.starts_with("aten.repeat_interleave.") ||
      target.starts_with("aten.expand.") || target.starts_with("aten.squeeze."))
    return "broadcast";
  if (target.starts_with("aten._assert_tensor_metadata.") ||
      target.starts_with("aten.sym_size.") || target.starts_with("aten.to."))
    return "frontend";
  return "elementwise";
}

/// The operators a classification rule covers. It is not an admission gate:
/// anything outside it still imports, as one generic task space.
llvm::StringSet<> const& known() {
  static llvm::StringSet<> values = [] {
    llvm::StringSet<> set;
    for (char const* value : {
      "<built-in function add>", "<built-in function getitem>",
      "aten._assert_tensor_metadata.default", "aten.add.Tensor",
      "aten.arange.default", "aten.arange.start", "aten.cat.default",
      "aten.chunk.default", "aten.contiguous.default", "aten.cos.default",
      "aten.div.Tensor", "aten.gt.Tensor", "aten.linear.default",
      "aten.masked_fill.Scalar", "aten.matmul.default", "aten.mean.dim",
      "aten.mul.Tensor", "aten.neg.default", "aten.outer.default",
      "aten.pow.Tensor_Scalar", "aten.repeat_interleave.self_int",
      "aten.rsqrt.default", "aten.silu.default", "aten.sin.default",
      "aten.softmax.int", "aten.sym_size.int", "aten.to.dtype",
      "aten.transpose.int", "aten.unsqueeze.default", "aten.view.default",
      // Core ATen spellings, so a graph normalized by
      // ExportedProgram.run_decompositions() classifies identically.
      "aten._softmax.default", "aten._to_copy.default", "aten.alias.default",
      "aten.arange.start_step", "aten.bmm.default", "aten.clone.default",
      "aten.expand.default", "aten.mm.default", "aten.permute.default",
      "aten.scalar_tensor.default", "aten.sigmoid.default",
      "aten.slice.Tensor", "aten.split_with_sizes.default",
      "aten.squeeze.dim", "aten.sum.dim_IntList", "aten.where.self"})
      set.insert(value);
    return set;
  }();
  return values;
}

mlir::DictionaryAttr dict(mlir::Builder& builder,
                          std::initializer_list<mlir::NamedAttribute> fields) {
  return builder.getDictionaryAttr(fields);
}

/// The CG task kind for a lifted operator.  It follows the role the semantic
/// lifting recognised, never the FX target string: an operator-granularity
/// task space is a fusion of several call_functions and has no single target.
llvm::StringRef taskKindOf(OpRole role) {
  switch (role) {
    case OpRole::kNorm: return "rmsnorm";
    case OpRole::kQkvProjection:
    case OpRole::kProjection: return "gemm";
    case OpRole::kRoPE: return "rope";
    case OpRole::kKVAppend: return "kvappend";
    case OpRole::kAttention: return "attention";
    case OpRole::kActivation:
    case OpRole::kResidualAdd: return "elementwise";
    case OpRole::kGeneric: return "generic";
  }
  return "generic";
}

/// A derived extent as a metric.  The granularity is substituted first
/// because isl needs a literal floor/ceildiv divisor; theta is deliberately
/// left free so a workload dimension survives as an isl parameter (I1).
analysis::QuasiPolynomial metricOf(analysis::ClosedForm const& value,
                                   analysis::ParamBinding const& granularity) {
  analysis::ClosedForm reduced = value.Substitute(granularity);
  std::vector<std::string> free = reduced.FreeSymbols();
  std::sort(free.begin(), free.end());
  free.erase(std::unique(free.begin(), free.end()), free.end());
  std::string prefix;
  if (!free.empty()) prefix = "[" + llvm::join(free, ", ") + "] -> ";
  return analysis::QuasiPolynomial::FromIslText(
      prefix + "{ (" + reduced.ToIslText() + ") }");
}

llvm::StringRef taskKindName(PlanTaskKind kind) {
  switch (kind) {
    case PlanTaskKind::kGemm: return "kGemm";
    case PlanTaskKind::kRMSNorm: return "kRMSNorm";
    case PlanTaskKind::kRoPE: return "kRoPE";
    case PlanTaskKind::kKVAppend: return "kKVAppend";
    case PlanTaskKind::kElementwise: return "kElementwise";
    case PlanTaskKind::kAttention: return "kAttention";
  }
  llvm_unreachable("unknown plan task kind");
}

mlir::DictionaryAttr modelPlanAttr(mlir::Builder& builder,
                                   ModelPlan const& plan) {
  llvm::SmallVector<mlir::Attribute> buffers, gemms, stages, outputs;
  for (auto const& buffer : plan.buffers) {
    llvm::StringRef source = "zero";
    if (buffer.source == PlanBuffer::Source::kFixture) source = "fixture";
    if (buffer.source == PlanBuffer::Source::kWeight) source = "weight";
    buffers.push_back(dict(builder, {
        builder.getNamedAttr("name", builder.getStringAttr(buffer.name)),
        builder.getNamedAttr("constant", builder.getI64IntegerAttr(buffer.constant)),
        builder.getNamedAttr("per_seq", builder.getI64IntegerAttr(buffer.per_seq)),
        builder.getNamedAttr("per_past", builder.getI64IntegerAttr(buffer.per_past)),
        builder.getNamedAttr("per_total", builder.getI64IntegerAttr(buffer.per_total)),
        builder.getNamedAttr("source", builder.getStringAttr(source)),
        builder.getNamedAttr("file", builder.getStringAttr(buffer.file))}));
  }
  for (auto const& gemm : plan.gemms)
    gemms.push_back(dict(builder, {
        builder.getNamedAttr("n", builder.getI64IntegerAttr(gemm.n)),
        builder.getNamedAttr("k", builder.getI64IntegerAttr(gemm.k)),
        builder.getNamedAttr("a", builder.getI64IntegerAttr(gemm.a)),
        builder.getNamedAttr("b", builder.getI64IntegerAttr(gemm.b)),
        builder.getNamedAttr("c", builder.getI64IntegerAttr(gemm.c)),
        builder.getNamedAttr("d", builder.getI64IntegerAttr(gemm.d)),
        builder.getNamedAttr("beta", builder.getF32FloatAttr(gemm.beta))}));
  for (auto const& stage : plan.stages) {
    llvm::SmallVector<std::int64_t> operands;
    for (auto operand : stage.operands) operands.push_back(operand);
    stages.push_back(dict(builder, {
        builder.getNamedAttr("kind", builder.getStringAttr(taskKindName(stage.kind))),
        builder.getNamedAttr("gemm", builder.getI64IntegerAttr(stage.gemm)),
        builder.getNamedAttr("extent", builder.getI64IntegerAttr(stage.extent)),
        builder.getNamedAttr("width", builder.getI64IntegerAttr(stage.width)),
        builder.getNamedAttr("group", builder.getI64IntegerAttr(stage.group)),
        builder.getNamedAttr("operands", builder.getDenseI64ArrayAttr(operands)),
        builder.getNamedAttr("representative", builder.getStringAttr(stage.representative)),
        builder.getNamedAttr("representative_index",
                             builder.getI64IntegerAttr(stage.representative_index))}));
  }
  for (auto const& output : plan.outputs)
    outputs.push_back(dict(builder, {
        builder.getNamedAttr("buffer", builder.getI64IntegerAttr(output.buffer)),
        builder.getNamedAttr("file", builder.getStringAttr(output.file))}));
  return dict(builder, {
      builder.getNamedAttr("buffers", builder.getArrayAttr(buffers)),
      builder.getNamedAttr("gemms", builder.getArrayAttr(gemms)),
      builder.getNamedAttr("stages", builder.getArrayAttr(stages)),
      builder.getNamedAttr("outputs", builder.getArrayAttr(outputs))});
}

}  // namespace

SymbolicShape SymbolicShapeBridge::Parse(
    std::unordered_map<std::string, std::string> const& ranges,
    std::vector<std::string> const& guards,
    std::vector<std::vector<std::string>> const& inputShapes) const {
  SymbolicShape result;
  std::unordered_map<std::string, std::string> parent;
  for (auto const& [symbol, text] : ranges) {
    auto [minimum, maximum] = parseRange(text);
    result.dimensions.push_back(symbol);
    result.ranges.emplace(symbol, ParameterRange{minimum, maximum});
    parent.emplace(symbol, symbol);
  }
  std::sort(result.dimensions.begin(), result.dimensions.end());
  for (auto const& guard : guards) {
    std::size_t split = guard.find("==");
    ShapeConstraint::Predicate predicate = ShapeConstraint::Predicate::kEqual;
    std::size_t width = 2;
    if (split == std::string::npos) {
      split = guard.find("<=");
      predicate = ShapeConstraint::Predicate::kLessEqual;
    }
    if (split == std::string::npos) {
      split = guard.find(">=");
      predicate = ShapeConstraint::Predicate::kLessEqual;
    }
    if (split == std::string::npos)
      throw std::invalid_argument("unsupported ShapeEnv guard predicate: " + guard);
    auto lhs = extractGuardTerms(guard.substr(0, split), inputShapes);
    auto rhs = extractGuardTerms(guard.substr(split + width), inputShapes);
    bool reverseInequality = guard.compare(split, 2, ">=") == 0;
    if (reverseInequality) std::swap(lhs, rhs);
    if (predicate == ShapeConstraint::Predicate::kEqual &&
        guard.substr(0, split).find('%') != std::string::npos) {
      static std::regex const modulo(R"(%\s*([0-9]+))");
      std::smatch divisor;
      std::string left = guard.substr(0, split);
      if (!std::regex_search(left, divisor, modulo))
        throw std::invalid_argument("cannot parse modulo guard: " + guard);
      rhs = {divisor[1].str()};
      predicate = ShapeConstraint::Predicate::kDivisible;
    }
    auto lhsReduced = lhs, rhsReduced = rhs;
    for (auto it = lhsReduced.begin(); it != lhsReduced.end();) {
      auto match = std::find(rhsReduced.begin(), rhsReduced.end(), *it);
      if (match == rhsReduced.end()) { ++it; continue; }
      rhsReduced.erase(match); it = lhsReduced.erase(it);
    }
    bool redundant = predicate == ShapeConstraint::Predicate::kEqual &&
                     lhsReduced.empty() && rhsReduced.empty();
    if (predicate == ShapeConstraint::Predicate::kEqual &&
        lhsReduced.size() == 1 && rhsReduced.size() == 1)
      unite(parent, lhsReduced.front(), rhsReduced.front());
    result.constraints.push_back({sumTerms(lhs), sumTerms(rhs),
                                  predicate,
                                  guard, redundant});
  }
  for (auto const& symbol : result.dimensions)
    result.canonical_symbol[symbol] = findRoot(parent, symbol);
  return result;
}

ExportBridge ReadExportBridge(std::string const& path) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file) throw std::runtime_error("cannot read export bridge JSON: " + path);
  auto parsed = llvm::json::parse(file.get()->getBuffer());
  if (!parsed) throw std::runtime_error("invalid export bridge JSON: " + path);
  auto* root = parsed->getAsObject();
  if (!root || root->getString("schema") != "tilemega.exported_program.v1")
    throw std::runtime_error("unsupported export bridge schema");

  ExportBridge bridge;
  std::set<std::string> unsupported;
  auto* nodes = root->getArray("nodes");
  if (!nodes) throw std::runtime_error("export JSON has no nodes array");
  for (auto const& item : *nodes) {
    auto* object = item.getAsObject();
    if (!object) throw std::runtime_error("node is not an object");
    FxNodeRecord node;
    node.index = static_cast<int>(*object->getInteger("index"));
    node.name = object->getString("name")->str();
    node.op = object->getString("op")->str();
    node.target = object->getString("target")->str();
    node.inputs = readStrings(object->getArray("inputs"));
    node.shape = readStrings(object->getArray("shape"));
    if (node.op == "call_function") {
      if (!known().contains(node.target)) unsupported.insert(node.target);
      bridge.tasks.push_back(node);
    }
    bridge.nodes.push_back(std::move(node));
  }
  bridge.unsupported.assign(unsupported.begin(), unsupported.end());
  auto* signature = root->getObject("signature");
  if (!signature)
    throw std::runtime_error(
        "export bridge has no structured signature; rerun export_bridge.py");
  if (auto* inputs = signature->getArray("inputs")) {
    for (auto const& item : *inputs) {
      auto* object = item.getAsObject();
      if (!object) throw std::runtime_error("signature input is not an object");
      SignatureInput input;
      input.name = object->getString("name")->str();
      input.kind = object->getString("kind")->str();
      if (auto target = object->getString("target")) input.target = target->str();
      if (auto persistent = object->getBoolean("persistent"))
        input.persistent = *persistent;
      bridge.inputs.push_back(std::move(input));
    }
  }
  if (auto* outputs = signature->getArray("outputs"))
    for (auto const& item : *outputs) {
      auto* object = item.getAsObject();
      if (!object) throw std::runtime_error("signature output is not an object");
      bridge.outputs.push_back(object->getString("name")->str());
    }
  if (bridge.inputs.empty() || bridge.outputs.empty())
    throw std::runtime_error("structured signature has no inputs or outputs");
  if (auto* ranges = root->getObject("range_constraints"))
    for (auto const& item : *ranges)
      bridge.range_texts[item.first.str()] = item.second.getAsString()->str();
  bridge.guards = readStrings(root->getArray("guards"));
  return bridge;
}

mlir::OwningOpRef<mlir::ModuleOp> TorchExportImporter::Import(
    std::string const& path, mlir::MLIRContext& context,
    ImportSummary* summary) const {
  context.getOrLoadDialect<dialect::CGDialect>();
  ExportBridge bridge = ReadExportBridge(path);
  std::vector<FxNodeRecord>& allNodes = bridge.nodes;
  std::vector<FxNodeRecord>& tasks = bridge.tasks;
  std::vector<SignatureInput>& signatureInputs = bridge.inputs;
  std::vector<std::string>& signatureOutputs = bridge.outputs;
  // Degradation, not refusal: the operators no rule covers are reported and
  // each becomes one conservative task space.
  if (!bridge.unsupported.empty())
    llvm::errs() << "IMPORT_DEGRADED " << llvm::join(bridge.unsupported, ", ")
                 << "\n";

  std::unordered_map<std::string, FxNodeRecord const*> nodeByName;
  for (auto const& node : allNodes) nodeByName.emplace(node.name, &node);
  std::vector<std::vector<std::string>> userShapes;
  for (auto const& input : signatureInputs)
    if (input.kind == "USER_INPUT") {
      auto found = nodeByName.find(input.name);
      if (found == nodeByName.end())
        throw std::runtime_error("signature names unknown input " + input.name);
      userShapes.push_back(found->second->shape);
    }

  std::unordered_map<std::string, std::string> const& rangeTexts =
      bridge.range_texts;
  std::vector<std::string> const& guards = bridge.guards;
  SymbolicShape symbolic = SymbolicShapeBridge{}.Parse(rangeTexts, guards, userShapes);
  ModelPlan plan = BuildModelPlan(allNodes, signatureInputs, signatureOutputs);
  std::vector<int> stages = FormSemanticStages(tasks, plan);

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  llvm::SmallVector<mlir::NamedAttribute> theta, domains, aliases;
  for (auto const& symbol : symbolic.dimensions) {
    theta.push_back(builder.getNamedAttr(symbol,
        builder.getI64IntegerAttr(symbolic.ranges.at(symbol).minimum)));
    domains.push_back(builder.getNamedAttr(symbol, builder.getStringAttr(rangeTexts.at(symbol))));
    aliases.push_back(builder.getNamedAttr(symbol,
        builder.getStringAttr(symbolic.canonical_symbol.at(symbol))));
  }
  module->setAttr("tilemega.theta", builder.getDictionaryAttr(theta));
  module->setAttr("tilemega.param_domain", builder.getDictionaryAttr(domains));
  module->setAttr("tilemega.symbol_aliases", builder.getDictionaryAttr(aliases));
  // §2.3's g, as the megakernel actually launches: the GEMM TaskBody owns one
  // 128x128 output tile per CTA, and the split-K chunking is a per-operator
  // decision the solver makes later, not a model-wide tile.
  module->setAttr("tilemega.g", dict(builder, {
      builder.getNamedAttr("Tm", builder.getI64IntegerAttr(128)),
      builder.getNamedAttr("Tn", builder.getI64IntegerAttr(128)),
      builder.getNamedAttr("Tkv", builder.getI64IntegerAttr(128))}));
  module->setAttr("tilemega.guard_count", builder.getI64IntegerAttr(guards.size()));
  if (plan.stages.empty())
    llvm::errs() << "IMPORT_DEGRADED no decoder layer; one task space per operator\n";
  else
    module->setAttr("tilemega.model_plan", modelPlanAttr(builder, plan));
  builder.setInsertionPointToStart(module.getBody());

  LiftOptions liftOptions;
  if (symbolic.dimensions.size() > 0) liftOptions.seq_symbol = symbolic.dimensions.front();
  for (auto const& symbol : symbolic.dimensions)
    if (symbol != liftOptions.seq_symbol) { liftOptions.past_symbol = symbol; break; }
  LiftedModel lifted = plan.stages.empty()
                           ? LiftGenericSemantics(tasks, stages, liftOptions)
                           : LiftSemantics(plan, liftOptions);
  analysis::Granularity g = LaunchGranularity(lifted);
  analysis::OperatorGraph graph = analysis::Instantiate(lifted.sem, g);

  analysis::ParamBinding granularityBinding;
  for (auto const& item :
       module->getAttrOfType<mlir::DictionaryAttr>("tilemega.g"))
    granularityBinding.Bind(
        item.getName().str(),
        llvm::cast<mlir::IntegerAttr>(item.getValue()).getInt());
  analysis::ParamBinding known;
  for (auto const& symbol : symbolic.dimensions)
    known.Bind(symbol, symbolic.ranges.at(symbol).minimum);
  for (auto const& [name, value] : granularityBinding.values) known.Bind(name, value);

  // A node Instantiate added (a split reduction's combiner) is named after the
  // operator it combines, so the stage and role follow that operator.
  std::unordered_map<std::string, std::size_t> liftedIndex;
  for (std::size_t i = 0; i < lifted.ops.size(); ++i)
    liftedIndex[lifted.ops[i].name] = i;
  auto liftedOf = [&](std::string const& node) -> LiftedOp const& {
    auto found = liftedIndex.find(node);
    if (found != liftedIndex.end()) return lifted.ops[found->second];
    std::size_t dot = node.rfind('.');
    if (dot != std::string::npos) {
      found = liftedIndex.find(node.substr(0, dot));
      if (found != liftedIndex.end()) return lifted.ops[found->second];
    }
    throw std::runtime_error("instantiated node names no lifted operator: " + node);
  };

  std::unordered_map<std::string, std::string> symbols;
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    auto const& node = graph.nodes[i];
    LiftedOp const& origin = liftedOf(node.name);
    std::string symbol = "t" + std::to_string(i);
    symbols[node.name] = symbol;
    llvm::SmallVector<mlir::NamedAttribute> tiles;
    for (std::size_t axis = 0; axis < node.output.axes.size(); ++axis)
      tiles.push_back(builder.getNamedAttr(
          node.output.axes[axis].name,
          builder.getStringAttr(node.tile[axis].ToString())));
    // A TaskBody that grid-strides over a linearized element range owns a set
    // `Granularity` (axis tiles only) cannot name.  The tiles above then model
    // it at element granularity -- exact, and finer than what one CTA runs --
    // and this field is what tells Codegen the composition is still owed.
    tiles.push_back(builder.getNamedAttr(
        "ownership", builder.getStringAttr(ToString(origin.ownership))));
    mlir::OperationState state(builder.getUnknownLoc(), "tilemega.task_space");
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(symbol));
    state.addAttribute("kind", dialect::TaskKindAttr::get(
        &context, builder.getStringAttr(taskKindOf(origin.role))));
    state.addAttribute("granularity", builder.getDictionaryAttr(tiles));
    llvm::SmallVector<std::string> extents;
    for (auto const& axis : node.output.axes) extents.push_back(axis.extent.ToString());
    state.addAttribute("write_map", dialect::AccessMapAttr::get(&context, dict(builder, {
        builder.getNamedAttr("kind", builder.getStringAttr(taskKindOf(origin.role))),
        builder.getNamedAttr("shape", builder.getStringAttr(
            extents.empty() ? "scalar" : llvm::join(extents, "x")))})));
    state.addAttribute("stage", builder.getI64IntegerAttr(origin.stage));
    state.addAttribute("operator_name", builder.getStringAttr(node.name));
    state.addAttribute("fx_name", builder.getStringAttr(origin.fx_name));
    builder.create(state);
  }

  // The real C, from the analysis layer, on the production path.  There is no
  // fallback: an edge the derivation cannot produce is an import failure, not
  // a placeholder.
  std::vector<analysis::CouplingEdge> derived =
      analysis::CouplingDerivation{}.Derive(graph, known);

  // Part 2: the wait window the generated kernel evaluates per CTA.  It is a
  // property of C at *every* sequence length, so it is fitted at three prefill
  // instantiations and kept only where all three agree -- at the symbolic
  // minimum (S = 1) every row axis collapses to one tile and windows fit that
  // do not hold in general.  A window is additionally admissible only when
  // both TaskBodies declare `kTilePerBlock`: `kElementChunk` makes the
  // CTA->task map a function of gridDim, so the same id names different tasks
  // on the two sides and no per-CTA narrowing is sound.
  std::vector<std::string> waitMaps(derived.size(), "all");
  {
    std::vector<bool> owned(derived.size(), false);
    for (std::size_t i = 0; i < derived.size(); ++i)
      owned[i] =
          liftedOf(derived[i].src.name).ownership == OwnershipKind::kTilePerBlock &&
          liftedOf(derived[i].dst.name).ownership == OwnershipKind::kTilePerBlock;
    std::vector<analysis::WaitWindow> fitted(derived.size());
    bool comparable = true, first_round = true;
    for (long sequence : {256L, 384L, 512L}) {
      analysis::ParamBinding probe = known;
      if (!liftOptions.seq_symbol.empty()) probe.Bind(liftOptions.seq_symbol, sequence);
      if (!liftOptions.past_symbol.empty())
        probe.Bind(liftOptions.past_symbol, sequence - 256L);
      std::vector<analysis::CouplingEdge> at =
          analysis::CouplingDerivation{}.Derive(graph, probe);
      if (at.size() != derived.size()) { comparable = false; break; }
      for (std::size_t i = 0; i < at.size(); ++i) {
        if (at[i].src.name != derived[i].src.name ||
            at[i].dst.name != derived[i].dst.name) { comparable = false; break; }
        analysis::OperatorNode const* source = graph.Find(at[i].src.name);
        analysis::OperatorNode const* sink = graph.Find(at[i].dst.name);
        analysis::WaitWindow here;
        if (owned[i] && source && sink)
          here = analysis::FitWaitWindow(at[i], *source, *sink, probe);
        if (first_round) fitted[i] = here;
        else if (here != fitted[i]) fitted[i] = analysis::WaitWindow{};
      }
      if (!comparable) break;
      first_round = false;
    }
    for (std::size_t i = 0; comparable && i < derived.size(); ++i)
      waitMaps[i] = fitted[i].ToString();
  }
  std::size_t edge = 0;
  for (auto const& item : derived) {
    auto source = symbols.find(item.src.name);
    auto target = symbols.find(item.dst.name);
    if (source == symbols.end() || target == symbols.end())
      throw std::runtime_error("derived coupling names an unknown task space");
    std::string eventName = "e" + std::to_string(edge);
    llvm::SmallVector<std::int64_t> shape;
    llvm::SmallVector<mlir::Attribute> dims;
    analysis::ClosedForm product = analysis::ClosedForm::Constant(1);
    for (auto const& axis : item.event_shape) {
      analysis::ClosedForm reduced = axis.Substitute(granularityBinding);
      shape.push_back(reduced.IsConstant() ? reduced.Eval(known, known)
                                           : mlir::ShapedType::kDynamic);
      dims.push_back(dialect::MetricAttr::get(&context, metricOf(axis, granularityBinding)));
      product = product * axis;
    }
    mlir::OperationState eventState(builder.getUnknownLoc(), "tilemega.event_tensor");
    eventState.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(eventName));
    eventState.addAttribute("event_type", mlir::TypeAttr::get(
        mlir::RankedTensorType::get(shape, builder.getI32Type())));
    eventState.addAttribute("extent", dialect::MetricAttr::get(
        &context, metricOf(product, granularityBinding)));
    eventState.addAttribute("dims", builder.getArrayAttr(dims));
    builder.create(eventState);

    LiftedOp const& consumer = liftedOf(item.dst.name);
    mlir::OperationState state(builder.getUnknownLoc(), "tilemega.coupling");
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr("c" + std::to_string(edge)));
    state.addAttribute("src", mlir::FlatSymbolRefAttr::get(&context, source->second));
    state.addAttribute("dst", mlir::FlatSymbolRefAttr::get(&context, target->second));
    state.addAttribute("read_map", dialect::AccessMapAttr::get(&context,
        dict(builder, {builder.getNamedAttr("kind",
            builder.getStringAttr(taskKindOf(consumer.role)))})));
    state.addAttribute("relation", dialect::CouplingMapAttr::get(&context, item.C));
    state.addAttribute("wait_map", builder.getStringAttr(waitMaps[edge]));
    state.addAttribute("wait", dialect::MetricAttr::get(&context, item.metrics.wait));
    state.addAttribute("fanout", dialect::MetricAttr::get(&context, item.metrics.fanout));
    state.addAttribute("volume", dialect::MetricAttr::get(&context, item.metrics.volume));
    state.addAttribute("count", dialect::MetricAttr::get(&context, item.metrics.count));
    state.addAttribute("tier", dialect::TierAttr::get(
        &context, std::stol(analysis::ToString(item.tier))));
    state.addAttribute("coupling_attrs", dialect::CouplingAttributesAttr::get(
        &context, builder.getStringAttr(analysis::ToString(item.attributes.relation_kind)),
        builder.getStringAttr(analysis::ToString(item.attributes.extent_kind)),
        builder.getStringAttr(analysis::ToString(item.attributes.exactness)),
        builder.getStringAttr(analysis::ToString(item.attributes.runtime_requirement)),
        builder.getStringAttr(analysis::ToString(item.attributes.countability))));
    state.addAttribute("sync_kind", dialect::SyncKindAttr::get(
        &context, builder.getStringAttr("global")));
    state.addAttribute("event", mlir::FlatSymbolRefAttr::get(&context, eventName));
    builder.create(state);
    ++edge;
  }
  for (auto const& node : graph.nodes) {
    mlir::OperationState state(builder.getUnknownLoc(), "tilemega.placement");
    state.addAttribute("task", mlir::FlatSymbolRefAttr::get(&context, symbols.at(node.name)));
    state.addAttribute("map", builder.getDenseI64ArrayAttr({0}));
    state.addAttribute("cluster", builder.getI64IntegerAttr(1));
    builder.create(state);
  }
  if (mlir::failed(mlir::verify(module)))
    throw std::runtime_error("C++ importer produced an invalid CG module");
  if (summary)
    *summary = {graph.nodes.size(), edge, plan.stages.size(), guards.size(),
                bridge.unsupported};
  return mlir::OwningOpRef<mlir::ModuleOp>(module);
}

}  // namespace tilemega::frontend
