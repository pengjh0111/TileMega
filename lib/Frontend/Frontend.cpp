// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/SymbolicShapeBridge.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Frontend/ModelPlan.h>
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

// Phase 1's explicit, fixed per-ATen-op coupling: not derived from W^-1 o R
// (real derivation is lib/Analysis/CouplingDerivation.cpp, exercised through
// the OperatorGraph abstraction ReferenceModels.cpp builds -- this importer
// still runs at per-call_function granularity and does not construct that
// abstraction from FX; wiring the two together is out of this round's scope,
// recorded as residual debt in TileMega_skeleton.md §1.5.1). This is a single
// fixed producer-to-consumer coupling, `{ [0] -> [0] }`: one placeholder task
// on each side, matching the fixed wait=fanout=volume=count=1 and event
// extent=1 this importer has always used. Before the isl/barvinok migration
// this shape varied by operator kind (gemm/reduction got a quantified k
// range, transpose swapped coordinates, ...); none of that per-kind shape
// was ever read by anything downstream (Codegen only force-evaluates it, see
// lib/Codegen/Codegen.cpp), so the migration collapses it to one honest,
// uniform placeholder rather than reproducing dead structure that cannot be
// expressed in CouplingMapAttr's new isl-map payload anyway.
analysis::CouplingRelation fixedRelation(llvm::StringRef /*kind*/) {
  return analysis::CouplingRelation::FromIslText("{ [0] -> [0] }");
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

mlir::OwningOpRef<mlir::ModuleOp> TorchExportImporter::Import(
    std::string const& path, mlir::MLIRContext& context,
    ImportSummary* summary) const {
  context.getOrLoadDialect<dialect::CGDialect>();
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file) throw std::runtime_error("cannot read export bridge JSON: " + path);
  auto parsed = llvm::json::parse(file.get()->getBuffer());
  if (!parsed) throw std::runtime_error("invalid export bridge JSON: " + path);
  auto* root = parsed->getAsObject();
  if (!root || root->getString("schema") != "tilemega.exported_program.v1")
    throw std::runtime_error("unsupported export bridge schema");

  std::vector<FxNodeRecord> allNodes, tasks;
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
      tasks.push_back(node);
    }
    allNodes.push_back(std::move(node));
  }
  // Degradation, not refusal: the operators no rule covers are reported and
  // each becomes one conservative task space.
  if (!unsupported.empty())
    llvm::errs() << "IMPORT_DEGRADED " << llvm::join(unsupported, ", ") << "\n";
  std::vector<SignatureInput> signatureInputs;
  std::vector<std::string> signatureOutputs;
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
      signatureInputs.push_back(std::move(input));
    }
  }
  if (auto* outputs = signature->getArray("outputs"))
    for (auto const& item : *outputs) {
      auto* object = item.getAsObject();
      if (!object) throw std::runtime_error("signature output is not an object");
      signatureOutputs.push_back(object->getString("name")->str());
    }
  if (signatureInputs.empty() || signatureOutputs.empty())
    throw std::runtime_error("structured signature has no inputs or outputs");

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

  std::unordered_map<std::string, std::string> rangeTexts;
  if (auto* ranges = root->getObject("range_constraints"))
    for (auto const& item : *ranges)
      rangeTexts[item.first.str()] = item.second.getAsString()->str();
  auto guards = readStrings(root->getArray("guards"));
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
  module->setAttr("tilemega.g", dict(builder, {
      builder.getNamedAttr("token_tile", builder.getI64IntegerAttr(1)),
      builder.getNamedAttr("hidden_tile", builder.getI64IntegerAttr(128)),
      builder.getNamedAttr("head_tile", builder.getI64IntegerAttr(1)),
      builder.getNamedAttr("head_dim_tile", builder.getI64IntegerAttr(128))}));
  module->setAttr("tilemega.guard_count", builder.getI64IntegerAttr(guards.size()));
  if (plan.stages.empty())
    llvm::errs() << "IMPORT_DEGRADED no decoder layer; one task space per operator\n";
  else
    module->setAttr("tilemega.model_plan", modelPlanAttr(builder, plan));
  builder.setInsertionPointToStart(module.getBody());

  std::unordered_map<std::string, std::string> symbols;
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    auto const& task = tasks[i];
    std::string symbol = "t" + std::to_string(task.index);
    symbols[task.name] = symbol;
    mlir::OperationState state(builder.getUnknownLoc(), "tilemega.task_space");
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(symbol));
    state.addAttribute("kind", dialect::TaskKindAttr::get(
        &context, builder.getStringAttr(classify(task.target))));
    state.addAttribute("granularity", dict(builder, {
        builder.getNamedAttr("token_tile", builder.getI64IntegerAttr(1)),
        builder.getNamedAttr("hidden_tile", builder.getI64IntegerAttr(128))}));
    state.addAttribute("write_map", dialect::AccessMapAttr::get(&context, dict(builder, {
        builder.getNamedAttr("kind", builder.getStringAttr(classify(task.target))),
        builder.getNamedAttr("shape", builder.getStringAttr(
            task.shape.empty() ? "scalar" : llvm::join(task.shape, "x")))})));
    state.addAttribute("stage", builder.getI64IntegerAttr(stages[i]));
    state.addAttribute("operator_name", builder.getStringAttr(task.target));
    state.addAttribute("fx_name", builder.getStringAttr(task.name));
    builder.create(state);
  }

  std::size_t edge = 0;
  for (auto const& task : tasks) {
    std::unordered_set<std::string> seen;
    for (auto const& input : task.inputs) {
      auto source = symbols.find(input);
      if (source == symbols.end() || !seen.insert(source->second).second) continue;
      std::string eventName = "e" + std::to_string(edge);
      mlir::OperationState eventState(builder.getUnknownLoc(), "tilemega.event_tensor");
      eventState.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(eventName));
      eventState.addAttribute("event_type", mlir::TypeAttr::get(
          mlir::RankedTensorType::get({1}, builder.getI32Type())));
      eventState.addAttribute("extent", dialect::MetricAttr::get(
          &context, analysis::QuasiPolynomial::Constant(1)));
      builder.create(eventState);
      std::string taskKind = classify(task.target);
      auto relation = fixedRelation(taskKind);
      mlir::OperationState state(builder.getUnknownLoc(), "tilemega.coupling");
      state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                         builder.getStringAttr("c" + std::to_string(edge)));
      state.addAttribute("src", mlir::FlatSymbolRefAttr::get(&context, source->second));
      state.addAttribute("dst", mlir::FlatSymbolRefAttr::get(&context, symbols.at(task.name)));
      state.addAttribute("read_map", dialect::AccessMapAttr::get(&context,
          dict(builder, {builder.getNamedAttr("kind", builder.getStringAttr(classify(task.target)))})));
      state.addAttribute("relation", dialect::CouplingMapAttr::get(&context, relation));
      for (llvm::StringRef metric : {"wait", "fanout", "volume", "count"})
        state.addAttribute(metric, dialect::MetricAttr::get(
            &context, analysis::QuasiPolynomial::Constant(1)));
      state.addAttribute("tier", dialect::TierAttr::get(&context, 0));
      // The importer's placeholder edge is the identity on one point, so all
      // five attributes are their trivial values; the verifier checks that
      // this is consistent with the tier above.
      state.addAttribute("coupling_attrs", dialect::CouplingAttributesAttr::get(
          &context, builder.getStringAttr("affine"),
          builder.getStringAttr("static_literal"),
          builder.getStringAttr("exact"), builder.getStringAttr("none"),
          builder.getStringAttr("constant")));
      state.addAttribute("sync_kind", dialect::SyncKindAttr::get(
          &context, builder.getStringAttr("global")));
      state.addAttribute("event", mlir::FlatSymbolRefAttr::get(&context, eventName));
      builder.create(state);
      ++edge;
    }
  }
  for (auto const& task : tasks) {
    mlir::OperationState state(builder.getUnknownLoc(), "tilemega.placement");
    state.addAttribute("task", mlir::FlatSymbolRefAttr::get(&context, symbols.at(task.name)));
    state.addAttribute("map", builder.getDenseI64ArrayAttr({0}));
    state.addAttribute("cluster", builder.getI64IntegerAttr(1));
    builder.create(state);
  }
  if (mlir::failed(mlir::verify(module)))
    throw std::runtime_error("C++ importer produced an invalid CG module");
  if (summary)
    *summary = {tasks.size(), edge, plan.stages.size(), guards.size(),
                {unsupported.begin(), unsupported.end()}};
  return mlir::OwningOpRef<mlir::ModuleOp>(module);
}

}  // namespace tilemega::frontend
