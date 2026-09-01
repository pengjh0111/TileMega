// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/SymbolicShapeBridge.h>
#include <tilemega/Frontend/TorchExportImporter.h>
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

struct NodeRecord {
  int index = 0;
  std::string name, op, target;
  std::vector<std::string> inputs, shape;
};

std::vector<std::string> readStrings(llvm::json::Array const* array) {
  std::vector<std::string> result;
  if (!array) return result;
  for (auto const& item : *array)
    if (auto value = item.getAsString()) result.push_back(value->str());
  return result;
}

std::string classify(llvm::StringRef target) {
  if (target == "aten.linear.default" || target == "aten.matmul.default" ||
      target == "aten.outer.default") return "gemm";
  if (target.starts_with("aten.mean.") || target.starts_with("aten.softmax."))
    return "reduction";
  if (target.starts_with("aten.transpose.") || target == "aten.contiguous.default")
    return "transpose";
  if (target == "aten.view.default") return "view";
  if (target == "<built-in function getitem>" || target.starts_with("aten.chunk."))
    return "slice";
  if (target.starts_with("aten.cat.")) return "concat";
  if (target.starts_with("aten.unsqueeze.") || target.starts_with("aten.repeat_interleave."))
    return "broadcast";
  if (target.starts_with("aten._assert_tensor_metadata.") ||
      target.starts_with("aten.sym_size.") || target.starts_with("aten.to."))
    return "frontend";
  return "elementwise";
}

llvm::StringSet<> const& whitelist() {
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
      "aten.transpose.int", "aten.unsqueeze.default", "aten.view.default"})
      set.insert(value);
    return set;
  }();
  return values;
}

// Explicit, tested two-layer Llama rule.  It is deliberately centralized: no
// stage decision is made by the Python bridge or hidden in code generation.
std::vector<int> formStages(std::vector<NodeRecord> const& tasks) {
  std::vector<std::size_t> linear;
  for (std::size_t i = 0; i < tasks.size(); ++i)
    if (tasks[i].target == "aten.linear.default") linear.push_back(i);
  if (linear.size() != 14)
    throw std::runtime_error("stage rule expects 14 Llama linear ops; found " +
                             std::to_string(linear.size()));
  std::vector<int> stage(tasks.size(), -1);
  for (int layer = 0; layer < 2; ++layer) {
    std::size_t q = linear[layer * 7], k = linear[layer * 7 + 1];
    std::size_t v = linear[layer * 7 + 2], o = linear[layer * 7 + 3];
    std::size_t gate = linear[layer * 7 + 4], up = linear[layer * 7 + 5];
    std::size_t down = linear[layer * 7 + 6];
    std::size_t begin = layer == 0 ? 0 : linear[6];
    if (layer == 1) {
      while (begin < q && tasks[begin].target != "aten.add.Tensor") ++begin;
      if (begin < q) ++begin;
    }
    std::size_t end = layer == 1 ? tasks.size() : begin;
    if (layer == 0) {
      end = down;
      while (end < linear[7] && tasks[end].target != "aten.add.Tensor") ++end;
      if (end < linear[7]) ++end;
    }
    std::size_t rope = v + 1;
    while (rope < o && (tasks[rope].target == "aten.view.default" ||
                        tasks[rope].target == "aten.transpose.int")) ++rope;
    std::size_t attention = rope;
    while (attention < o && tasks[attention].target != "aten.matmul.default") ++attention;
    std::size_t postNorm = o + 1;
    while (postNorm < gate && tasks[postNorm].target != "aten.add.Tensor") ++postNorm;
    if (postNorm < gate) ++postNorm;
    auto fill = [&](std::size_t first, std::size_t last, int local) {
      for (std::size_t i = first; i < last; ++i) stage[i] = layer * 12 + local;
    };
    fill(begin, q, 0); fill(q, k, 1); fill(k, v, 2); fill(v, rope, 3);
    fill(rope, attention, 4); fill(attention, o, 5); fill(o, postNorm, 6);
    fill(postNorm, gate, 7); fill(gate, up, 8); fill(up, down, 9);
    for (std::size_t i = gate + 1; i < down; ++i)
      if (tasks[i].target == "aten.silu.default" || tasks[i].target == "aten.mul.Tensor")
        stage[i] = layer * 12 + 10;
    fill(down, end, 11);
  }
  if (llvm::any_of(stage, [](int value) { return value < 0; }))
    throw std::runtime_error("explicit stage rule left an ATen task unassigned");
  return stage;
}

mlir::DictionaryAttr dict(mlir::Builder& builder,
                          std::initializer_list<mlir::NamedAttribute> fields) {
  return builder.getDictionaryAttr(fields);
}

mlir::DictionaryAttr fixedRelation(mlir::Builder& builder,
                                   mlir::MLIRContext& context,
                                   std::string const& source,
                                   llvm::StringRef kind) {
  mlir::ArrayAttr consumers;
  mlir::ArrayAttr coordinates;
  mlir::ArrayAttr ranges = builder.getArrayAttr({});
  if (kind == "gemm" || kind == "reduction") {
    consumers = builder.getArrayAttr({builder.getStringAttr("m"),
                                      builder.getStringAttr("n")});
    coordinates = builder.getArrayAttr({builder.getStringAttr("m")});
    ranges = builder.getArrayAttr({dict(builder, {
        builder.getNamedAttr("name", builder.getStringAttr("k")),
        builder.getNamedAttr("begin", builder.getStringAttr("0")),
        // Phase 1 retains the quantified structure but conservatively uses a
        // unit extent. Phase 3 replaces it from W^-1 o R.
        builder.getNamedAttr("extent", builder.getStringAttr("1"))})});
  } else if (kind == "broadcast") {
    consumers = builder.getArrayAttr({builder.getStringAttr("m"),
                                      builder.getStringAttr("n")});
    coordinates = builder.getArrayAttr({builder.getStringAttr("m")});
  } else if (kind == "transpose") {
    consumers = builder.getArrayAttr({builder.getStringAttr("m"),
                                      builder.getStringAttr("n")});
    coordinates = builder.getArrayAttr({builder.getStringAttr("n"),
                                        builder.getStringAttr("m")});
  } else {
    consumers = builder.getArrayAttr({builder.getStringAttr("i")});
    coordinates = builder.getArrayAttr({builder.getStringAttr("i")});
  }
  return dict(builder, {
      builder.getNamedAttr("consumer", consumers),
      builder.getNamedAttr("producers", builder.getArrayAttr({dict(builder, {
          builder.getNamedAttr("source", builder.getStringAttr(source)),
          builder.getNamedAttr("coordinates", coordinates),
          builder.getNamedAttr("ranges", ranges)})})),
      builder.getNamedAttr("parameters", builder.getArrayAttr({})),
      builder.getNamedAttr("fiber", dialect::ClosedFormAttr::get(
          &context, analysis::ClosedForm::Constant(1))),
      builder.getNamedAttr("image", dialect::ClosedFormAttr::get(
          &context, analysis::ClosedForm::Constant(1)))});
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

  std::vector<NodeRecord> tasks;
  std::vector<std::vector<std::string>> userShapes;
  std::set<std::string> unsupported;
  auto* nodes = root->getArray("nodes");
  if (!nodes) throw std::runtime_error("export JSON has no nodes array");
  for (auto const& item : *nodes) {
    auto* object = item.getAsObject();
    if (!object) throw std::runtime_error("node is not an object");
    NodeRecord node;
    node.index = static_cast<int>(*object->getInteger("index"));
    node.name = object->getString("name")->str();
    node.op = object->getString("op")->str();
    node.target = object->getString("target")->str();
    node.inputs = readStrings(object->getArray("inputs"));
    node.shape = readStrings(object->getArray("shape"));
    if (node.op == "placeholder" && node.index >= 20) userShapes.push_back(node.shape);
    if (node.op == "call_function") {
      if (!whitelist().contains(node.target)) unsupported.insert(node.target);
      tasks.push_back(std::move(node));
    }
  }
  if (!unsupported.empty())
    throw std::runtime_error("operators outside Phase-1 whitelist: " +
                             llvm::join(unsupported, ", "));
  std::unordered_map<std::string, std::string> rangeTexts;
  if (auto* ranges = root->getObject("range_constraints"))
    for (auto const& item : *ranges)
      rangeTexts[item.first.str()] = item.second.getAsString()->str();
  auto guards = readStrings(root->getArray("guards"));
  SymbolicShape symbolic = SymbolicShapeBridge{}.Parse(rangeTexts, guards, userShapes);
  std::vector<int> stages = formStages(tasks);

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
      eventState.addAttribute("extent", dialect::ClosedFormAttr::get(
          &context, analysis::ClosedForm::Constant(1)));
      builder.create(eventState);
      std::string taskKind = classify(task.target);
      auto relation = fixedRelation(builder, context, source->second, taskKind);
      mlir::OperationState state(builder.getUnknownLoc(), "tilemega.coupling");
      state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                         builder.getStringAttr("c" + std::to_string(edge)));
      state.addAttribute("src", mlir::FlatSymbolRefAttr::get(&context, source->second));
      state.addAttribute("dst", mlir::FlatSymbolRefAttr::get(&context, symbols.at(task.name)));
      state.addAttribute("read_map", dialect::AccessMapAttr::get(&context,
          dict(builder, {builder.getNamedAttr("kind", builder.getStringAttr(classify(task.target)))})));
      state.addAttribute("relation", dialect::CouplingMapAttr::get(&context, relation));
      for (llvm::StringRef metric : {"wait", "fanout", "volume", "count"})
        state.addAttribute(metric, dialect::ClosedFormAttr::get(
            &context, analysis::ClosedForm::Constant(1)));
      state.addAttribute("tier", dialect::TierAttr::get(&context, 0));
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
  if (summary) *summary = {tasks.size(), edge, 24, guards.size()};
  return mlir::OwningOpRef<mlir::ModuleOp>(module);
}

}  // namespace tilemega::frontend
