// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <mlir/IR/Verifier.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <limits>

namespace tilemega::solver {
namespace {

std::string ReadFile(std::string const& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open generated model: " + path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

/// The body of `constexpr <type> <name>[] = { ... };`.
std::string TableBody(std::string const& source, char const* declaration,
                      std::string const& path) {
  std::size_t begin = source.find(declaration);
  if (begin == std::string::npos) {
    throw std::runtime_error(std::string("no ") + declaration + " in " + path);
  }
  begin = source.find('{', begin);
  std::size_t end = source.find("\n};", begin);
  if (begin == std::string::npos || end == std::string::npos) {
    throw std::runtime_error(std::string("malformed ") + declaration + " in " +
                             path);
  }
  return source.substr(begin + 1, end - begin - 1);
}

/// Split a table body into its top-level `{...}` records.
std::vector<std::string> Records(std::string const& body) {
  std::vector<std::string> out;
  int depth = 0;
  std::size_t start = 0;
  for (std::size_t i = 0; i < body.size(); ++i) {
    if (body[i] == '{') {
      if (depth++ == 0) start = i + 1;
    } else if (body[i] == '}') {
      if (--depth == 0) out.push_back(body.substr(start, i - start));
    }
  }
  return out;
}

/// Fields of one record, split on top-level commas so a nested operand list
/// stays a single field.
std::vector<std::string> Fields(std::string const& record) {
  std::vector<std::string> out;
  int depth = 0;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= record.size(); ++i) {
    if (i == record.size() || (record[i] == ',' && depth == 0)) {
      std::string field = record.substr(start, i - start);
      std::size_t a = field.find_first_not_of(" \t\n");
      std::size_t b = field.find_last_not_of(" \t\n");
      out.push_back(a == std::string::npos ? std::string()
                                           : field.substr(a, b - a + 1));
      start = i + 1;
    } else if (record[i] == '{') {
      ++depth;
    } else if (record[i] == '}') {
      --depth;
    }
  }
  return out;
}

int AsInt(std::string const& text, char const* what) {
  try {
    return std::stoi(text);
  } catch (std::exception const&) {
    throw std::runtime_error(std::string("bad integer for ") + what + ": " +
                             text);
  }
}

StageKind ParseKind(std::string const& text) {
  if (text == "TaskKind::kGemm") return StageKind::kGemm;
  if (text == "TaskKind::kRMSNorm") return StageKind::kRMSNorm;
  if (text == "TaskKind::kRoPE") return StageKind::kRoPE;
  if (text == "TaskKind::kKVAppend") return StageKind::kKVAppend;
  if (text == "TaskKind::kElementwise") return StageKind::kElementwise;
  if (text == "TaskKind::kAttention") return StageKind::kAttention;
  throw std::runtime_error("unmodelled stage kind: " + text);
}

/// The generated operand list is a brace-enclosed run of buffer ids padded
/// with `kNoOperand`; the padding is dropped so an empty list means "reads
/// nothing this model names".
std::vector<int> ParseOperands(std::string const& text) {
  std::vector<int> out;
  std::string token;
  for (char c : text) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
      token.push_back(c);
      continue;
    }
    if (!token.empty()) {
      if (token != "kNoOperand" && std::isdigit(static_cast<unsigned char>(token[0]))) {
        out.push_back(std::stoi(token));
      }
      token.clear();
    }
  }
  if (!token.empty() && token != "kNoOperand" &&
      std::isdigit(static_cast<unsigned char>(token[0]))) {
    out.push_back(std::stoi(token));
  }
  return out;
}

}  // namespace

ModelDims ModelDims::Symbolic(std::string seq_name, int concrete_past) {
  if (seq_name.empty()) throw std::invalid_argument("empty seq parameter name");
  if (concrete_past < 0) throw std::invalid_argument("negative past dimension");
  // Parse through isl rather than accepting arbitrary text as a parameter.
  (void)analysis::QuasiPolynomial::FromIslText(
      "[" + seq_name + "] -> { " + seq_name + " }");
  ModelDims dims;
  dims.past = concrete_past;
  dims.seq_parameter = std::move(seq_name);
  return dims;
}

ModelDescription ModelDescription::FromCouplingGraph(
    mlir::ModuleOp module, ModelDims dims, std::string name) {
#if !TILEMEGA_PARAMETRIC_INPUT
  throw std::runtime_error("parametric CG input disabled at compile time");
#endif
  if (!module || mlir::failed(mlir::verify(module)))
    throw std::invalid_argument("cost input requires a verified CG module");
  auto plan = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  if (!plan) throw std::invalid_argument("CG has no semantic model plan");
  auto integer = [](mlir::DictionaryAttr dict, llvm::StringRef key) -> int {
    auto value = dict.getAs<mlir::IntegerAttr>(key);
    if (!value || value.getInt() < std::numeric_limits<int>::min() ||
        value.getInt() > std::numeric_limits<int>::max())
      throw std::invalid_argument("invalid CG model integer: " + key.str());
    return static_cast<int>(value.getInt());
  };
  auto array = [&](llvm::StringRef key) {
    auto value = plan.getAs<mlir::ArrayAttr>(key);
    if (!value) throw std::invalid_argument("missing CG model array: " + key.str());
    return value;
  };
  ModelDescription model;
  model.name = std::move(name); model.dims = std::move(dims);
  auto roles = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.dimension_roles");
  if (!roles || !roles.getAs<mlir::StringAttr>("seq") || !roles.getAs<mlir::StringAttr>("past"))
    throw std::invalid_argument("CG has no semantic dimension roles; re-import the original export");
  model.seq_metric_parameter = roles.getAs<mlir::StringAttr>("seq").getValue().str();
  model.past_metric_parameter = roles.getAs<mlir::StringAttr>("past").getValue().str();
  if (auto aliases = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.symbol_aliases"))
    for (auto item : aliases)
      model.metric_aliases.emplace_back(item.getName().str(),
          llvm::cast<mlir::StringAttr>(item.getValue()).getValue().str());
  auto dtype = plan.getAs<mlir::StringAttr>("dtype");
  if (!dtype || (dtype.getValue() != "f32" && dtype.getValue() != "bf16"))
    throw std::invalid_argument("unsupported CG model dtype");
  model.dtype = dtype.getValue() == "bf16" ? ScalarType::kBF16 : ScalarType::kF32;
  for (auto item : array("gemms")) {
    auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(item);
    if (!dict) throw std::invalid_argument("malformed CG GEMM plan");
    model.gemms.push_back({integer(dict, "n"), integer(dict, "k"),
                           integer(dict, "a"), integer(dict, "d")});
  }
  for (auto item : array("stages")) {
    auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(item);
    if (!dict) throw std::invalid_argument("malformed CG stage plan");
    auto kind = dict.getAs<mlir::StringAttr>("kind");
    auto operands = dict.getAs<mlir::DenseI64ArrayAttr>("operands");
    if (!kind || !operands) throw std::invalid_argument("incomplete CG stage plan");
    ModelStage stage;
    stage.kind = ParseKind("TaskKind::" + kind.getValue().str());
    stage.gemm = stage.kind == StageKind::kGemm ? integer(dict, "gemm") : -1;
    stage.extent = integer(dict, "extent"); stage.width = integer(dict, "width");
    stage.group = integer(dict, "group");
    for (auto operand : operands.asArrayRef())
      if (operand != std::numeric_limits<std::uint32_t>::max())
        stage.operands.push_back(static_cast<int>(operand));
    model.stages.push_back(std::move(stage));
  }
  std::map<std::string, int> task_stage;
  for (auto task : module.getOps<dialect::TaskSpaceOp>())
    task_stage.emplace(task.getSymName().str(), task.getStage());
  model.stage_successors.resize(model.stages.size());
  for (auto edge : module.getOps<dialect::CouplingOp>()) {
    int const producer = task_stage.at(edge.getSrc().str());
    int const consumer = task_stage.at(edge.getDst().str());
    model.coupling_metrics.push_back({producer, consumer,
        edge.getWait().getValue(), edge.getFanout().getValue(),
        edge.getVolume().getValue(), edge.getCount().getValue()});
    if (producer != consumer) model.stage_successors.at(producer).push_back(consumer);
  }
  for (auto& successors : model.stage_successors) {
    std::sort(successors.begin(), successors.end());
    successors.erase(std::unique(successors.begin(), successors.end()), successors.end());
  }
  for (auto const* attr_name : {"tilemega.theta", "tilemega.g"})
    if (auto attr = module->getAttrOfType<mlir::DictionaryAttr>(attr_name))
      for (auto item : attr)
        if (auto value = llvm::dyn_cast<mlir::IntegerAttr>(item.getValue()))
          model.metric_bindings.Bind(item.getName().str(), value.getInt());
  if (!model.dims.IsSymbolic() && !model.dims.total)
    model.dims.total = model.dims.seq + model.dims.past;
  return model;
}

ModelDescription ModelDescription::SubstituteParams(analysis::ParamBinding const& bindings) const {
  ModelDescription out = *this;
  auto bind = [&](std::string& parameter, int& value) {
    if (parameter.empty()) return;
    auto expression = analysis::QuasiPolynomial::FromIslText(
        "[" + parameter + "] -> { " + parameter + " }");
    long const evaluated = expression.SubstituteParams(bindings).Eval({});
    if (evaluated < 0 || evaluated > std::numeric_limits<int>::max())
      throw std::out_of_range("model dimension outside int range");
    value = static_cast<int>(evaluated); parameter.clear();
  };
  bool const symbolic = out.dims.IsSymbolic();
  bind(out.dims.seq_parameter, out.dims.seq); bind(out.dims.past_parameter, out.dims.past);
  if (out.dims.seq < 0 || out.dims.past < 0 ||
      out.dims.seq > std::numeric_limits<int>::max() - out.dims.past)
    throw std::out_of_range("model total dimension outside int range");
  if (symbolic || !out.dims.total) out.dims.total = out.dims.seq + out.dims.past;
  auto known = metric_bindings;
  for (auto const& [name, value] : bindings.values) known.Bind(name, value);
  known.Bind("S", out.dims.seq).Bind("past", out.dims.past).Bind("L_s", out.dims.total);
  if (!seq_metric_parameter.empty()) known.Bind(seq_metric_parameter, out.dims.seq);
  if (!past_metric_parameter.empty()) known.Bind(past_metric_parameter, out.dims.past);
  for (auto const& [alias, canonical] : metric_aliases)
    if (known.Contains(canonical)) known.Bind(alias, known.At(canonical));
  for (auto& edge : out.coupling_metrics) {
    edge.wait = edge.wait.SubstituteParams(known);
    edge.fanout = edge.fanout.SubstituteParams(known);
    edge.volume = edge.volume.SubstituteParams(known);
    edge.count = edge.count.SubstituteParams(known);
  }
  out.metric_bindings = std::move(known);
  return out;
}

int ModelStage::ReadGranularity() const {
  if (width > 0) return width;
  if (extent > 0) return extent;
  return 0;
}

ModelDescription ModelDescription::FromGeneratedCuda(std::string const& path,
                                                     ModelDims dims,
                                                     std::string name) {
  std::string const source = ReadFile(path);
  ModelDescription model;
  model.name = std::move(name);
  model.dims = dims;
  model.dtype = source.find("ScalarType::kBF16") != std::string::npos
                    ? ScalarType::kBF16
                    : ScalarType::kF32;
  if (model.dims.total == 0) model.dims.total = dims.seq + dims.past;

  for (auto const& record :
       Records(TableBody(source, "constexpr GemmDesc kGemms[]", path))) {
    auto fields = Fields(record);
    if (fields.size() < 2) throw std::runtime_error("short GemmDesc in " + path);
    if (fields.size() < 6) throw std::runtime_error("short GemmDesc in " + path);
    model.gemms.push_back({AsInt(fields[0], "gemm.n"), AsInt(fields[1], "gemm.k"),
                           AsInt(fields[2], "gemm.a"), AsInt(fields[5], "gemm.d")});
  }
  for (auto const& record :
       Records(TableBody(source, "constexpr StageDesc kStages[]", path))) {
    auto fields = Fields(record);
    if (fields.size() < 5) throw std::runtime_error("short StageDesc in " + path);
    ModelStage stage;
    stage.kind = ParseKind(fields[0]);
    stage.gemm = stage.kind == StageKind::kGemm ? AsInt(fields[1], "stage.gemm")
                                                : -1;
    stage.extent = AsInt(fields[2], "stage.extent");
    stage.width = AsInt(fields[3], "stage.width");
    stage.group = AsInt(fields[4], "stage.group");
    if (fields.size() > 5) stage.operands = ParseOperands(fields[5]);
    if (stage.gemm >= static_cast<int>(model.gemms.size())) {
      throw std::runtime_error("stage names a GEMM outside kGemms in " + path);
    }
    model.stages.push_back(stage);
  }
  if (model.gemms.empty() || model.stages.empty()) {
    throw std::runtime_error("empty model tables in " + path);
  }
  model.stage_successors.assign(model.stages.size(), {});
  // Since runtime variants each carry their own exact table the generator emits
  // `kDependencies0`, `kDependencies1`, ...  The stage *graph* is the same in
  // every one of them -- variants differ in the affine window constants, not in
  // which stage feeds which -- and this parser only reads producer/consumer, so
  // variant 0 is read and the rest are equivalent for the cost model.
  char const* const kDependencyTable =
      source.find("constexpr StageDependency kDependencies[]") !=
              std::string::npos
          ? "constexpr StageDependency kDependencies[]"
          : "constexpr StageDependency kDependencies0[]";
  for (auto const& record :
       Records(TableBody(source, kDependencyTable, path))) {
    auto fields = Fields(record);
    if (fields.size() < 2)
      throw std::runtime_error("short StageDependency in " + path);
    int const producer = AsInt(fields[0], "dependency.producer");
    int const consumer = AsInt(fields[1], "dependency.consumer");
    if (producer == consumer) continue;  // the generator's empty-table filler
    if (producer < 0 || consumer < 0 ||
        producer >= static_cast<int>(model.stages.size()) ||
        consumer >= static_cast<int>(model.stages.size()))
      throw std::runtime_error("dependency names a stage outside kStages in " +
                               path);
    model.stage_successors[producer].push_back(consumer);
  }
  return model;
}

double ModelDescription::LiveFootprintBytes() const {
  if (dims.IsSymbolic()) throw std::invalid_argument("bind theta before evaluating footprint");
  double bytes = 0.0;
  double const element_bytes = dtype == ScalarType::kBF16 ? 2.0 : 4.0;
  for (auto const& gemm : gemms) {
    // B is the parameter; A and D are the activations either side of it.
    bytes += element_bytes * gemm.n * gemm.k;
    bytes += element_bytes * dims.seq * (gemm.n + gemm.k);
  }
  for (auto const& stage : stages) {
    if (stage.kind == StageKind::kGemm) continue;
    bytes += element_bytes * dims.total * std::max(stage.extent, 1) *
             std::max(stage.width, 1);
  }
  return bytes;
}

}  // namespace tilemega::solver
