// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ModelDescription.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

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
