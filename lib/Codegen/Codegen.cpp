// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Codegen/HostLauncherEmitter.h>
#include <tilemega/Codegen/ScheduleTableEmitter.h>
#include <tilemega/Codegen/SyncEmitter.h>
#include <tilemega/Codegen/TaskBodyEmitter.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>

#include <mlir/IR/Verifier.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace tilemega::codegen {
namespace {
analysis::ParamBinding readBinding(mlir::ModuleOp module, llvm::StringRef name) {
  analysis::ParamBinding result;
  if (auto values = module->getAttrOfType<mlir::DictionaryAttr>(name))
    for (auto item : values)
      if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(item.getValue()))
        result.Bind(item.getName().str(), integer.getInt());
  return result;
}

mlir::Attribute requireField(mlir::DictionaryAttr dictionary,
                             llvm::StringRef name) {
  auto value = dictionary.get(name);
  if (!value)
    throw std::invalid_argument("tilemega.model_plan entry lacks " + name.str());
  return value;
}

std::int64_t integerField(mlir::DictionaryAttr dictionary,
                          llvm::StringRef name) {
  auto value = llvm::dyn_cast<mlir::IntegerAttr>(requireField(dictionary, name));
  if (!value)
    throw std::invalid_argument("tilemega.model_plan field is not integer: " +
                                name.str());
  return value.getInt();
}

std::string stringField(mlir::DictionaryAttr dictionary,
                        llvm::StringRef name) {
  auto value = llvm::dyn_cast<mlir::StringAttr>(requireField(dictionary, name));
  if (!value)
    throw std::invalid_argument("tilemega.model_plan field is not string: " +
                                name.str());
  return value.str();
}

std::string quoteCString(std::string const& text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    if (c == '\\' || c == '"') out << '\\' << static_cast<char>(c);
    else if (c == '\n') out << "\\n";
    else if (c >= 0x20 && c < 0x7f) out << static_cast<char>(c);
    else out << "\\x" << std::hex << std::setw(2) << std::setfill('0')
             << static_cast<unsigned>(c) << std::dec;
  }
  out << '"';
  return out.str();
}

mlir::ArrayAttr arrayField(mlir::DictionaryAttr dictionary,
                           llvm::StringRef name) {
  auto value = llvm::dyn_cast<mlir::ArrayAttr>(requireField(dictionary, name));
  if (!value)
    throw std::invalid_argument("tilemega.model_plan field is not array: " +
                                name.str());
  return value;
}

mlir::DictionaryAttr dictionaryEntry(mlir::Attribute value,
                                     llvm::StringRef collection) {
  auto result = llvm::dyn_cast<mlir::DictionaryAttr>(value);
  if (!result)
    throw std::invalid_argument("tilemega.model_plan " + collection.str() +
                                " entry is not a dictionary");
  return result;
}

std::string emitModelPlan(
    mlir::ModuleOp module,
    std::vector<std::tuple<std::uint32_t, std::uint32_t, bool>> const& dependencies) {
  auto plan = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  if (!plan)
    throw std::invalid_argument(
        "verified CG has no tilemega.model_plan; import through the C++ frontend");
  auto buffers = arrayField(plan, "buffers");
  auto gemms = arrayField(plan, "gemms");
  auto stages = arrayField(plan, "stages");
  auto outputs = arrayField(plan, "outputs");
  if (buffers.empty() || stages.empty() || outputs.empty())
    throw std::invalid_argument("tilemega.model_plan has an empty required table");

  std::ostringstream out;
  out << "namespace {\nusing namespace tilemega::codegen;\n\n"
      << "constexpr ModelDims kDims = {};\n\n"
      << "constexpr BufferDesc kBuffers[] = {\n";
  for (auto value : buffers) {
    auto item = dictionaryEntry(value, "buffers");
    std::string source = stringField(item, "source");
    std::string sourceEnum;
    if (source == "zero") sourceEnum = "kZero";
    else if (source == "fixture") sourceEnum = "kFixture";
    else if (source == "weight") sourceEnum = "kWeight";
    else throw std::invalid_argument("unknown model buffer source: " + source);
    std::string file = stringField(item, "file");
    out << "  {" << quoteCString(stringField(item, "name")) << ", "
        << integerField(item, "constant") << "u, "
        << integerField(item, "per_seq") << "u, "
        << integerField(item, "per_past") << "u, "
        << integerField(item, "per_total") << "u, BufferSource::"
        << sourceEnum << ", "
        << (file.empty() ? "nullptr" : quoteCString(file)) << "},\n";
  }
  out << "};\n\nconstexpr GemmDesc kGemms[] = {\n";
  for (auto value : gemms) {
    auto item = dictionaryEntry(value, "gemms");
    auto beta = llvm::dyn_cast<mlir::FloatAttr>(requireField(item, "beta"));
    if (!beta) throw std::invalid_argument("model GEMM beta is not floating point");
    out << "  {" << integerField(item, "n") << ", "
        << integerField(item, "k") << ", " << integerField(item, "a")
        << "u, " << integerField(item, "b") << "u, "
        << integerField(item, "c") << "u, " << integerField(item, "d")
        << "u, " << std::showpoint << std::setprecision(9)
        << beta.getValueAsDouble() << std::noshowpoint << "f},\n";
  }
  out << "};\n\nconstexpr StageDesc kStages[] = {\n";
  for (auto value : stages) {
    auto item = dictionaryEntry(value, "stages");
    auto operands = llvm::dyn_cast<mlir::DenseI64ArrayAttr>(
        requireField(item, "operands"));
    if (!operands || operands.size() != 8)
      throw std::invalid_argument("model stage must have exactly eight operands");
    out << "  {TaskKind::" << stringField(item, "kind") << ", "
        << integerField(item, "gemm") << "u, "
        << integerField(item, "extent") << "u, "
        << integerField(item, "width") << "u, "
        << integerField(item, "group") << "u, {";
    for (std::size_t i = 0; i < operands.size(); ++i) {
      if (i) out << ", ";
      auto operand = operands[i];
      out << (operand == std::numeric_limits<std::uint32_t>::max()
                  ? "kNoOperand"
                  : std::to_string(operand) + "u");
    }
    out << "}},\n";
  }
  out << "};\n\nconstexpr OutputDesc kOutputs[] = {\n";
  for (auto value : outputs) {
    auto item = dictionaryEntry(value, "outputs");
    out << "  {" << integerField(item, "buffer") << "u, "
        << quoteCString(stringField(item, "file")) << "},\n";
  }
  out << "};\n\nconstexpr StageDependency kDependencies[] = {\n";
  for (auto [producer, consumer, identity] : dependencies)
    out << "  {" << producer << "u, " << consumer
        << "u, StageDependency::Map::"
        << (identity ? "kIdentity" : "kAll") << "},\n";
  out << "};\n\nconstexpr ModelSpec kModel = {kDims, kBuffers, "
      << buffers.size() << "u, kGemms, " << gemms.size()
      << "u, kStages, " << stages.size() << "u, kOutputs, "
      << outputs.size() << "u, kDependencies, " << dependencies.size()
      << "u};\n\n}  // namespace\n\n"
      << "int main(int argc, char** argv) {\n"
      << "  if (argc != 2) { std::fprintf(stderr, \"usage: e2e FIXTURE_DIR\\n\"); return 2; }\n"
      << "  return tilemega::codegen::RunModel(kModel, argv[1]);\n}\n";
  return out.str();
}

}  // namespace

std::string TaskBodyEmitter::Emit(mlir::ModuleOp module) const {
  (void)module;
  return "#include <tilemega/Codegen/tasks/ModelHarness.cuh>\n";
}

std::string SyncEmitter::EmitWait(std::string const& event) const {
  return "#define TILEMEGA_GENERATED_WAIT_" + event +
      "(ev, need) do { \\\n"
      "  while (atomicAdd((ev), 0ull) < (need)) __nanosleep(64); \\\n"
      "} while (0)\n";
}

std::string SyncEmitter::EmitSignal(std::string const& event) const {
  return "#define TILEMEGA_GENERATED_NOTIFY_" + event +
      "(ev, value) atomicExch((ev), (value))\n";
}

std::vector<ScheduleEntry> ScheduleTableEmitter::Emit(
    std::vector<int> const& order) const {
  std::vector<ScheduleEntry> result;
  for (int tile : order)
    result.push_back({0u, static_cast<std::uint32_t>(tile)});
  return result;
}

std::string ScheduleTableEmitter::EmitStageCounts(
    std::vector<std::size_t> const& counts) const {
  std::ostringstream out;
  out << "#define TILEMEGA_GENERATED_STAGE_TASK_COUNTS {";
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (i) out << ",";
    out << counts[i];
  }
  out << "}\n"
      << "#define TILEMEGA_GENERATED_SCHEDULE {\\\n";
  for (std::size_t i = 0; i < counts.size(); ++i)
    out << "  {" << i << "u, " << i << "u, 0u, "
        << counts[i] << "u}, \\\n";
  out << "}\n";
  return out.str();
}

std::string HostLauncherEmitter::Emit(std::string const& kernel) const {
  return "#define TILEMEGA_GENERATED_RESIDENT_GRID(target, function, block_size, dynamic_smem) \\\n"
      "  ((target).res.num_sms * (target).ActiveBlocksPerSM( \\\n"
      "      reinterpret_cast<void const*>(function), (block_size), (dynamic_smem)))\n"
      "// Model invocation tables are allocated on device and passed as one "
      "Params const* by ModelHarness (F-17b).\n"
      "// Launcher specialization: " + kernel + "\n";
}

std::string CouplingGraphToCUDA::Lower(mlir::ModuleOp module) const {
  if (!module || mlir::failed(mlir::verify(module)))
    throw std::invalid_argument("CouplingGraphToCUDA requires a verified CG ModuleOp");

  std::size_t tasks = 0, couplings = 0, placements = 0;
  auto theta = readBinding(module, "tilemega.theta");
  auto granularity = readBinding(module, "tilemega.g");
  analysis::ParamBinding known = theta;
  for (auto const& [name, value] : granularity.values) known.Bind(name, value);
  int maxStage = -1;
  std::unordered_map<std::string, std::uint32_t> taskStages;
  for (auto task : module.getOps<dialect::TaskSpaceOp>()) {
    ++tasks;
    maxStage = std::max(maxStage, static_cast<int>(task.getStage()));
    taskStages.emplace(task.getSymName().str(),
                       static_cast<std::uint32_t>(task.getStage()));
  }
  if (maxStage < 0) throw std::invalid_argument("CG has no task spaces");
  std::vector<std::size_t> stageCounts(maxStage + 1, 0);
  for (auto task : module.getOps<dialect::TaskSpaceOp>())
    ++stageCounts[task.getStage()];
  // Every producer-earlier/consumer-later stage pair needs a device-side
  // event dependency; record the set of pairs only.
  std::set<std::pair<std::uint32_t, std::uint32_t>> dependencyPairs;
  for (auto coupling : module.getOps<dialect::CouplingOp>()) {
    ++couplings;
    if (coupling.getSyncKind().getValue().getValue() != "global")
      throw std::invalid_argument("Phase-2 generator accepts only global synchronization");
    // Force semantic conversion through L3a; codegen never reads the printed
    // quasi-polynomial payload as an ad-hoc integer.
    (void)coupling.getWait().getValue().Eval(known);
    (void)coupling.getFanout().getValue().Eval(known);
    (void)coupling.getVolume().getValue().Eval(known);
    (void)coupling.getCount().getValue().Eval(known);
    (void)coupling.getRelation().getMap();
    auto source = taskStages.find(coupling.getSrc().str());
    auto target = taskStages.find(coupling.getDst().str());
    if (source == taskStages.end() || target == taskStages.end())
      throw std::invalid_argument("coupling names an unknown task space");
    if (source->second < target->second)
      dependencyPairs.emplace(source->second, target->second);
  }
  for (auto placement : module.getOps<dialect::PlacementOp>()) {
    ++placements;
    if (placement.getCluster() != 1)
      throw std::invalid_argument("Phase-2 generator accepts round-robin cluster=1 placement");
  }
  if (placements != tasks)
    throw std::invalid_argument("every task space must have exactly one placement");

  std::ostringstream out;
  out << "// SPDX-License-Identifier: BSD-3-Clause\n"
      << "// Generated by CouplingGraphToCUDA from verified tilemega.* ops.\n"
      << SyncEmitter{}.EmitWait("global")
      << SyncEmitter{}.EmitSignal("global")
      << HostLauncherEmitter{}.Emit("l1_kernel")
      << TaskBodyEmitter{}.Emit(module) << "\n";
  // A semantic identity C is insufficient to prove that blockIdx.x names the
  // same tile in two independently implemented TaskBodies.  Until the
  // TaskBody ABI carries an explicit CTA->task ownership map, I2 requires the
  // all-active-producer relaxation (`kAll`) for every stage dependency.
  std::vector<std::tuple<std::uint32_t, std::uint32_t, bool>> dependencies;
  for (auto const& edge : dependencyPairs)
    dependencies.emplace_back(edge.first, edge.second, false);
  out << emitModelPlan(module, dependencies);
  return out.str();
}

}  // namespace tilemega::codegen
