// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Codegen/HostLauncherEmitter.h>
#include <tilemega/Codegen/ScheduleTableEmitter.h>
#include <tilemega/Codegen/SyncEmitter.h>
#include <tilemega/Codegen/TaskBodyEmitter.h>
#include <tilemega/Analysis/DependencyForm.h>
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

/// One row of the emitted `kDependencies[]`. `window.narrowed == false` is the
/// I2 relaxation to the producer's whole launch axis.
struct DependencyRecord {
  std::uint32_t producer;
  std::uint32_t consumer;
  analysis::WaitWindow window;
};

std::string emitModelPlan(mlir::ModuleOp module,
                          std::vector<DependencyRecord> const& dependencies) {
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
  // Sorted by consumer, with a per-stage offset table, so a consumer reads
  // only its own incoming edges. Scanning the whole table once per stage is
  // what made the first L2 slower than the L1 grid barrier (see
  // StageDependency in ModelRuntime.h).
  std::vector<DependencyRecord> byConsumer = dependencies;
  std::stable_sort(byConsumer.begin(), byConsumer.end(),
                   [](auto const& a, auto const& b) {
                     return a.consumer < b.consumer;
                   });
  out << "};\n\nconstexpr StageDependency kDependencies[] = {\n";
  for (auto const& edge : byConsumer) {
    analysis::WaitWindow const& w = edge.window;
    char const* kind = !w.narrowed  ? "kAll"
                       : w.IsIdentity() ? "kIdentity"
                                        : "kWindow";
    out << "  {" << edge.producer << "u, " << edge.consumer
        << "u, StageDependency::Map::" << kind << ", " << w.div << "u, "
        << w.scale << ", " << w.offset << ", " << w.count << "u},\n";
  }
  if (byConsumer.empty())
    out << "  {0u, 0u, StageDependency::Map::kAll, 1u, 0, 0, 1u},\n";
  out << "};\n\nconstexpr std::uint32_t kDependencyOffsets[] = {\n  ";
  {
    std::size_t cursor = 0;
    for (std::size_t stage = 0; stage <= stages.size(); ++stage) {
      while (cursor < byConsumer.size() && byConsumer[cursor].consumer < stage)
        ++cursor;
      out << cursor << "u, ";
    }
  }
  out << "\n};\n\nconstexpr ModelSpec kModel = {kDims, kBuffers, "
      << buffers.size() << "u, kGemms, " << gemms.size()
      << "u, kStages, " << stages.size() << "u, kOutputs, "
      << outputs.size() << "u, kDependencies, " << byConsumer.size()
      << "u, kDependencyOffsets};\n\n}  // namespace\n\n"
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

namespace {

/// Drop every stage dependency that is already implied by a longer path.
///
/// The device-side wait is `epoch[producer] >= iteration + 1`, and a stage's
/// epoch is published only once *every* active CTA of that stage has arrived
/// -- which each of them does only after clearing its own waits. So if
/// `p -> q -> c` is in the graph, then `epoch[q]` published implies
/// `epoch[p]` was already published, and `c`'s direct wait on `p` can never
/// be the one that blocks. Removing it is not a relaxation of the ordering:
/// the happens-before edge survives through `q`, so I2 is untouched and the
/// generated schedule is the same partial order with fewer redundant polls.
///
/// That argument is stage-wide, and a narrowed edge does not support it: a
/// consumer that waits on a *window* of `q` has not established that all of
/// `q` published, so nothing follows about `p`. The reduction therefore runs
/// on the `kAll` sub-graph only -- both hops of the implying path and the
/// path's last edge into `c` must be `kAll` -- while narrowed edges are only
/// ever candidates for removal, never links in the argument.
///
/// The graph is a DAG topologically ordered by stage index (an edge is only
/// recorded when `producer < consumer`), so reachability is one forward
/// sweep.
std::vector<std::pair<std::uint32_t, std::uint32_t>> TransitiveReduction(
    std::map<std::pair<std::uint32_t, std::uint32_t>, analysis::WaitWindow> const&
        edges,
    std::size_t stage_count) {
  std::vector<std::vector<bool>> reaches(stage_count,
                                         std::vector<bool>(stage_count, false));
  std::vector<std::vector<std::uint32_t>> incoming(stage_count);
  for (auto const& [pair, window] : edges)
    if (!window.narrowed) incoming[pair.second].push_back(pair.first);
  for (std::size_t consumer = 0; consumer < stage_count; ++consumer)
    for (std::uint32_t producer : incoming[consumer]) {
      reaches[consumer][producer] = true;
      for (std::size_t s = 0; s < stage_count; ++s)
        if (reaches[producer][s]) reaches[consumer][s] = true;
    }

  std::vector<std::pair<std::uint32_t, std::uint32_t>> kept;
  for (auto const& [pair, window] : edges) {
    bool implied = false;
    for (std::uint32_t other : incoming[pair.second])
      if (other != pair.first && reaches[other][pair.first]) implied = true;
    if (!implied) kept.push_back(pair);
  }
  return kept;
}

}  // namespace

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
  // event dependency, carrying the narrowest wait set every coupling that
  // maps onto the pair agrees on.  A stage holds several task spaces, so one
  // pair can collect several couplings; they are joined, and one relaxed or
  // disagreeing member forces the whole pair back to kAll.
  std::map<std::pair<std::uint32_t, std::uint32_t>, analysis::WaitWindow>
      dependencyPairs;
  // P4.7: a cluster is a property of the launch, not of one edge -- one grid,
  // one cluster dimension, one kind of stage barrier.  So the sync kind and
  // the placement's cluster width have to agree exactly, and a mixture is
  // rejected rather than resolved: there is no kernel that is cluster-scoped
  // on some barriers and grid-scoped on others.
  std::size_t clusterEdges = 0;
  for (auto coupling : module.getOps<dialect::CouplingOp>()) {
    ++couplings;
    llvm::StringRef sync = coupling.getSyncKind().getValue().getValue();
    if (sync == "cluster")
      ++clusterEdges;
    else if (sync != "global")
      throw std::invalid_argument(
          "generator accepts global or cluster synchronization");
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
    if (source->second < target->second) {
      analysis::WaitWindow window;
      if (auto text = coupling.getWaitMap())
        window = analysis::ParseWaitWindow(text->str());
      auto pair = std::make_pair(source->second, target->second);
      auto [at, fresh] = dependencyPairs.emplace(pair, window);
      if (!fresh && at->second != window) at->second = analysis::WaitWindow{};
    }
  }
  int clusterDim = 1;
  for (auto placement : module.getOps<dialect::PlacementOp>()) {
    ++placements;
    int const cluster = static_cast<int>(placement.getCluster());
    if (placements == 1) clusterDim = cluster;
    else if (cluster != clusterDim)
      throw std::invalid_argument(
          "every placement must name the same cluster dimension");
  }
  if (placements != tasks)
    throw std::invalid_argument("every task space must have exactly one placement");
  if (clusterDim > 1 && clusterEdges != couplings)
    throw std::invalid_argument(
        "a clustered placement requires every coupling to synchronize at "
        "cluster scope");
  if (clusterDim == 1 && clusterEdges != 0)
    throw std::invalid_argument(
        "cluster synchronization requires a placement cluster larger than 1");

  // The granularity every wait window was fitted against.  A build that
  // reparameterizes the GEMM task space invalidates the fitted constants, and
  // the harness needs to see that at compile time rather than under-wait.
  std::string windowGrain;
  if (auto g = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.g")) {
    auto extent = [&](llvm::StringRef name) -> long {
      auto value = llvm::dyn_cast_or_null<mlir::IntegerAttr>(g.get(name));
      return value ? value.getInt() : 0;
    };
    long const tm = extent("Tm"), tn = extent("Tn");
    if (tm > 0 && tn > 0)
      windowGrain = "#define TILEMEGA_GENERATED_WINDOW_TILE_M " +
                    std::to_string(tm) + "\n" +
                    "#define TILEMEGA_GENERATED_WINDOW_TILE_N " +
                    std::to_string(tn) + "\n" +
                    "#define TILEMEGA_GENERATED_WINDOW_SPLIT_K 1\n";
  }

  std::ostringstream out;
  out << "// SPDX-License-Identifier: BSD-3-Clause\n"
      << "// Generated by CouplingGraphToCUDA from verified tilemega.* ops.\n"
      << (clusterDim > 1 ? "#define TILEMEGA_GENERATED_CLUSTER_DIM " +
                               std::to_string(clusterDim) + "\n"
                          : std::string())
      << windowGrain
      << SyncEmitter{}.EmitWait("global")
      << SyncEmitter{}.EmitSignal("global")
      << HostLauncherEmitter{}.Emit("l1_kernel")
      << TaskBodyEmitter{}.Emit(module) << "\n";
  std::vector<DependencyRecord> dependencies;
  for (auto const& edge : TransitiveReduction(dependencyPairs, maxStage + 1))
    dependencies.push_back({edge.first, edge.second, dependencyPairs.at(edge)});
  out << emitModelPlan(module, dependencies);
  return out.str();
}

}  // namespace tilemega::codegen
