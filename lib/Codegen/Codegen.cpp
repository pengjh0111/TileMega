// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/SymbolicShapeBridge.h>
#include <tilemega/Codegen/HostLauncherEmitter.h>
#include <tilemega/Codegen/ScheduleTableEmitter.h>
#include <tilemega/Codegen/SyncEmitter.h>
#include <tilemega/Codegen/TaskBodyEmitter.h>
#include <tilemega/Analysis/DependencyForm.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Solver/ListScheduler.h>

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
bool readResidentConstraint(mlir::ModuleOp module) {
  std::size_t count=0, explicit_count=0;
  for (auto placement:module.getOps<dialect::PlacementOp>()) {
    ++count;
    if (auto attr=placement->getAttr("resident_only")) {
      auto flag=llvm::dyn_cast<mlir::BoolAttr>(attr);
      if (!flag || !flag.getValue())
        throw std::invalid_argument("placement requires resident_only=true; no over-resident proof");
      ++explicit_count;
    }
  }
  if (explicit_count && explicit_count!=count)
    throw std::invalid_argument("resident constraint must cover every placement");
  return explicit_count!=0;
}

bool readBalancedPlacement(mlir::ModuleOp module) {
  std::size_t count=0,balanced=0;
  for (auto placement:module.getOps<dialect::PlacementOp>()) {
    ++count;
    if (auto attr=placement->getAttr("mapping_mode")) {
      auto mode=llvm::dyn_cast<mlir::StringAttr>(attr);
      if (!mode || mode.getValue()!="balanced")
        throw std::invalid_argument("unknown placement mapping_mode");
      ++balanced;
    }
  }
  if (balanced && (balanced!=count || !readResidentConstraint(module)))
    throw std::invalid_argument("balanced mapping requires complete resident-only L-sched");
  return balanced!=0;
}

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

struct ScheduleStageRecord {
  std::uint32_t stage = 0;
  std::uint32_t dependency_begin = 0;
  std::uint32_t dependency_count = 0;
};

struct RuntimeVariantRecord {
  std::uint32_t seq_begin = 1;
  std::uint32_t seq_end = 65535;
  std::vector<GemmRuntimeRecord> gemms;
  std::vector<AttentionRuntimeRecord> attention;
  std::vector<DependencyRecord> dependencies;
  std::vector<ScheduleStageRecord> schedule;
  std::uint32_t max_dependency_span = 0;
  std::uint32_t ownership_flags = 0;
  bool explicit_resident_constraint = false;
  bool balanced_placement = false;
};

void BuildVariantSchedule(RuntimeVariantRecord& variant,
                          std::size_t stage_count) {
  std::vector<std::vector<int>> successors(stage_count);
  for (auto const& edge : variant.dependencies) {
    if (edge.producer >= stage_count || edge.consumer >= stage_count)
      throw std::invalid_argument("dependency names a stage outside the model");
    successors[edge.producer].push_back(static_cast<int>(edge.consumer));
  }
  solver::ListScheduler scheduler;
  std::vector<int> const order = scheduler.Schedule(successors);
  solver::ScheduleSafety const safety = scheduler.Validate(successors, order);
  variant.max_dependency_span =
      static_cast<std::uint32_t>(safety.max_dependency_span);
  variant.schedule.reserve(order.size());
  for (int stage : order) {
    auto const first = std::lower_bound(
        variant.dependencies.begin(), variant.dependencies.end(), stage,
        [](DependencyRecord const& edge, int consumer) {
          return edge.consumer < static_cast<std::uint32_t>(consumer);
        });
    auto const last = std::upper_bound(
        first, variant.dependencies.end(), stage,
        [](int consumer, DependencyRecord const& edge) {
          return static_cast<std::uint32_t>(consumer) < edge.consumer;
        });
    variant.schedule.push_back(
        {static_cast<std::uint32_t>(stage),
         static_cast<std::uint32_t>(first - variant.dependencies.begin()),
         static_cast<std::uint32_t>(last - first)});
  }
}

std::uint32_t readOwnershipFlags(mlir::ModuleOp module) {
  std::uint32_t flags = 0;
  auto add = [&](llvm::StringRef name, std::uint32_t bit) {
    if (auto value = module->getAttrOfType<mlir::BoolAttr>(name))
      if (value.getValue()) flags |= bit;
  };
  add("tilemega.rope_tile_per_block", kRoPETileOwnership);
  add("tilemega.kv_tile_per_block", kKVTileOwnership);
  add("tilemega.activation_tile_per_block", kActivationTileOwnership);
  add("tilemega.combiner_tile_per_block", kCombinerTileOwnership);
  return flags;
}

std::vector<GemmRuntimeRecord> readRuntimeGemms(mlir::ModuleOp module,
                                                 std::size_t expected) {
  auto plan = module->getAttrOfType<mlir::ArrayAttr>("tilemega.gemm_runtime");
  if (!plan)
    throw std::invalid_argument(
        "verified CG has no tilemega.gemm_runtime generator plan");
  if (plan.size() != expected)
    throw std::invalid_argument("tilemega.gemm_runtime has wrong length");
  std::vector<GemmRuntimeRecord> result;
  auto u16 = [](std::int64_t value, llvm::StringRef field) {
    if (value <= 0 || value > std::numeric_limits<std::uint16_t>::max())
      throw std::invalid_argument("invalid tilemega.gemm_runtime field " +
                                  field.str());
    return static_cast<std::uint16_t>(value);
  };
  for (auto value : plan) {
    auto item = dictionaryEntry(value, "gemm_runtime");
    GemmRuntimeRecord record;
    record.tile_m = u16(integerField(item, "tile_m"), "tile_m");
    record.tile_n = u16(integerField(item, "tile_n"), "tile_n");
    record.tile_k = u16(integerField(item, "tile_k"), "tile_k");
    record.stages = u16(integerField(item, "stages"), "stages");
    record.split_k = u16(integerField(item, "split_k"), "split_k");
    result.push_back(record);
  }
  return result;
}

std::vector<AttentionRuntimeRecord> readRuntimeAttention(mlir::ModuleOp module) {
  auto choices = module->getAttrOfType<mlir::ArrayAttr>("tilemega.attention_runtime");
  if (!choices) {
    if (module->hasAttr("tilemega.attention_runtime"))
      throw std::invalid_argument("attention runtime plan must be an array");
    return {};
  }
  auto model = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  if (!model) throw std::invalid_argument("attention runtime plan has no model");
  auto stages = arrayField(model,"stages");
  std::vector<AttentionRuntimeRecord> result(stages.size());
  std::set<long> selected;
  for (auto value : choices) {
    auto choice = dictionaryEntry(value,"attention_runtime");
    long stage = integerField(choice,"stage"), chunks = integerField(choice,"chunks");
    long extent = integerField(choice,"chunk_extent");
    if (stage < 0 || static_cast<std::size_t>(stage) >= stages.size() ||
        !selected.insert(stage).second || chunks <= 0 || extent <= 0 ||
        chunks > std::numeric_limits<int>::max() || extent > std::numeric_limits<int>::max() ||
        stringField(dictionaryEntry(stages[stage],"stages"),"kind") != "kAttention")
      throw std::invalid_argument("invalid stage or geometry in attention runtime plan");
    result[stage] = {static_cast<std::uint32_t>(chunks),static_cast<std::uint32_t>(extent)};
  }
  return result;
}

std::string emitAttentionStorage(mlir::ModuleOp module,
                                 std::vector<RuntimeVariantRecord> const& variants) {
  auto model = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  auto stages = arrayField(model,"stages");
  std::uint32_t extent = 0;
  bool direct_default = false, enabled = false;
  for (auto const& variant : variants) {
    enabled = enabled || !variant.attention.empty();
    for (std::size_t s=0; s<stages.size(); ++s) {
      if (stringField(dictionaryEntry(stages[s],"stages"),"kind") != "kAttention") continue;
      if (variant.attention.empty() || !variant.attention[s].chunk_extent) direct_default = true;
      else extent = std::max(extent,variant.attention[s].chunk_extent);
    }
  }
  if (!enabled) return {};
  std::string size = std::to_string(extent);
  if (direct_default)
    size = "((TILEMEGA_ATTENTION_MAX_TOTAL > "+size+") ? TILEMEGA_ATTENTION_MAX_TOTAL : "+size+")";
  return "#ifndef TILEMEGA_CHUNKED_ATTENTION\n#define TILEMEGA_CHUNKED_ATTENTION 1\n#endif\n"
      "#define TILEMEGA_ATTENTION_SCRATCH_EXTENT "+size+"\n";
}

std::string emitModelPlan(mlir::ModuleOp module,
                          std::vector<RuntimeVariantRecord> variants) {
  auto plan = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  if (!plan)
    throw std::invalid_argument(
        "verified CG has no tilemega.model_plan; import through the C++ frontend");
  auto buffers = arrayField(plan, "buffers");
  auto gemms = arrayField(plan, "gemms");
  auto stages = arrayField(plan, "stages");
  auto outputs = arrayField(plan, "outputs");
  std::string dtype = stringField(plan, "dtype");
  if (dtype != "f32" && dtype != "bf16")
    throw std::invalid_argument("unsupported tilemega.model_plan dtype: " + dtype);
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
  out << "};\n\n";
  if (variants.empty())
    throw std::invalid_argument("generator needs at least one runtime variant");
  std::sort(variants.begin(), variants.end(), [](auto const& a, auto const& b) {
    return a.seq_begin < b.seq_begin;
  });
  std::uint32_t cursor_seq = 1;
  for (std::size_t v = 0; v < variants.size(); ++v) {
    auto& variant = variants[v];
    if (variant.seq_begin != cursor_seq || variant.seq_end < variant.seq_begin)
      throw std::invalid_argument(
          "runtime variant intervals must be contiguous, disjoint, and start at seq=1");
    cursor_seq = variant.seq_end + 1;
    if (variant.gemms.size() != gemms.size())
      throw std::invalid_argument("runtime variant GEMM table has wrong length");
    std::stable_sort(variant.dependencies.begin(), variant.dependencies.end(),
                     [](auto const& a, auto const& b) {
                       return a.consumer < b.consumer;
                     });
    BuildVariantSchedule(variant, stages.size());
    if (!variant.attention.empty()) {
      if (variant.attention.size()!=stages.size())
        throw std::invalid_argument("attention runtime table has wrong length");
      out << "constexpr AttentionRuntimeRecord kRuntimeAttention" << v << "[] = {\n";
      for (auto const& item : variant.attention)
        out << "  {" << item.chunks << "u, " << item.chunk_extent << "u},\n";
      out << "};\n\n";
    }
    out << "constexpr GemmRuntimeDesc kRuntimeGemms" << v << "[] = {\n";
    for (auto const& impl : variant.gemms)
      out << "  {" << impl.compiled_variant << "u, " << impl.split_k << "u, "
          << impl.tile_m << "u, " << impl.tile_n << "u, " << impl.tile_k
          << "u, " << impl.stages << "u},\n";
    out << "};\n\nconstexpr StageDependency kDependencies" << v << "[] = {\n";
    for (auto const& edge : variant.dependencies) {
      analysis::WaitWindow const& w = edge.window;
      char const* kind = !w.narrowed ? "kAll"
                         : w.IsIdentity() ? "kIdentity" : "kWindow";
      out << "  {" << edge.producer << "u, " << edge.consumer
          << "u, StageDependency::Map::" << kind << ", " << w.div << "u, "
          << w.scale << ", " << w.offset << ", " << w.count << "u},\n";
    }
    if (variant.dependencies.empty())
      out << "  {0u, 0u, StageDependency::Map::kAll, 1u, 0, 0, 1u},\n";
    out << "};\n\nconstexpr std::uint32_t kDependencyOffsets" << v
        << "[] = {\n  ";
    std::size_t edge = 0;
    for (std::size_t stage = 0; stage <= stages.size(); ++stage) {
      while (edge < variant.dependencies.size() &&
             variant.dependencies[edge].consumer < stage)
        ++edge;
      out << edge << "u, ";
    }
    out << "\n};\n\n";
    out << "constexpr ScheduleStageDesc kSchedule" << v << "[] = {\n";
    for (auto const& scheduled : variant.schedule) {
      out << "  {" << scheduled.stage << "u, "
          << scheduled.dependency_begin << "u, "
          << scheduled.dependency_count << "u},\n";
    }
    out << "};\n\n";
  }
  out << "constexpr RuntimeVariantDesc kRuntimeVariants[] = {\n";
  for (std::size_t v = 0; v < variants.size(); ++v) {
    out << "  {kRuntimeGemms" << v << ", kDependencies" << v << ", "
        << variants[v].dependencies.size() << "u, kDependencyOffsets" << v
        << ", kSchedule" << v << ", " << variants[v].schedule.size()
        << "u, " << variants[v].max_dependency_span << "u, "
        << variants[v].seq_begin << "u, " << variants[v].seq_end
        << "u, " << variants[v].ownership_flags << "u";
    if (!variants[v].attention.empty()) out << ", kRuntimeAttention" << v;
    else if (variants[v].explicit_resident_constraint) out << ", nullptr";
    if (variants[v].explicit_resident_constraint) out << ", true";
    if (variants[v].balanced_placement) out << ", true";
    out << "},\n";
  }
  std::uint32_t const seq_count = variants.back().seq_end + 1;
  out << "};\n\nconstexpr auto MakeSeqVariant() {\n"
      << "  std::array<std::uint16_t, " << seq_count << "> table{};\n";
  for (std::size_t v = 0; v < variants.size(); ++v)
    out << "  for (std::uint32_t s = " << variants[v].seq_begin << "u; s <= "
        << variants[v].seq_end << "u; ++s) table[s] = " << v << "u;\n";
  out << "  return table;\n}\n"
      << "constexpr auto kSeqVariant = MakeSeqVariant();\n\n"
      << "constexpr ModelSpec kModel = {kDims, ScalarType::"
      << (dtype == "bf16" ? "kBF16" : "kF32") << ", kBuffers, "
      << buffers.size() << "u, kGemms, " << gemms.size()
      << "u, kStages, " << stages.size() << "u, kOutputs, "
      << outputs.size() << "u, kRuntimeVariants, " << variants.size()
      << "u, kSeqVariant.data(), " << seq_count
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
      "  while (::tilemega::codegen::EventPoll((ev)) < (need)) __nanosleep(64); \\\n"
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

struct VariantAnalysis {
  std::vector<DependencyRecord> dependencies;
  int cluster_dim = 1;
  std::map<std::string,std::uint32_t> task_stages;
};

VariantAnalysis AnalyzeVariantModule(mlir::ModuleOp module) {
  if (!module || mlir::failed(mlir::verify(module)))
    throw std::invalid_argument(
        "CouplingGraphToCUDA requires a verified CG ModuleOp");
  auto theta = readBinding(module, "tilemega.theta");
  auto granularity = readBinding(module, "tilemega.g");
  analysis::ParamBinding known = theta;
  for (auto const& [name, value] : granularity.values) known.Bind(name, value);
  int max_stage = -1;
  std::unordered_map<std::string, std::uint32_t> task_stages;
  std::size_t tasks = 0;
  for (auto task : module.getOps<dialect::TaskSpaceOp>()) {
    ++tasks;
    max_stage = std::max(max_stage, static_cast<int>(task.getStage()));
    task_stages.emplace(task.getSymName().str(), task.getStage());
  }
  if (max_stage < 0) throw std::invalid_argument("CG has no task spaces");

  std::map<std::pair<std::uint32_t, std::uint32_t>, analysis::WaitWindow> pairs;
  std::size_t couplings = 0, cluster_edges = 0;
  for (auto coupling : module.getOps<dialect::CouplingOp>()) {
    ++couplings;
    llvm::StringRef sync = coupling.getSyncKind().getValue().getValue();
    if (sync == "cluster") ++cluster_edges;
    else if (sync != "global")
      throw std::invalid_argument("generator accepts global or cluster synchronization");
    (void)coupling.getWait().getValue().Eval(known);
    (void)coupling.getFanout().getValue().Eval(known);
    (void)coupling.getVolume().getValue().Eval(known);
    (void)coupling.getCount().getValue().Eval(known);
    (void)coupling.getRelation().getMap();
    auto source = task_stages.find(coupling.getSrc().str());
    auto target = task_stages.find(coupling.getDst().str());
    if (source == task_stages.end() || target == task_stages.end())
      throw std::invalid_argument("coupling names an unknown task space");
    if (source->second >= target->second) continue;
    analysis::WaitWindow window;
    if (auto text = coupling.getWaitMap())
      window = analysis::ParseWaitWindow(text->str());
    auto pair = std::make_pair(source->second, target->second);
    auto [at, fresh] = pairs.emplace(pair, window);
    if (!fresh && at->second != window) at->second = analysis::WaitWindow{};
  }

  VariantAnalysis result;
  result.task_stages.insert(task_stages.begin(),task_stages.end());
  std::size_t placements = 0;
  for (auto placement : module.getOps<dialect::PlacementOp>()) {
    ++placements;
    int cluster = static_cast<int>(placement.getCluster());
    if (placements == 1) result.cluster_dim = cluster;
    else if (cluster != result.cluster_dim)
      throw std::invalid_argument(
          "every placement must name the same cluster dimension");
  }
  if (placements != tasks)
    throw std::invalid_argument("every task space must have exactly one placement");
  if (result.cluster_dim > 1 && cluster_edges != couplings)
    throw std::invalid_argument(
        "a clustered placement requires every coupling to synchronize at cluster scope");
  if (result.cluster_dim == 1 && cluster_edges != 0)
    throw std::invalid_argument(
        "cluster synchronization requires a placement cluster larger than 1");
  for (auto const& edge : TransitiveReduction(pairs, max_stage + 1))
    result.dependencies.push_back(
        {edge.first, edge.second, pairs.at(edge)});
  return result;
}

}  // namespace

std::vector<AttentionRuntimeRecord> ReadAttentionRuntime(mlir::ModuleOp module) {
  return readRuntimeAttention(module);
}

RuntimePlan ReadRuntimePlan(mlir::ModuleOp module) {
  analysis::IslReferenceAudit audit(__func__);
  auto analysis = AnalyzeVariantModule(module);
  auto model = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  if (!model) throw std::invalid_argument("CG has no runtime model plan");
  RuntimePlan result;
  (void)readResidentConstraint(module);
  result.dependencies = std::move(analysis.dependencies);
  result.gemms = readRuntimeGemms(module, arrayField(model, "gemms").size());
  result.attention = readRuntimeAttention(module);
  result.ownership_flags = readOwnershipFlags(module);
  result.cluster_dim = analysis.cluster_dim;
  result.task_stages = std::move(analysis.task_stages);
  if (auto domains = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.param_domain")) {
    std::unordered_map<std::string,std::string> text;
    for (auto domain : domains) {
      auto value = llvm::dyn_cast<mlir::StringAttr>(domain.getValue());
      if (!value) throw std::invalid_argument("malformed CG parameter range");
      text.emplace(domain.getName().str(),value.getValue().str());
    }
    auto shape = frontend::SymbolicShapeBridge{}.Parse(text,{},{});
    for (auto const& [name,range] : shape.ranges)
      result.parameter_ranges.emplace(name,std::make_pair(range.minimum,range.maximum));
  }
  return result;
}

namespace {
std::string EmitGemmInstantiations(
    std::vector<std::tuple<int, int, int, int>> const& shapes) {
  if (shapes.empty() || shapes.size() > 16)
    throw std::invalid_argument(
        "one generated binary must contain between 1 and 16 GEMM shapes");
  std::ostringstream out;
  auto emit = [&](std::size_t index, std::tuple<int, int, int, int> shape) {
    auto [m, n, k, stages] = shape;
    std::string prefix = index == 0 ? "TILEMEGA_GEMM_"
                                    : "TILEMEGA_GEMM_V" +
                                          std::to_string(index) + "_";
    out << "#define " << prefix << "TILE_M " << m << "\n"
        << "#define " << prefix << "TILE_N " << n << "\n"
        << "#define " << prefix << "TILE_K " << k << "\n"
        << "#define " << prefix << "STAGES " << stages << "\n";
  };
  for (std::size_t i = 0; i < shapes.size(); ++i) emit(i, shapes[i]);
  out << "#define TILEMEGA_GEMM_VARIANT_COUNT " << shapes.size() << "\n";
  return out.str();
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

  // The template list is emitted by the generator. ModelSpec below carries
  // the same values and the harness verifies the two views before launch.
  std::string generatedGemm;
  if (auto g = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.g")) {
    auto extent = [&](llvm::StringRef name) -> long {
      auto value = llvm::dyn_cast_or_null<mlir::IntegerAttr>(g.get(name));
      return value ? value.getInt() : 0;
    };
    long const tm = extent("Tm"), tn = extent("Tn");
    if (tm > 0 && tn > 0)
      generatedGemm = "#define TILEMEGA_GEMM_TILE_M " +
                      std::to_string(tm) + "\n" +
                      "#define TILEMEGA_GEMM_TILE_N " +
                      std::to_string(tn) + "\n" +
                      "#define TILEMEGA_GEMM_TILE_K 16\n"
                      "#define TILEMEGA_GEMM_STAGES 3\n"
                      "#define TILEMEGA_GEMM_VARIANT_COUNT 1\n";
  }

  std::ostringstream out;
  auto emittedPlan = module->getAttrOfType<mlir::DictionaryAttr>(
      "tilemega.model_plan");
  if (!emittedPlan)
    throw std::invalid_argument("verified CG has no tilemega.model_plan");
  RuntimeVariantRecord attention_storage;
  attention_storage.attention = readRuntimeAttention(module);
  out << "// SPDX-License-Identifier: BSD-3-Clause\n"
      << "// Generated by CouplingGraphToCUDA from verified tilemega.* ops.\n"
      << (stringField(emittedPlan, "dtype") == "bf16"
              ? "#define TILEMEGA_MODEL_BF16 1\n" : std::string())
      << (clusterDim > 1 ? "#define TILEMEGA_GENERATED_CLUSTER_DIM " +
                               std::to_string(clusterDim) + "\n"
                          : std::string())
      << generatedGemm
      << emitAttentionStorage(module, {attention_storage})
      << SyncEmitter{}.EmitWait("global")
      << SyncEmitter{}.EmitSignal("global")
      << HostLauncherEmitter{}.Emit("l1_kernel")
      << TaskBodyEmitter{}.Emit(module) << "\n";
  std::vector<DependencyRecord> dependencies;
  for (auto const& edge : TransitiveReduction(dependencyPairs, maxStage + 1))
    dependencies.push_back({edge.first, edge.second, dependencyPairs.at(edge)});
  auto modelPlan = emittedPlan;
  RuntimeVariantRecord runtime;
  runtime.gemms = readRuntimeGemms(module,
                                   arrayField(modelPlan, "gemms").size());
  runtime.dependencies = std::move(dependencies);
  runtime.ownership_flags = readOwnershipFlags(module);
  runtime.explicit_resident_constraint = readResidentConstraint(module);
  runtime.balanced_placement = readBalancedPlacement(module);
  runtime.attention = std::move(attention_storage.attention);
  out << emitModelPlan(module, {std::move(runtime)});
  return out.str();
}

std::string CouplingGraphToCUDA::LowerVariants(
    std::vector<RuntimeVariantModule> const& variants) const {
  if (variants.empty())
    throw std::invalid_argument("LowerVariants requires at least one variant");
  mlir::ModuleOp first = variants.front().module;
  auto first_plan = first->getAttrOfType<mlir::DictionaryAttr>(
      "tilemega.model_plan");
  if (!first_plan)
    throw std::invalid_argument("variant has no tilemega.model_plan");
  std::size_t const gemm_count = arrayField(first_plan, "gemms").size();

  std::vector<RuntimeVariantRecord> records;
  std::vector<std::tuple<int, int, int, int>> shapes;
  int cluster_dim = -1;
  for (auto const& input : variants) {
    auto plan = input.module->getAttrOfType<mlir::DictionaryAttr>(
        "tilemega.model_plan");
    if (!plan || plan != first_plan)
      throw std::invalid_argument(
          "runtime variants must have byte-identical tilemega.model_plan data");
    RuntimePlan runtime_plan = ReadRuntimePlan(input.module);
    if (cluster_dim < 0) cluster_dim = runtime_plan.cluster_dim;
    else if (cluster_dim != runtime_plan.cluster_dim)
      throw std::invalid_argument(
          "runtime variants cannot change the kernel cluster dimension");
    RuntimeVariantRecord record;
    record.seq_begin = input.seq_begin;
    record.seq_end = input.seq_end;
    record.ownership_flags = runtime_plan.ownership_flags;
    record.explicit_resident_constraint = readResidentConstraint(input.module);
    record.balanced_placement = readBalancedPlacement(input.module);
    record.dependencies = std::move(runtime_plan.dependencies);
    record.gemms = std::move(runtime_plan.gemms);
    record.attention = std::move(runtime_plan.attention);
    if (record.gemms.size() != gemm_count)
      throw std::invalid_argument("runtime variant GEMM count changed");
    for (auto& impl : record.gemms) {
      auto key = std::make_tuple(static_cast<int>(impl.tile_m),
                                 static_cast<int>(impl.tile_n),
                                 static_cast<int>(impl.tile_k),
                                 static_cast<int>(impl.stages));
      auto found = std::find(shapes.begin(), shapes.end(), key);
      if (found == shapes.end()) {
        if (shapes.size() == 16)
          throw std::invalid_argument(
              "runtime plans require more than 16 unique GEMM shapes");
        shapes.push_back(key);
        found = std::prev(shapes.end());
      }
      impl.compiled_variant = static_cast<std::uint16_t>(
          std::distance(shapes.begin(), found));
    }
    records.push_back(std::move(record));
  }

  std::ostringstream out;
  out << "// SPDX-License-Identifier: BSD-3-Clause\n"
      << "// Generated by CouplingGraphToCUDA from verified tilemega.* ops.\n"
      << (stringField(first_plan, "dtype") == "bf16"
              ? "#define TILEMEGA_MODEL_BF16 1\n" : std::string())
      << EmitGemmInstantiations(shapes)
      << emitAttentionStorage(first, records)
      << (cluster_dim > 1 ? "#define TILEMEGA_GENERATED_CLUSTER_DIM " +
                                std::to_string(cluster_dim) + "\n"
                          : std::string())
      << SyncEmitter{}.EmitWait("global")
      << SyncEmitter{}.EmitSignal("global")
      << HostLauncherEmitter{}.Emit("l1_kernel")
      << TaskBodyEmitter{}.Emit(first) << "\n"
      << emitModelPlan(first, std::move(records));
  return out.str();
}

}  // namespace tilemega::codegen
