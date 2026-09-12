// SPDX-License-Identifier: BSD-3-Clause
// EX-S1 driver: evaluate every candidate Plan of a cell through
// `SimulateExecution` and write the predicted numbers as data.
//
// Every input is the runtime's own, not a re-derivation of it:
//
//   * the task DAG is `MaterializeRuntimeTaskGraph` over the `kDependencies`
//     table parsed out of the generated .cu -- literally the array the harness
//     feeds it (ModelHarness.cuh:1434);
//   * the per-stage active task counts and the stage order come from the
//     `E2E_PLACE_BASE` lines a mode 5 run dumps, so they are the counts the
//     kernel actually launched with rather than this tool's idea of them;
//   * (pi, sigma) comes from `MaterializePlanPlacement`, the one routine the
//     host also calls, and the re-materialized modes 0/4/5 are checked against
//     the `E2E_PLACE_STATS` line of their own run.  If `same_worker_edges`,
//     `cross_worker_edges` and `max_queue` all reproduce, then the DAG and the
//     placement under simulation are provably the ones that ran, and a new
//     candidate is materialized on the same footing.
//
// Node durations are `CostModel::TaskInstanceNs` at `active_ctas_per_sm = 1`,
// i.e. the solo duration of one task under the residency the kernel compiled
// to.  Concurrency is then applied by the simulator's co-residency model rather
// than baked into the price, which is the point: a task that happens to run
// alone on its SM should not be charged for a co-resident that is idle.
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/ExecutionSimulator.h>
#include <tilemega/Solver/PlanMaterialize.h>
#include <tilemega/Solver/TaskModel.h>
#include <mlir/IR/MLIRContext.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace tilemega;
using namespace tilemega::solver;

std::vector<std::string> Split(std::string const& line, char separator) {
  std::istringstream input(line);
  std::string field;
  std::vector<std::string> result;
  while (std::getline(input, field, separator)) result.push_back(field);
  return result;
}

/// `key=value` tokens of one `E2E_*` line, whitespace separated.
std::map<std::string, std::string> Fields(std::string const& line) {
  std::map<std::string, std::string> result;
  std::istringstream input(line);
  std::string token;
  while (input >> token) {
    auto equals = token.find('=');
    if (equals != std::string::npos)
      result[token.substr(0, equals)] = token.substr(equals + 1);
  }
  return result;
}

long Field(std::map<std::string, std::string> const& fields, std::string const& key) {
  auto found = fields.find(key);
  if (found == fields.end()) throw std::runtime_error("missing " + key + " in dump line");
  return std::stol(found->second);
}

struct CellDump {
  std::vector<int> counts;                 ///< by stage
  std::vector<std::uint32_t> stage_order;  ///< by position
  int grid = 0, ctas_per_sm = 0, num_sms = 0;
};

/// The mode 5 run's own dump: `E2E_PLACE_BASE` covers every stage in order,
/// `E2E_RESOURCE` carries the launch geometry.
CellDump ReadCellDump(std::string const& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read dump " + path);
  CellDump cell;
  std::map<int, int> counts;
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("E2E_PLACE_BASE", 0) == 0) {
      auto fields = Fields(line);
      int position = int(Field(fields, "pos"));
      int stage = int(Field(fields, "stage"));
      if (int(cell.stage_order.size()) <= position) cell.stage_order.resize(position + 1);
      cell.stage_order[position] = std::uint32_t(stage);
      counts[stage] = int(Field(fields, "active"));
    } else if (line.rfind("E2E_RESOURCE", 0) == 0) {
      auto fields = Fields(line);
      cell.grid = int(Field(fields, "grid"));
      cell.ctas_per_sm = int(Field(fields, "ctas_per_sm"));
      cell.num_sms = int(Field(fields, "num_sms"));
    }
  }
  if (cell.stage_order.empty())
    throw std::runtime_error("no E2E_PLACE_BASE lines in " + path +
                             " (a mode 5 run with TILEMEGA_PLACEMENT_BASE_DUMP=1 is required)");
  if (cell.grid <= 0) throw std::runtime_error("no E2E_RESOURCE line in " + path);
  cell.counts.assign(counts.size(), 0);
  for (auto const& [stage, count] : counts) {
    if (stage < 0 || stage >= int(cell.counts.size()))
      throw std::runtime_error("E2E_PLACE_BASE stages are not dense in " + path);
    cell.counts[stage] = count;
  }
  return cell;
}

struct PlaceStats {
  long max_queue = -1, same_worker_edges = -1, cross_worker_edges = -1;
  bool present = false;
};

PlaceStats ReadPlaceStats(std::string const& path) {
  std::ifstream input(path);
  PlaceStats stats;
  if (!input) return stats;
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("E2E_PLACE_STATS", 0) != 0) continue;
    auto fields = Fields(line);
    stats.max_queue = Field(fields, "max_queue");
    stats.same_worker_edges = Field(fields, "same_worker_edges");
    stats.cross_worker_edges = Field(fields, "cross_worker_edges");
    stats.present = true;
  }
  return stats;
}

/// The `kDependencies<variant>` array of a generated .cu, in the shape
/// `MaterializeRuntimeTaskGraph` consumes.  Parsed rather than recomputed: this
/// table is the DAG the megakernel enforces, and any reconstruction of it would
/// be the thing under test instead of the input.
std::vector<codegen::RuntimeDependencyWindow> ReadDependencies(std::string const& path,
                                                               int variant) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read generated source " + path);
  std::string const opening =
      "constexpr StageDependency kDependencies" + std::to_string(variant) + "[] = {";
  std::string line;
  bool inside = false;
  std::vector<codegen::RuntimeDependencyWindow> windows;
  while (std::getline(input, line)) {
    if (!inside) {
      inside = line.rfind(opening, 0) == 0;
      continue;
    }
    if (line.rfind("};", 0) == 0) return windows;
    std::string cleaned;
    for (char character : line)
      cleaned += (character == '{' || character == '}' || character == 'u') ? ' ' : character;
    auto tokens = Split(cleaned, ',');
    if (tokens.size() < 7) throw std::runtime_error("malformed StageDependency row: " + line);
    auto map = tokens[2];
    bool all = map.find("kAll") != std::string::npos;
    if (!all && map.find("kWindow") == std::string::npos && map.find("kIdentity") == std::string::npos)
      throw std::runtime_error("unknown StageDependency::Map in " + line);
    windows.push_back({std::stoi(tokens[0]), std::stoi(tokens[1]), all, std::stol(tokens[3]),
                       std::stol(tokens[4]), std::stol(tokens[5]), std::stol(tokens[6])});
  }
  throw std::runtime_error("no " + opening + " in " + path);
}

/// Worker -> SM as the hardware actually chose it, read off a trace v2 dump.
/// A worker appears on exactly one SM for the life of a persistent launch, so
/// the first row for a worker settles it; a disagreement is a hard error rather
/// than a majority vote, because it would mean the launch was not persistent.
std::vector<int> ReadWorkerSm(std::string const& path, int grid) {
  std::ifstream input(path);
  if (!input) return {};
  std::string header;
  if (!std::getline(input, header)) return {};
  auto columns = Split(header, '\t');
  auto column = [&](char const* name) {
    for (std::size_t i = 0; i < columns.size(); ++i)
      if (columns[i] == name) return int(i);
    throw std::runtime_error(std::string("trace dump has no ") + name + " column: " + path);
  };
  int worker_column = column("worker"), sm_column = column("smid");
  std::vector<int> worker_sm(grid, -1);
  std::string line;
  while (std::getline(input, line)) {
    auto row = Split(line, '\t');
    if (int(row.size()) <= std::max(worker_column, sm_column)) continue;
    int worker = std::stoi(row[worker_column]), sm = std::stoi(row[sm_column]);
    if (worker < 0 || worker >= grid) throw std::runtime_error("trace worker out of grid: " + line);
    if (worker_sm[worker] == -1) worker_sm[worker] = sm;
    else if (worker_sm[worker] != sm)
      throw std::runtime_error("trace shows worker " + std::to_string(worker) +
                               " on two SMs; the launch was not persistent");
  }
  for (int worker = 0; worker < grid; ++worker)
    if (worker_sm[worker] == -1) return {};  // an unobserved worker: fall back to w % sms
  return worker_sm;
}

struct StagePrice {
  double solo_ns = 0.0;
  bool from_instance = false;
  std::string op;
};

/// One task of `stage`, alone on its SM, under the compiled residency.
std::vector<StagePrice> PriceStages(CostModel const& cost, ModelDescription& model,
                                    analysis::OperatorGraph const& graph,
                                    std::vector<GemmConfig> const& configs, Residency residency,
                                    int threads, std::size_t stages) {
  std::vector<StagePrice> prices(stages);
  std::map<int, ModelTaskSemantics const*> semantics;
  for (auto const& semantic : model.task_semantics)
    if (semantic.stage >= 0 && semantic.stage < int(stages)) semantics.emplace(semantic.stage, &semantic);
  for (std::size_t stage = 0; stage < stages; ++stage) {
    auto const& description = model.stages.at(stage);
    auto found = semantics.find(int(stage));
    if (found == semantics.end()) {
      // No CG semantic for this runtime stage: fall back to the stage closed
      // form divided by its wave count.  Reported per stage so a cell that
      // leans on the fallback cannot be mistaken for one that does not.
      auto const& config = configs.at(description.gemm >= 0 ? description.gemm : 0);
      prices[stage] = {cost.TaskStageNs(model, int(stage), config, residency), false, "-"};
      continue;
    }
    auto const& semantic = *found->second;
    analysis::ParamBinding point;
    if (description.kind == StageKind::kGemm) {
      auto const& config = configs.at(description.gemm);
      auto input = DeriveModelTaskInput(model, semantic, graph, &config);
      for (auto const& coordinate : input.cost_coordinates) point.Bind(coordinate, 0);
      int chunks = 0;
      cost.GemmStageNs(model.gemms.at(description.gemm), config, residency, model, &chunks);
      auto traits = model.dtype == ScalarType::kBF16
                        ? TensorBF16Traits(config.tile_m, config.tile_n, config.tile_k, config.stages)
                        : SimtF32Traits(config.tile_m, config.tile_n, config.tile_k, config.stages);
      prices[stage] = {cost.TaskInstanceNs(input, traits, residency, model, chunks, point, 1.0),
                       true, semantic.op.name};
    } else {
      auto input = DeriveModelTaskInput(model, semantic, graph, nullptr);
      for (auto const& coordinate : input.cost_coordinates) point.Bind(coordinate, 0);
      BackendTraits traits;
      traits.threads = threads;
      traits.smem_bytes = sizeof(float) * codegen::SimtSharedElements(
          static_cast<codegen::TaskKind>(description.kind), threads, TILEMEGA_ATTENTION_MAX_TOTAL);
      prices[stage] = {cost.TaskInstanceNs(input, traits, residency, model, 1, point, 1.0), true,
                       semantic.op.name};
    }
  }
  return prices;
}

struct Candidate {
  char const* name;
  dialect::PlacementMode mode;
  std::vector<std::int64_t> params;
};

/// §6.2's six.  `eft`, `band` and `wavefront` are listed before EX-S2 fills
/// them in, so the driver records `unimplemented` for them rather than hiding a
/// candidate the report has to account for.
std::vector<Candidate> Candidates() {
  return {{"legacy_grid_stride", dialect::PlacementMode::kLegacyGridStride, {}},
          {"rotate", dialect::PlacementMode::kRotate, {}},
          {"balanced", dialect::PlacementMode::kBalanced, {}},
          {"eft", dialect::PlacementMode::kEft, {}},
          {"band", dialect::PlacementMode::kTemplate, {long(dialect::PlacementTemplate::kBand)}},
          {"wavefront", dialect::PlacementMode::kTemplate,
           {long(dialect::PlacementTemplate::kWavefront)}}};
}

int ModeOfCandidate(Candidate const& candidate) {
  // The TILEMEGA_PLACEMENT value whose dump this candidate must reproduce, or
  // -1 when no measured run of it exists yet.
  switch (candidate.mode) {
    case dialect::PlacementMode::kLegacyGridStride: return 0;
    case dialect::PlacementMode::kBalanced: return 4;
    case dialect::PlacementMode::kRotate: return 5;
    default: return -1;
  }
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 4)
    throw std::invalid_argument(
        "usage: tilemega-simulate REPO MANIFEST.tsv OUT_DIR\n"
        "manifest columns: model seq past generated_cu out_prefix trace_dir [export_json]\n"
        "  out_prefix   cell dumps are read as ${out_prefix}_p<mode>_s<seq>.out\n"
        "  trace_dir    '-' to map worker -> SM as w % sms instead of from a trace\n"
        "  export_json  '-' or absent for SEQSCAN/raw/export/<model>.json");
  analysis::IslContext isl;
  std::string repo = argv[1], manifest_path = argv[2], out = argv[3];

  HopCurve hop;
  std::string error;
  if (!HopCurve::FromTsv(repo + "/docs/experiments/SIMULATOR/hop_ns.tsv", &hop, &error))
    throw std::runtime_error("hop curve: " + error);

  auto target = TargetSpec::FromJson(repo + "/configs/targets/sm_89.json");
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();

  std::ifstream manifest(manifest_path);
  if (!manifest) throw std::runtime_error("cannot read manifest " + manifest_path);
  std::string line;
  if (!std::getline(manifest, line)) throw std::runtime_error("empty manifest");

  std::ofstream predicted(out + "/predicted.tsv");
  predicted << std::setprecision(10)
            << "model\tseq\tpast\tcandidate\tarm\tstatus\tmakespan_ns\tbusiest_worker_ns\t"
               "busiest_worker\ttotal_block_ns\tsolo_work_ns\ttotal_work_ns\tcritical_path_ns\t"
               "cross_worker_edges\tsame_worker_edges\tmax_queue\teval_us\n";
  std::ofstream checks(out + "/materialize_check.tsv");
  checks << "model\tseq\tmode\tcandidate\tfield\texpected\tactual\tstatus\n";
  std::ofstream stage_price(out + "/stage_price.tsv");
  stage_price << std::setprecision(10) << "model\tseq\tstage\tkind\top\tcount\tsolo_ns\tsource\n";

  int mismatches = 0, cells = 0;
  while (std::getline(manifest, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto row = Split(line, '\t');
    if (row.size() < 6) throw std::runtime_error("manifest row needs 6 columns: " + line);
    // A 7th column names the torch export.  Real width does not live under
    // SEQSCAN/raw/export, and a cost model whose speed gate is only ever
    // checked on a two-layer model has not been checked.
    std::string name = row[0];
    int seq = std::stoi(row[1]), past = std::stoi(row[2]);
    std::string generated = row[3], prefix = row[4], trace = row[5];
    std::string export_json = row.size() > 6 && row[6] != "-"
                                  ? row[6]
                                  : repo + "/docs/experiments/SEQSCAN/raw/export/" + name + ".json";

    auto cell = ReadCellDump(prefix + "_p5_s" + std::to_string(seq) + ".out");
    auto windows = ReadDependencies(generated, 0);
    auto graph = codegen::MaterializeRuntimeTaskGraph(cell.counts, windows, cell.grid);
    std::vector<int> node_stage(graph.stage_offsets.back(), 0);
    for (std::size_t stage = 0; stage + 1 < graph.stage_offsets.size(); ++stage)
      for (int node = graph.stage_offsets[stage]; node < graph.stage_offsets[stage + 1]; ++node)
        node_stage[node] = int(stage);

    frontend::ImportOptions options;
    options.rope_tile_per_block = options.kv_tile_per_block =
        options.activation_tile_per_block = options.combiner_tile_per_block = true;
    auto module = frontend::TorchExportImporter{}.Import(
        export_json, context, nullptr, options);
    auto runtime_plan = codegen::ReadRuntimePlan(*module);
    std::vector<GemmConfig> configs;
    for (auto const& gemm : runtime_plan.gemms)
      configs.push_back({int(gemm.tile_m), int(gemm.tile_n), int(gemm.tile_k), int(gemm.stages),
                         int(gemm.split_k)});
    auto model = ModelDescription::FromCouplingGraph(*module, {seq, past, seq + past}, name);
    if (model.stages.size() != cell.counts.size())
      throw std::runtime_error("the CG model has " + std::to_string(model.stages.size()) +
                               " stages but the dump reports " + std::to_string(cell.counts.size()));
    auto operators = InstantiateModelTasks(model, configs);
    int threads = model.dtype == ScalarType::kBF16 ? kTensorBF16Threads : kSimtF32Threads;
    CostModelOptions prices;
    prices.unified_task_cost = true;
    CostModel cost(target, model.dtype, prices);
    Residency residency{cell.ctas_per_sm};
    auto stage_prices =
        PriceStages(cost, model, operators, configs, residency, threads, cell.counts.size());

    std::vector<double> task_ns(graph.stage_offsets.back(), 0.0);
    for (std::size_t stage = 0; stage < cell.counts.size(); ++stage) {
      stage_price << name << '\t' << seq << '\t' << stage << '\t'
                  << int(model.stages.at(stage).kind) << '\t' << stage_prices[stage].op << '\t'
                  << cell.counts[stage] << '\t' << stage_prices[stage].solo_ns << '\t'
                  << (stage_prices[stage].from_instance ? "instance" : "stage_fallback") << '\n';
      for (int task = 0; task < cell.counts[stage]; ++task)
        task_ns[graph.stage_offsets[stage] + task] = stage_prices[stage].solo_ns;
    }

    SimulatorInput input;
    input.graph = &graph;
    input.task_ns = task_ns;
    if (trace != "-") input.worker_sm = ReadWorkerSm(trace + "/slots.tsv", cell.grid);

    PlanRequest request;
    request.grid = cell.grid;
    request.counts = cell.counts;
    request.stage_order = cell.stage_order;
    request.physical_worker.resize(cell.grid);
    std::iota(request.physical_worker.begin(), request.physical_worker.end(), 0);
    request.graph = &graph;

    for (auto const& candidate : Candidates()) {
      request.mode = candidate.mode;
      request.params = candidate.params;
      MaterializedPlan plan;
      if (!MaterializePlanPlacement(request, &plan, &error)) {
        predicted << name << '\t' << seq << '\t' << past << '\t' << candidate.name
                  << "\t-\tunimplemented\t\t\t\t\t\t\t\t\t\t\n";
        std::cerr << "SIMULATE_SKIP model=" << name << " seq=" << seq
                  << " candidate=" << candidate.name << " reason=\"" << error << "\"\n";
        continue;
      }
      if (!CheckPlanLegality(graph, plan, &error))
        throw std::runtime_error(std::string("candidate ") + candidate.name +
                                 " is illegal under §5.7.3: " + error);

      long max_queue = 0, same_worker = 0;
      for (auto const& queue : plan.queue) max_queue = std::max(max_queue, long(queue.size()));
      // Counted exactly as ModelHarness.cuh:1496 counts it, over every node of
      // the graph rather than every active task, so the two numbers are
      // comparable without an argument about which edges are in scope.
      auto owner_of = [&](int node) {
        int stage = node_stage[node];
        return plan.owner[stage][node - graph.stage_offsets[stage]];
      };
      for (std::size_t producer = 0; producer < graph.successors.size(); ++producer)
        for (int consumer : graph.successors[producer])
          if (owner_of(int(producer)) == owner_of(consumer)) ++same_worker;

      int mode = ModeOfCandidate(candidate);
      if (mode >= 0) {
        auto measured = ReadPlaceStats(prefix + "_p" + std::to_string(mode) + "_s" +
                                       std::to_string(seq) + ".out");
        if (measured.present) {
          auto record = [&](char const* field, long expected, long actual) {
            bool ok = expected == actual;
            mismatches += ok ? 0 : 1;
            checks << name << '\t' << seq << '\t' << mode << '\t' << candidate.name << '\t' << field
                   << '\t' << expected << '\t' << actual << '\t' << (ok ? "MATCH" : "MISMATCH")
                   << '\n';
          };
          record("max_queue", measured.max_queue, max_queue);
          record("same_worker_edges", measured.same_worker_edges, same_worker);
          // The dump counts every DAG edge once, so the cross count is implied.
          long total_edges = 0;
          for (auto const& successors : graph.successors) total_edges += long(successors.size());
          record("cross_worker_edges", measured.cross_worker_edges, total_edges - same_worker);
        }
      }

      struct Arm {
        char const* name;
        bool proportional, flat;
      };
      for (Arm arm : {Arm{"proportional", true, false}, Arm{"flat_hop", true, true}}) {
        SimulatorOptions simulator;
        simulator.sms = cell.num_sms;
        simulator.ctas_per_sm = cell.ctas_per_sm;
        simulator.proportional_sharing = arm.proportional;
        simulator.flat_hop = arm.flat;
        SimulatorResult result;
        auto begin = std::chrono::steady_clock::now();
        bool ok = SimulateExecution(input, plan, simulator, hop, &result, &error);
        auto elapsed = std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - begin)
                           .count();
        if (!ok)
          throw std::runtime_error(std::string("simulating ") + candidate.name + " (" + arm.name +
                                   "): " + error);
        predicted << name << '\t' << seq << '\t' << past << '\t' << candidate.name << '\t'
                  << arm.name << "\tok\t" << result.makespan_ns << '\t' << result.busiest_worker_ns
                  << '\t' << result.busiest_worker << '\t' << result.total_block_ns << '\t'
                  << result.solo_work_ns << '\t' << result.total_work_ns << '\t'
                  << result.critical_path_ns << '\t' << result.cross_worker_edges << '\t'
                  << result.same_worker_edges << '\t' << max_queue << '\t' << elapsed << '\n';
        if (!arm.flat && mode >= 0) {
          std::ofstream tasks(out + "/tasks_" + name + "_s" + std::to_string(seq) + "_p" +
                              std::to_string(mode) + ".tsv");
          tasks << std::setprecision(10) << "stage\tlogical\tworker\tstart_ns\tend_ns\tblock_ns\tstretch\n";
          for (std::size_t stage = 0; stage < cell.counts.size(); ++stage)
            for (int task = 0; task < cell.counts[stage]; ++task) {
              auto const& entry = result.tasks.at(graph.stage_offsets[stage] + task);
              tasks << stage << '\t' << task << '\t' << entry.worker << '\t' << entry.start_ns
                    << '\t' << entry.end_ns << '\t' << entry.block_ns << '\t' << entry.stretch
                    << '\n';
            }
        }
      }
    }
    ++cells;
    std::cerr << "SIMULATE_CELL model=" << name << " seq=" << seq << " past=" << past
              << " stages=" << cell.counts.size() << " nodes=" << graph.stage_offsets.back()
              << " grid=" << cell.grid << " ctas_per_sm=" << cell.ctas_per_sm
              << " worker_sm=" << (input.worker_sm.empty() ? "modulo" : "traced") << '\n';
  }

  std::cerr << "SIMULATE_SUMMARY cells=" << cells << " stat_mismatches=" << mismatches << '\n';
  if (mismatches) {
    std::cerr << "the re-materialized placement disagrees with the measured dump; "
                 "the simulated DAG is not the one that ran\n";
    return 1;
  }
  if (isl.ReferenceCount() != 0) throw std::runtime_error("simulate retained isl objects");
  return 0;
} catch (std::exception const& error) {
  std::cerr << "tilemega-simulate: " << error.what() << '\n';
  return 2;
}
