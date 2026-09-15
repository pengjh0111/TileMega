// SPDX-License-Identifier: BSD-3-Clause
// EX-S2c driver: run the critical-chain scheduler at a bound theta, write the
// resulting (pi, sigma) into the Coupling Graph as a materialized Plan, and
// emit the generated source that carries it to the GPU.
//
// This is `place_eft.cpp` with a different scheduler and four more metrics.  It
// deliberately shares that tool's import options, variant interval and codegen
// path -- run.sh diffs the `legacy_grid_stride` source this tool emits against
// the committed control exactly as PLACE_EFT does, so a `chain` arm differs
// from the control only in the Plan (H2).
//
// Why a separate tool rather than a seventh mode in `place_eft.cpp`: the six
// candidates in `SIMULATOR/cell_inputs.h` are shared with EX-S1 so a cell is
// never priced two ways, and that header is a pre-existing file this round's
// scope fence (H1) freezes.  The chain candidate is therefore appended here.
//
// §6.2 asks for node weights from both sources.  The emitted plan is priced by
// the cost model, because that is the only basis available for every cell in
// the manifest -- the real-width cells have no trace dump (`trace_dir` is `-`),
// and pricing them a second way is the thing `cell_inputs.h` exists to prevent.
// The trace-weighted schedule is computed anyway and reported beside it in
// `chain_weights.tsv`, which is where §6.2's "report the difference" lands.
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/ChainPlacement.h>
#include <tilemega/Solver/EftPlacement.h>
#include <tilemega/Solver/ExecutionSimulator.h>
#include <tilemega/Solver/PlanMaterialize.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include "../SIMULATOR/cell_inputs.h"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace tilemega;
using namespace tilemega::solver;
using namespace tilemega::experiments;

frontend::ImportOptions PlanOptions() {
  frontend::ImportOptions options;
  options.rope_tile_per_block = true;
  options.kv_tile_per_block = true;
  options.activation_tile_per_block = true;
  options.combiner_tile_per_block = true;
  return options;
}

void AttachPlan(mlir::ModuleOp module, Candidate const& candidate,
                dialect::PlacementTable const* table) {
  mlir::OpBuilder builder(module.getContext());
  llvm::SmallVector<std::int64_t> params(candidate.params.begin(), candidate.params.end());
  for (auto placement : module.getOps<dialect::PlacementOp>()) {
    placement->setAttr("mode", builder.getStringAttr(dialect::PlacementModeName(candidate.mode)));
    placement->setAttr("params", builder.getDenseI64ArrayAttr(params));
    placement->setAttr("window",
                       builder.getI64IntegerAttr(dialect::kPlacementWindowImplemented));
    placement->setAttr("policy", builder.getStringAttr(dialect::kPlacementPolicyAot));
    placement->setAttr("resident_only", builder.getBoolAttr(true));
  }
  if (table) {
    llvm::SmallVector<std::int64_t> worker(table->worker.begin(), table->worker.end());
    llvm::SmallVector<std::int64_t> slot(table->slot.begin(), table->slot.end());
    module->setAttr(dialect::kPlacementTableAttr, builder.getDictionaryAttr({
        builder.getNamedAttr("worker", builder.getDenseI64ArrayAttr(worker)),
        builder.getNamedAttr("slot", builder.getDenseI64ArrayAttr(slot)),
        builder.getNamedAttr("seq", builder.getI64IntegerAttr(table->seq)),
        builder.getNamedAttr("past", builder.getI64IntegerAttr(table->past)),
        builder.getNamedAttr("grid", builder.getI64IntegerAttr(table->grid))}));
  }
  if (mlir::failed(mlir::verify(module)))
    throw std::runtime_error(std::string("the ") + candidate.name +
                             " plan does not verify on the CG");
}

constexpr std::uint32_t kSeqBegin = 1, kSeqEnd = 2048;

std::string Emit(std::string const& export_json, mlir::MLIRContext& context,
                 Candidate const& candidate, dialect::PlacementTable const* table,
                 bool carry) {
  auto module = frontend::TorchExportImporter{}.Import(export_json, context, nullptr,
                                                       PlanOptions());
  if (carry) AttachPlan(*module, candidate, table);
  std::vector<codegen::RuntimeVariantModule> inputs{{*module, kSeqBegin, kSeqEnd}};
  return codegen::CouplingGraphToCUDA{}.LowerVariants(inputs);
}

/// The six shared candidates plus this round's chain.  `chain` rides the `kEft`
/// table path because the materializer's only table-carrying mode is that one:
/// the mode selects "read pi and sigma from the attached table", and which
/// scheduler produced the table is not something the host can or should see.
std::vector<Candidate> ChainCandidates() {
  auto candidates = Candidates();
  candidates.push_back({"chain", dialect::PlacementMode::kEft, {}});
  return candidates;
}

/// Measured solo duration per graph node from a round-one trace `slots.tsv`.
/// Empty when the dump is absent, so the caller falls back to the cost model.
///
/// `globaltimer_resolution_ns` is 1024 on sm_89 and the shortest tasks in these
/// dumps measure exactly one tick, so short stages are quantised at or below
/// the clock: a traced weight is an upper bound on how well the trace can
/// resolve a small task, not a more precise number than the cost model.  That
/// is reported, not corrected.
std::vector<double> ReadTaskNs(std::string const& path,
                               codegen::RuntimeTaskGraph const& graph,
                               std::vector<int> const& counts, int* observed) {
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
  int stage_column = column("stage"), task_column = column("logical_task");
  int begin_column = column("run_begin"), end_column = column("run_end");
  int const nodes = graph.stage_offsets.back();
  std::vector<double> task_ns(nodes, 0.0);
  std::vector<char> seen(nodes, 0);
  int widest = std::max(std::max(stage_column, task_column), std::max(begin_column, end_column));
  std::string line;
  while (std::getline(input, line)) {
    auto row = Split(line, '\t');
    if (int(row.size()) <= widest) continue;
    int stage = std::stoi(row[stage_column]), task = std::stoi(row[task_column]);
    if (stage < 0 || stage >= int(counts.size()) || task < 0 || task >= counts[stage])
      throw std::runtime_error("trace row outside the dumped cell: " + line);
    int node = graph.stage_offsets[stage] + task;
    double duration = double(std::stoll(row[end_column]) - std::stoll(row[begin_column]));
    if (duration < 0) throw std::runtime_error("trace task ends before it begins: " + line);
    // At kappa = 1 a node runs once; take the longest observation if a dump
    // ever carries more, so a weight is never an average over replicas.
    if (!seen[node] || duration > task_ns[node]) task_ns[node] = duration;
    seen[node] = 1;
  }
  *observed = int(std::count(seen.begin(), seen.end(), 1));
  return task_ns;
}

struct PathMetrics {
  int hops = 0;               ///< cross-worker task edges on the critical path
  int same_worker_edges = 0;  ///< task edges on it that cost no hop
  int queue_edges = 0;        ///< steps taken along a worker's queue, not the DAG
  int length = 0;
  double end_ns = 0.0;
  /// The path itself, last task first, with `edge[i]` the step from `path[i]`
  /// to `path[i + 1]`: 'h' a hop, 's' a same-worker task edge, 'q' a queue step.
  std::vector<int> path;
  std::vector<char> edge;
};

/// §6.4.  `SimulatorResult` reports `cross_worker_edges` over the whole graph,
/// which is not the quantity the lever moves: a plan can trade a hop off the
/// critical path for two hops elsewhere and come out ahead.  So walk the actual
/// path backwards from the last task to finish, stepping each time to whichever
/// predecessor -- a DAG predecessor plus its hop, or the previous slot on the
/// same worker -- determined this task's start, and count the edges it crosses.
///
/// The walk reconstructs the path under the same hop model the scheduler was
/// given; a different constant could pick a different tie, which is why the
/// counts are reported beside the makespan rather than in place of it.
PathMetrics WalkCriticalPath(codegen::RuntimeTaskGraph const& graph,
                             std::vector<std::vector<int>> const& predecessors,
                             std::vector<int> const& node_worker,
                             std::vector<int> const& node_slot,
                             std::vector<std::vector<int>> const& queue_node,
                             std::vector<double> const& hop_cost,
                             SimulatorResult const& result) {
  PathMetrics metrics;
  int const nodes = graph.stage_offsets.back();
  if (int(result.tasks.size()) != nodes) return metrics;
  int current = 0;
  for (int node = 1; node < nodes; ++node)
    if (result.tasks[node].end_ns > result.tasks[current].end_ns) current = node;
  metrics.end_ns = result.tasks[current].end_ns;
  metrics.path.push_back(current);
  for (int step = 0; step <= nodes; ++step) {
    if (step == nodes) throw std::runtime_error("the critical path walk did not terminate");
    int best = -1;
    bool best_is_queue = false;
    double best_finish = 0.0;
    for (int predecessor : predecessors[current]) {
      double finish = result.tasks[predecessor].end_ns +
                      (node_worker[predecessor] != node_worker[current] ? hop_cost[predecessor] : 0.0);
      if (best == -1 || finish > best_finish) {
        best = predecessor;
        best_finish = finish;
        best_is_queue = false;
      }
    }
    int worker = node_worker[current], slot = node_slot[current];
    if (worker >= 0 && slot > 0) {
      int previous = queue_node[worker][slot - 1];
      double finish = result.tasks[previous].end_ns;
      if (best == -1 || finish > best_finish) {
        best = previous;
        best_finish = finish;
        best_is_queue = true;
      }
    }
    if (best == -1) break;
    if (best_is_queue) {
      ++metrics.queue_edges;
      metrics.edge.push_back('q');
    } else if (node_worker[best] != node_worker[current]) {
      ++metrics.hops;
      metrics.edge.push_back('h');
    } else {
      ++metrics.same_worker_edges;
      metrics.edge.push_back('s');
    }
    ++metrics.length;
    current = best;
    metrics.path.push_back(current);
  }
  return metrics;
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 4 && argc != 8)
    throw std::invalid_argument(
        "usage: tilemega-place-chain REPO MANIFEST.tsv OUT_DIR [--target TARGET.json --hop HOP.tsv]\n"
        "manifest columns: model seq past generated_cu out_prefix trace_dir export_json");
  analysis::IslContext isl;
  std::string repo = argv[1], manifest_path = argv[2], out = argv[3];
  std::string target_path = repo + "/configs/targets/sm_89.json";
  std::string hop_path = repo + "/docs/experiments/SIMULATOR/hop_ns.tsv";
  if (argc == 8) {
    if (std::string(argv[4]) != "--target" || std::string(argv[6]) != "--hop")
      throw std::invalid_argument("expected --target TARGET.json --hop HOP.tsv");
    target_path = argv[5];
    hop_path = argv[7];
  }

  HopCurve hop;
  std::string error;
  if (!HopCurve::FromTsv(hop_path, &hop, &error))
    throw std::runtime_error("hop curve: " + error);
  auto target = TargetSpec::FromJson(target_path);
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();

  std::ifstream manifest(manifest_path);
  if (!manifest) throw std::runtime_error("cannot read manifest " + manifest_path);
  std::string line;
  if (!std::getline(manifest, line)) throw std::runtime_error("empty manifest");

  std::ofstream predicted(out + "/predicted.tsv");
  predicted << std::setprecision(10)
            << "model\tseq\tpast\tcandidate\tstatus\tmakespan_ns\tbusiest_worker_ns\t"
               "busiest_worker\ttotal_block_ns\tsolo_work_ns\tcritical_path_ns\t"
               "cross_worker_edges\tsame_worker_edges\tmax_queue\tmax_queue_ns\t"
               "spine_length_ns\tcritical_path_hops\tcritical_path_same_worker_edges\t"
               "critical_path_queue_edges\tcritical_path_len\teval_us\n";
  std::ofstream weights(out + "/chain_weights.tsv");
  weights << std::setprecision(10)
          << "model\tseq\tsource\tnodes\tobserved\tsolo_work_ns\tspine_length_ns\t"
             "max_queue_ns\tchain_count\tchain_interleaves\tsplit_count\tmakespan_ns"
             "\tfill_overflows\tworker_diff\tsolve_us\n";

  int cells = 0;
  while (std::getline(manifest, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto row = Split(line, '\t');
    if (row.size() < 7) throw std::runtime_error("manifest row needs 7 columns: " + line);
    std::string name = row[0];
    int seq = std::stoi(row[1]), past = std::stoi(row[2]);
    std::string generated = row[3], prefix = row[4], trace = row[5], export_json = row[6];

    auto cell = ReadCellDump(prefix + "_p5_s" + std::to_string(seq) + ".out");
    auto windows = ReadDependencies(generated, 0);
    auto graph = codegen::MaterializeRuntimeTaskGraph(cell.counts, windows, cell.grid);
    int const nodes = graph.stage_offsets.back();

    auto module = frontend::TorchExportImporter{}.Import(export_json, context, nullptr,
                                                         PlanOptions());
    auto runtime_plan = codegen::ReadRuntimePlan(*module);
    std::vector<GemmConfig> configs;
    for (auto const& gemm : runtime_plan.gemms)
      configs.push_back({int(gemm.tile_m), int(gemm.tile_n), int(gemm.tile_k),
                         int(gemm.stages), int(gemm.split_k)});
    auto model = ModelDescription::FromCouplingGraph(*module, {seq, past, seq + past}, name);
    if (model.stages.size() != cell.counts.size())
      throw std::runtime_error("the CG model has " + std::to_string(model.stages.size()) +
                               " stages but the dump reports " +
                               std::to_string(cell.counts.size()));
    auto operators = InstantiateModelTasks(model, configs);
    int threads = model.dtype == ScalarType::kBF16 ? kTensorBF16Threads : kSimtF32Threads;
    CostModelOptions prices;
    prices.unified_task_cost = true;
    CostModel cost(target, model.dtype, prices);
    Residency residency{cell.ctas_per_sm};
    auto stage_prices =
        PriceStages(cost, model, operators, configs, residency, threads, cell.counts.size());
    std::vector<double> task_ns(nodes, 0.0);
    for (std::size_t stage = 0; stage < cell.counts.size(); ++stage)
      for (int task = 0; task < cell.counts[stage]; ++task)
        task_ns[graph.stage_offsets[stage] + task] = stage_prices[stage].solo_ns;

    auto worker_sm = trace == "-" ? std::vector<int>{}
                                  : ReadWorkerSm(trace + "/slots.tsv", cell.grid);

    ChainRequest chain;
    chain.graph = &graph;
    chain.task_ns = task_ns;
    chain.grid = cell.grid;
    chain.sms = cell.num_sms;
    chain.ctas_per_sm = cell.ctas_per_sm;
    chain.worker_sm = worker_sm;
    chain.hop = hop;
    // Both fill arms are measured rather than asserted; see
    // `ChainRequest::cap_fill`.  The capped arm stays the default.
    if (char const* cap = std::getenv("TILEMEGA_CHAIN_CAP_FILL"))
      chain.cap_fill = (cap[0] != '0');
    // The stop rule is where chaining stops covering the graph: at mha4 s128
    // it admits four chains for 11920 nodes, so 63 of the 88 critical-path
    // steps are placed by the fill rather than by any chain.  Swept rather
    // than retuned in place, so the default stays the prompt's 1.0.
    if (char const* stop = std::getenv("TILEMEGA_CHAIN_STOP_RATIO"))
      chain.chain_stop_ratio = std::atof(stop);
    // The queue term the static extraction cannot see, handed back one round
    // at a time; see `ChainRequest::feedback_rounds`.  Default off.
    if (char const* fb = std::getenv("TILEMEGA_CHAIN_FEEDBACK"))
      chain.feedback_rounds = std::atoi(fb);
    // The arbiter, built from the same options and hop curve the emitted plan
    // is scored with below, so the extraction is ranked and selected by that
    // simulation instead of by the pass's own estimate (`ChainRequest::evaluate`).
    // Weights travel by value: the traced arm is a different set, and scoring it
    // on the cost model's would compare two plans under one weighting.
    auto make_evaluator = [&](std::vector<double> weights) {
      return [&, weights](ChainSchedule const& pass, double* makespan,
                          std::vector<double>* blocked, std::string* err) {
        PlanRequest probe;
        probe.mode = dialect::PlacementMode::kEft;
        probe.grid = cell.grid;
        probe.counts = cell.counts;
        probe.stage_order = cell.stage_order;
        probe.physical_worker.resize(cell.grid);
        std::iota(probe.physical_worker.begin(), probe.physical_worker.end(), 0);
        probe.graph = &graph;
        probe.eft_worker = pass.worker;
        probe.eft_slot = pass.slot;
        MaterializedPlan probe_plan;
        if (!MaterializePlanPlacement(probe, &probe_plan, err)) return false;
        SimulatorInput probe_input;
        probe_input.graph = &graph;
        probe_input.task_ns = weights;
        probe_input.worker_sm = worker_sm;
        SimulatorOptions probe_options;
        probe_options.sms = cell.num_sms;
        probe_options.ctas_per_sm = cell.ctas_per_sm;
        probe_options.proportional_sharing = true;
        SimulatorResult probe_result;
        if (!SimulateExecution(probe_input, probe_plan, probe_options, hop,
                               &probe_result, err))
          return false;
        *makespan = probe_result.makespan_ns;
        blocked->assign(probe_result.tasks.size(), 0.0);
        for (std::size_t node = 0; node < probe_result.tasks.size(); ++node)
          (*blocked)[node] = probe_result.tasks[node].block_ns;
        return true;
      };
    };
    if (chain.feedback_rounds > 0) chain.evaluate = make_evaluator(task_ns);

    ChainSchedule schedule;
    auto solve_begin = std::chrono::steady_clock::now();
    if (!ScheduleByCriticalChain(chain, &schedule, &error))
      throw std::runtime_error("chain scheduling " + name + " seq " + std::to_string(seq) +
                               ": " + error);
    double solve_us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - solve_begin).count();
    double solo_work_ns = 0.0;
    for (double value : task_ns) solo_work_ns += value;
    weights << name << '\t' << seq << "\tcost_model\t" << nodes << '\t' << nodes << '\t'
            << solo_work_ns << '\t' << schedule.spine_length_ns << '\t'
            << schedule.max_queue_ns << '\t' << schedule.chain_count << '\t'
            << schedule.chain_interleaves << '\t' << schedule.split_count << '\t' << schedule.makespan_ns << '\t'
            << schedule.fill_overflows << "\t0\t"
            << solve_us << '\n';

    // §6.2's second weight source.  Solved and reported, never emitted: see the
    // file header for why the emitted plan stays on one basis for every cell.
    int observed = 0;
    auto traced_ns = trace == "-" ? std::vector<double>{}
                                  : ReadTaskNs(trace + "/slots.tsv", graph, cell.counts, &observed);
    if (!traced_ns.empty()) {
      // A node the dump never observed keeps its cost-model weight rather than
      // a zero, which would silently shorten every chain through it.
      for (int node = 0; node < nodes; ++node)
        if (traced_ns[node] == 0.0) traced_ns[node] = task_ns[node];
      ChainRequest traced = chain;
      traced.task_ns = traced_ns;
      if (traced.feedback_rounds > 0) traced.evaluate = make_evaluator(traced_ns);
      ChainSchedule traced_schedule;
      auto traced_begin = std::chrono::steady_clock::now();
      if (!ScheduleByCriticalChain(traced, &traced_schedule, &error))
        throw std::runtime_error("chain scheduling on traced weights " + name + ": " + error);
      double traced_us = std::chrono::duration<double, std::micro>(
                             std::chrono::steady_clock::now() - traced_begin).count();
      int worker_diff = 0;
      for (int node = 0; node < nodes; ++node)
        worker_diff += traced_schedule.worker[node] != schedule.worker[node];
      double traced_work = 0.0;
      for (double value : traced_ns) traced_work += value;
      weights << name << '\t' << seq << "\ttrace\t" << nodes << '\t' << observed << '\t'
              << traced_work << '\t' << traced_schedule.spine_length_ns << '\t'
              << traced_schedule.max_queue_ns << '\t' << traced_schedule.chain_count << '\t'
              << traced_schedule.chain_interleaves << '\t'
              << traced_schedule.split_count << '\t' << traced_schedule.makespan_ns << '\t'
              << traced_schedule.fill_overflows << '\t'
              << worker_diff << '\t' << traced_us << '\n';
    }

    dialect::PlacementTable table;
    table.worker = schedule.worker;
    table.slot = schedule.slot;
    table.seq = seq;
    table.past = past;
    table.grid = cell.grid;
    if (!dialect::ValidatePlacementTable(table, &error))
      throw std::runtime_error("the chain schedule is not a dense (pi, sigma): " + error);

    std::vector<std::vector<int>> predecessors(nodes);
    for (int node = 0; node < nodes; ++node)
      for (int successor : graph.successors[node]) predecessors[successor].push_back(node);
    std::vector<double> hop_cost(nodes, 0.0);
    for (int node = 0; node < nodes; ++node)
      hop_cost[node] = hop.Ns(int(graph.successors[node].size()), 1);

    SimulatorInput input;
    input.graph = &graph;
    input.task_ns = task_ns;
    input.worker_sm = worker_sm;

    PlanRequest request;
    request.grid = cell.grid;
    request.counts = cell.counts;
    request.stage_order = cell.stage_order;
    request.physical_worker.resize(cell.grid);
    std::iota(request.physical_worker.begin(), request.physical_worker.end(), 0);
    request.graph = &graph;

    // §6.5's S2c-b compares six candidates, so `eft` may not stay a hole in
    // the table.  It is solved on the same weights, grid and hop curve the
    // chain scheduler saw, which is what makes the two rows comparable at all.
    tilemega::solver::EftRequest eft;
    eft.graph = &graph;
    eft.task_ns = task_ns;
    eft.grid = cell.grid;
    eft.sms = cell.num_sms;
    eft.ctas_per_sm = cell.ctas_per_sm;
    eft.worker_sm = worker_sm;
    eft.hop = hop;
    tilemega::solver::EftSchedule eft_schedule;
    bool const have_eft = ScheduleByEarliestFinish(eft, &eft_schedule, &error);
    if (!have_eft)
      std::cerr << "PLACE_SKIP model=" << name << " seq=" << seq
                << " candidate=eft reason=\"" << error << "\"\n";

    for (auto const& candidate : ChainCandidates()) {
      request.mode = candidate.mode;
      request.params = candidate.params;
      request.eft_worker.clear();
      request.eft_slot.clear();
      // Both table-driven candidates carry their own (pi, sigma): `chain` from
      // the critical-chain solver above, `eft` from the earliest-finish solver.
      if (candidate.name == std::string("chain")) {
        request.eft_worker = table.worker;
        request.eft_slot = table.slot;
      } else if (candidate.mode == dialect::PlacementMode::kEft) {
        if (!have_eft) {
          predicted << name << '\t' << seq << '\t' << past << '\t' << candidate.name
                    << "\tno_table\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n";
          continue;
        }
        request.eft_worker = eft_schedule.worker;
        request.eft_slot = eft_schedule.slot;
      }
      MaterializedPlan plan;
      if (!MaterializePlanPlacement(request, &plan, &error)) {
        predicted << name << '\t' << seq << '\t' << past << '\t' << candidate.name
                  << "\tunimplemented\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n";
        std::cerr << "PLACE_SKIP model=" << name << " seq=" << seq
                  << " candidate=" << candidate.name << " reason=\"" << error << "\"\n";
        continue;
      }
      if (!CheckPlanLegality(graph, plan, &error))
        throw std::runtime_error(std::string("candidate ") + candidate.name +
                                 " is illegal under §5.7.3: " + error);

      std::vector<int> node_worker(nodes, -1), node_slot(nodes, -1);
      for (std::size_t stage = 0; stage < cell.counts.size(); ++stage)
        for (int task = 0; task < cell.counts[stage]; ++task) {
          int node = graph.stage_offsets[stage] + task;
          node_worker[node] = plan.owner[stage][task];
          node_slot[node] = plan.slot[stage][task];
        }
      std::vector<std::vector<int>> queue_node(plan.queue.size());
      long max_queue = 0;
      double max_queue_ns = 0.0;
      for (std::size_t worker = 0; worker < plan.queue.size(); ++worker) {
        double queue_ns = 0.0;
        for (auto const& item : plan.queue[worker]) {
          int node = graph.stage_offsets[item.stage] + item.logical;
          queue_node[worker].push_back(node);
          queue_ns += task_ns[node];
        }
        max_queue = std::max(max_queue, long(plan.queue[worker].size()));
        max_queue_ns = std::max(max_queue_ns, queue_ns);
      }

      SimulatorOptions simulator;
      simulator.sms = cell.num_sms;
      simulator.ctas_per_sm = cell.ctas_per_sm;
      simulator.proportional_sharing = true;
      SimulatorResult result;
      auto begin = std::chrono::steady_clock::now();
      if (!SimulateExecution(input, plan, simulator, hop, &result, &error))
        throw std::runtime_error(std::string("simulating ") + candidate.name + ": " + error);
      double eval_us = std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - begin).count();
      auto path = WalkCriticalPath(graph, predecessors, node_worker, node_slot, queue_node,
                                   hop_cost, result);

      // Diagnostic, silent unless asked for.  §6.4's counts say how many hops
      // the critical path pays, not *which* path pays them, and at gqa2 s4 the
      // spine is contiguous on one worker while only 2 of 19 critical-path
      // edges are same-worker: the two are different paths, and only the node
      // lists say why.
      if (char const* dump_dir = std::getenv("TILEMEGA_CHAIN_PATH_DUMP")) {
        std::string stem = std::string(dump_dir) + "/" + name + "_s" +
                           std::to_string(seq) + "_" + candidate.name;
        bool const is_chain = candidate.name == std::string("chain");
        std::ofstream path_out(stem + "_path.tsv");
        path_out << std::setprecision(10)
                 << "step\tnode\tstage\tworker\tsm\tchain\ttask_ns\tstart_ns\tend_ns\t"
                    "stretch\tblock_ns\tedge_to_next\n";
        for (std::size_t i = 0; i < path.path.size(); ++i) {
          int const node = path.path[i];
          int const stage =
              int(std::upper_bound(graph.stage_offsets.begin(),
                                   graph.stage_offsets.end(), node) -
                  graph.stage_offsets.begin()) - 1;
          int const worker = node_worker[node];
          path_out << i << '\t' << node << '\t' << stage << '\t' << worker << '\t'
                   << (worker >= 0 && worker < int(worker_sm.size()) ? worker_sm[worker] : -1)
                   << '\t'
                   << (is_chain && node < int(schedule.chain_of.size())
                           ? schedule.chain_of[node] : -1)
                   << '\t' << task_ns[node] << '\t' << result.tasks[node].start_ns
                   << '\t' << result.tasks[node].end_ns << '\t'
                   << result.tasks[node].stretch << '\t' << result.tasks[node].block_ns
                   << '\t' << (i < path.edge.size() ? path.edge[i] : '-') << '\n';
        }
        // Every predecessor of every critical-path node, with what it cost to
        // reach: this is what separates "the extractor chained the wrong path"
        // from "the gating predecessor is in another lane and no chain can own
        // both".  `gating` marks the latest arrival, which is what set the start.
        std::ofstream pred_out(stem + "_preds.tsv");
        pred_out << std::setprecision(10)
                 << "step\tnode\tpred\tpred_stage\tpred_chain\tpred_worker\t"
                    "pred_end_ns\tarrival_ns\tcross\tgating\n";
        for (std::size_t i = 0; i < path.path.size(); ++i) {
          int const node = path.path[i];
          double best_arrival = -1.0;
          for (int pred : predecessors[node])
            best_arrival = std::max(
                best_arrival,
                result.tasks[pred].end_ns +
                    (node_worker[pred] != node_worker[node] ? hop_cost[pred] : 0.0));
          for (int pred : predecessors[node]) {
            bool const cross = node_worker[pred] != node_worker[node];
            double const arrival =
                result.tasks[pred].end_ns + (cross ? hop_cost[pred] : 0.0);
            int const pred_stage =
                int(std::upper_bound(graph.stage_offsets.begin(),
                                     graph.stage_offsets.end(), pred) -
                    graph.stage_offsets.begin()) - 1;
            pred_out << i << '\t' << node << '\t' << pred << '\t' << pred_stage << '\t'
                     << (is_chain && pred < int(schedule.chain_of.size())
                             ? schedule.chain_of[pred] : -1)
                     << '\t' << node_worker[pred] << '\t' << result.tasks[pred].end_ns
                     << '\t' << arrival << '\t' << (cross ? 1 : 0) << '\t'
                     << (arrival >= best_arrival ? 1 : 0) << '\n';
          }
        }

        if (is_chain) {
          std::vector<int> chain_nodes(schedule.chain_count, 0);
          std::vector<double> chain_ns(schedule.chain_count, 0.0);
          std::vector<int> chain_worker(schedule.chain_count, -1);
          for (int node = 0; node < nodes; ++node) {
            int const id = schedule.chain_of[node];
            if (id < 0) continue;
            ++chain_nodes[id];
            chain_ns[id] += task_ns[node];
            // -2 would mean a chain straddling two workers, which the placement
            // is supposed to make impossible; it is reported, not assumed away.
            if (chain_worker[id] == -1) chain_worker[id] = node_worker[node];
            else if (chain_worker[id] != node_worker[node]) chain_worker[id] = -2;
          }
          std::ofstream chains_out(stem + "_chains.tsv");
          chains_out << std::setprecision(10) << "chain\tnodes\tns\tworker\n";
          for (int id = 0; id < schedule.chain_count; ++id)
            chains_out << id << '\t' << chain_nodes[id] << '\t' << chain_ns[id] << '\t'
                       << chain_worker[id] << '\n';
        }
      }

      predicted << name << '\t' << seq << '\t' << past << '\t' << candidate.name << "\tok\t"
                << result.makespan_ns << '\t' << result.busiest_worker_ns << '\t'
                << result.busiest_worker << '\t' << result.total_block_ns << '\t'
                << result.solo_work_ns << '\t' << result.critical_path_ns << '\t'
                << result.cross_worker_edges << '\t' << result.same_worker_edges << '\t'
                << max_queue << '\t' << max_queue_ns << '\t'
                << (candidate.name == std::string("chain") ? schedule.spine_length_ns : 0.0)
                << '\t' << path.hops << '\t' << path.same_worker_edges << '\t'
                << path.queue_edges << '\t' << path.length << '\t' << eval_us << '\n';

      bool const carry = candidate.name == std::string("chain") ||
                         candidate.mode == dialect::PlacementMode::kTemplate;
      bool const provenance = candidate.mode == dialect::PlacementMode::kLegacyGridStride;
      if (!carry && !provenance) continue;
      std::string path_cu = out + "/plan/" + name + "_s" + std::to_string(seq) + "_" +
                            candidate.name + ".cu";
      std::ofstream source(path_cu);
      if (!source) throw std::runtime_error("cannot write " + path_cu);
      source << Emit(export_json, context, candidate,
                     candidate.name == std::string("chain") ? &table : nullptr, carry);
    }
    ++cells;
    std::cerr << "CHAIN_CELL model=" << name << " seq=" << seq << " nodes=" << nodes
              << " grid=" << cell.grid << " solve_us=" << solve_us
              << " chains=" << schedule.chain_count
              << " interleaves=" << schedule.chain_interleaves
              << " splits=" << schedule.split_count
              << " fill_overflows=" << schedule.fill_overflows
              << " spine_ns=" << schedule.spine_length_ns
              << " worker_sm=" << (worker_sm.empty() ? "modulo" : "traced") << '\n';
  }
  std::cerr << "CHAIN_SUMMARY cells=" << cells << '\n';
  if (isl.ReferenceCount() != 0) throw std::runtime_error("place-chain retained isl objects");
  return 0;
} catch (std::exception const& error) {
  std::cerr << "tilemega-place-chain: " << error.what() << '\n';
  return 2;
}
