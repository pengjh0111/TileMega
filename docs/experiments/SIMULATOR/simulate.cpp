// SPDX-License-Identifier: BSD-3-Clause
// EX-S1 driver: evaluate every candidate Plan of a cell through
// `SimulateExecution` and write the predicted numbers as data.
//
// The cell's inputs -- DAG, active counts, worker -> SM, node durations -- come
// from `cell_inputs.h`, which states where each one is read from.  What this
// file adds is the check that closes the provenance argument: (pi, sigma) comes
// from `MaterializePlanPlacement`, the one routine the host also calls, and the
// re-materialized modes 0/4/5 are compared against the `E2E_PLACE_STATS` line
// of their own run.  If `same_worker_edges`, `cross_worker_edges` and
// `max_queue` all reproduce, then the DAG and the placement under simulation
// are provably the ones that ran, and a new candidate is materialized on the
// same footing rather than on a parallel code path.
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/ExecutionSimulator.h>
#include <tilemega/Solver/PlanMaterialize.h>
#include <mlir/IR/MLIRContext.h>

#include "cell_inputs.h"

#include <algorithm>
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
