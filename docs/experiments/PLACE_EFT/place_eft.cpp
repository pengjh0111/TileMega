// SPDX-License-Identifier: BSD-3-Clause
// EX-S2 driver: run the EFT scheduler at a bound theta, write the resulting
// (pi, sigma) into the Coupling Graph as a materialized Plan, and emit the
// generated source that carries it to the GPU.
//
// The scheduling decision is taken here, in the solver, once per cell.  Codegen
// and the host only transport the answer (H4), which is why this tool exists at
// all: `eft` is the one mode with no closed form, so there is no layer below L2
// that could evaluate it after theta is bound.
//
// It also predicts every §6.2 candidate through the frozen `SimulateExecution`,
// on the same cell inputs, so S2-e compares a prediction and a measurement of
// the same six plans rather than two differently-built sets.
//
// Provenance: the `legacy_grid_stride` source this tool emits is diffed by
// run.sh against the committed one the control arms are built from.  If they
// are byte-identical then the import options, the variant interval and the
// codegen path here are the ones the measured arms went through, and the only
// difference between an arm and the control is the Plan (E1-b, H2).
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/EftPlacement.h>
#include <tilemega/Solver/ExecutionSimulator.h>
#include <tilemega/Solver/PlanMaterialize.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include "../SIMULATOR/cell_inputs.h"

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

/// The import options `docs/experiments/OWNERSHIP/plan_structured.json` spells
/// out.  Its GEMM block is the `GemmGranularity` default (128x128x16, 3 stages,
/// no split-K), so leaving `gemms` empty is the same request, and run.sh's diff
/// against the committed legacy source is what proves it rather than this
/// comment.
frontend::ImportOptions PlanOptions() {
  frontend::ImportOptions options;
  options.rope_tile_per_block = true;
  options.kv_tile_per_block = true;
  options.activation_tile_per_block = true;
  options.combiner_tile_per_block = true;
  return options;
}

/// Write a Plan onto every task space of the module, and for `eft` the
/// materialized table beside them.  The verifier is re-run afterwards because
/// the importer's own `verify` ran before these attributes existed, and an
/// unpaired mode or a table that is not a dense (pi, sigma) must hard-fail here
/// rather than reach codegen (H5).
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

/// `seq_begin`/`seq_end` of the one variant `plan_structured.json` declares.
/// An `eft` table is pinned to a single seq inside that interval and the host
/// refuses the rest; the interval is kept anyway so the emitted source differs
/// from the control only in the Plan.
constexpr std::uint32_t kSeqBegin = 1, kSeqEnd = 2048;

std::string Emit(std::string const& export_json, mlir::MLIRContext& context,
                 Candidate const& candidate, dialect::PlacementTable const* table,
                 bool carry) {
  auto module = frontend::TorchExportImporter{}.Import(export_json, context, nullptr,
                                                       PlanOptions());
  // `legacy_grid_stride` is emitted with no Plan attributes at all: absent is
  // what the pre-plan importer produced and what E1-b pins, and writing the
  // mode explicitly would be a different module for the same schedule.
  if (carry) AttachPlan(*module, candidate, table);
  std::vector<codegen::RuntimeVariantModule> inputs{{*module, kSeqBegin, kSeqEnd}};
  return codegen::CouplingGraphToCUDA{}.LowerVariants(inputs);
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 4)
    throw std::invalid_argument(
        "usage: tilemega-place-eft REPO MANIFEST.tsv OUT_DIR\n"
        "manifest columns: model seq past generated_cu out_prefix trace_dir export_json");
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
            << "model\tseq\tpast\tcandidate\tstatus\tmakespan_ns\tbusiest_worker_ns\t"
               "busiest_worker\ttotal_block_ns\tsolo_work_ns\tcritical_path_ns\t"
               "cross_worker_edges\tsame_worker_edges\tmax_queue\teval_us\n";
  std::ofstream schedules(out + "/eft_schedule.tsv");
  schedules << std::setprecision(10)
            << "model\tseq\tnodes\tgrid\tsolve_us\tgreedy_makespan_ns\tmax_queue\t"
               "same_worker_edges\tcross_worker_edges\tworkers_used\n";

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
    std::vector<int> node_stage(nodes, 0);
    for (std::size_t stage = 0; stage + 1 < graph.stage_offsets.size(); ++stage)
      for (int node = graph.stage_offsets[stage]; node < graph.stage_offsets[stage + 1]; ++node)
        node_stage[node] = int(stage);

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

    EftRequest eft;
    eft.graph = &graph;
    eft.task_ns = task_ns;
    eft.grid = cell.grid;
    eft.sms = cell.num_sms;
    eft.ctas_per_sm = cell.ctas_per_sm;
    eft.worker_sm = worker_sm;
    eft.hop = hop;
    EftSchedule schedule;
    auto solve_begin = std::chrono::steady_clock::now();
    if (!ScheduleByEarliestFinish(eft, &schedule, &error))
      throw std::runtime_error("eft scheduling " + name + " seq " + std::to_string(seq) +
                               ": " + error);
    double solve_us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - solve_begin).count();

    dialect::PlacementTable table;
    table.worker = schedule.worker;
    table.slot = schedule.slot;
    table.seq = seq;
    table.past = past;
    table.grid = cell.grid;
    if (!dialect::ValidatePlacementTable(table, &error))
      throw std::runtime_error("the eft schedule is not a dense (pi, sigma): " + error);

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

    for (auto const& candidate : Candidates()) {
      request.mode = candidate.mode;
      request.params = candidate.params;
      request.eft_worker.clear();
      request.eft_slot.clear();
      if (candidate.mode == dialect::PlacementMode::kEft) {
        request.eft_worker = table.worker;
        request.eft_slot = table.slot;
      }
      MaterializedPlan plan;
      if (!MaterializePlanPlacement(request, &plan, &error)) {
        predicted << name << '\t' << seq << '\t' << past << '\t' << candidate.name
                  << "\tunimplemented\t\t\t\t\t\t\t\t\t\n";
        std::cerr << "PLACE_SKIP model=" << name << " seq=" << seq
                  << " candidate=" << candidate.name << " reason=\"" << error << "\"\n";
        continue;
      }
      if (!CheckPlanLegality(graph, plan, &error))
        throw std::runtime_error(std::string("candidate ") + candidate.name +
                                 " is illegal under §5.7.3: " + error);
      long max_queue = 0;
      for (auto const& queue : plan.queue) max_queue = std::max(max_queue, long(queue.size()));

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
      predicted << name << '\t' << seq << '\t' << past << '\t' << candidate.name << "\tok\t"
                << result.makespan_ns << '\t' << result.busiest_worker_ns << '\t'
                << result.busiest_worker << '\t' << result.total_block_ns << '\t'
                << result.solo_work_ns << '\t' << result.critical_path_ns << '\t'
                << result.cross_worker_edges << '\t' << result.same_worker_edges << '\t'
                << max_queue << '\t' << eval_us << '\n';

      if (candidate.mode == dialect::PlacementMode::kEft) {
        int used = 0;
        for (auto const& queue : plan.queue) used += !queue.empty();
        schedules << name << '\t' << seq << '\t' << nodes << '\t' << cell.grid << '\t'
                  << solve_us << '\t' << schedule.makespan_ns << '\t' << max_queue << '\t'
                  << result.same_worker_edges << '\t' << result.cross_worker_edges << '\t'
                  << used << '\n';
      }

      // Only the modes with no macro emit a source: 0, 4 and 5 are measured
      // from the committed control sources so that the arms this round compares
      // against are the same binaries round one measured (H6).
      bool const carry = candidate.mode == dialect::PlacementMode::kEft ||
                         candidate.mode == dialect::PlacementMode::kTemplate;
      bool const provenance = candidate.mode == dialect::PlacementMode::kLegacyGridStride;
      if (!carry && !provenance) continue;
      std::string path = out + "/plan/" + name + "_s" + std::to_string(seq) + "_" +
                         candidate.name + ".cu";
      std::ofstream source(path);
      if (!source) throw std::runtime_error("cannot write " + path);
      source << Emit(export_json, context, candidate,
                     candidate.mode == dialect::PlacementMode::kEft ? &table : nullptr, carry);
    }
    ++cells;
    std::cerr << "PLACE_CELL model=" << name << " seq=" << seq << " nodes=" << nodes
              << " grid=" << cell.grid << " solve_us=" << solve_us
              << " worker_sm=" << (worker_sm.empty() ? "modulo" : "traced") << '\n';
  }
  std::cerr << "PLACE_SUMMARY cells=" << cells << '\n';
  if (isl.ReferenceCount() != 0) throw std::runtime_error("place-eft retained isl objects");
  return 0;
} catch (std::exception const& error) {
  std::cerr << "tilemega-place-eft: " << error.what() << '\n';
  return 2;
}
