// SPDX-License-Identifier: BSD-3-Clause
// B1-d: the read-only frontier as a dimension of sigma, checked at the three
// places it is written -- derived in Analysis, priced in Solver, and carried
// through the bounds into the plan table the dialect verifies.
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/tasks/ScalarDataflow.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/ExecutionSimulator.h>
#include <tilemega/Solver/TaskModel.h>
#include <mlir/IR/MLIRContext.h>
#include <iostream>
#include <stdexcept>

using namespace tilemega;
using namespace tilemega::solver;

static void Require(bool value, char const* what) {
  if (!value) throw std::runtime_error(what);
}

/// EX-E4's hand table, as a rule rather than 102 rows: a tensor is off the
/// frontier exactly when some stage writes it, and in these two models the
/// tensors nobody writes are the checkpoint parameters and the model input.
/// The rule reads tensor names; `DeriveTaskWork` reads the producer relation.
/// The point of the check is that the two agree on every stage.
static bool HandFrontier(std::string const& tensor) {
  return tensor.rfind("p_", 0) == 0 || tensor == "hidden";
}

/// Locations 1 and 2: the derived mask on all stages of both reference models,
/// and what the calibrated cost model charges for the bytes behind it.
static void CheckReferenceModels(mlir::MLIRContext& ctx) {
  int stages = 0, frontier_stages = 0, rope_stages = 0;
  analysis::ParamBinding theta;
  theta.values = {{"s11", 4}, {"s12", 3}};
  for (auto model : {"gqa2", "mha4"}) {
    auto module = frontend::TorchExportImporter{}.Import(
        std::string(TILEMEGA_SOURCE_DIR) + "/docs/experiments/SEQSCAN/raw/export/" +
        std::string(model) + ".json", ctx);
    auto description = ModelDescription::FromCouplingGraph(*module, {4, 3, 7}, model);
    auto plan = codegen::ReadRuntimePlan(*module);
    std::vector<GemmConfig> configs;
    for (auto const& g : plan.gemms)
      configs.push_back({g.tile_m, g.tile_n, g.tile_k, g.stages, g.split_k});
    auto semantic = InstantiateModelTasks(description, configs);
    for (auto const& task : description.task_semantics) {
      auto const& stage = description.stages.at(task.stage);
      auto derived = DeriveModelTaskInput(description, task, semantic,
                                          stage.IsCollective() ? &configs.at(stage.gemm) : nullptr);
      auto const* node = semantic.Find(task.op.name);
      Require(node != nullptr, "instantiated task missing");
      bool any_operand_frontier = false, all_operands_frontier = true;
      for (auto const& operand : node->operands) {
        bool const derived_frontier = operand.producer.empty();
        if (derived_frontier != HandFrontier(operand.tensor.name))
          throw std::runtime_error(std::string(model) + ' ' + task.op.name + ": " +
              operand.tensor.name + " is " + (derived_frontier ? "" : "not ") +
              "on the derived frontier and " + (derived_frontier ? "not " : "") +
              "on the hand table");
        any_operand_frontier |= derived_frontier;
        all_operands_frontier &= derived_frontier;
      }
      auto const frontier = derived.work.frontier_read_elements.ToString();
      bool const empty_frontier = frontier == "{ 0 }";
      // A stage whose operands are all produced still has a frontier if it
      // reads a constant table no operator writes: the rope frequencies are
      // exactly that, and they are the reason the mask is derived from the
      // read relations rather than from the operand list.
      bool const rope = task.op.name.find(".rope") != std::string::npos;
      Require(empty_frontier != (any_operand_frontier || rope), "frontier disagrees with the mask");
      // The two sets are printed over different coordinate names, so compare
      // the totals at the bound theta rather than the text.
      if (all_operands_frontier &&
          derived.work.frontier_read_elements.SumDomain().Eval(theta) !=
              derived.work.read_elements.SumDomain().Eval(theta))
        throw std::runtime_error(std::string(model) + ' ' + task.op.name +
            ": every operand unproduced, yet the frontier is not the whole read");
      stages += 1;
      frontier_stages += !empty_frontier;
      rope_stages += rope;
    }
  }
  std::cout << "PIPELINE_FRONTIER stages=" << stages << " with_frontier=" << frontier_stages
            << " rope_element_reads=" << rope_stages << '\n';
}

/// Location 2, priced: what the calibrated model charges for the operand the
/// body prefetches, under the executor's own page test.
static void CheckPricing(mlir::MLIRContext& ctx) {
  auto module = frontend::TorchExportImporter{}.Import(
      std::string(TILEMEGA_SOURCE_DIR) + "/docs/experiments/SEQSCAN/raw/export/gqa2.json", ctx);
  dialect::PlacementSolveOptions options;
  options.target = TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR) + "/configs/targets/sm_89.json");
  options.dims = {4, 3, 7};
  options.residency = 1;
  options.kappa = 1;
  options.verified_resident_limit = 1;
  options.requested_grid = 8;
  auto prepared = dialect::PreparePlacementProblem(*module, options);
  Require(prepared.prefetch_ns.size() == prepared.task_ns.size(), "prefetch prices are not per task");
  int priced = 0;
  double share = 0;
  for (std::size_t s = 0; s + 1 < prepared.graph.stage_offsets.size(); ++s) {
    auto const& stage = prepared.model.stages.at(prepared.projection.stages[s].logical_stage);
    // Only the norm bodies declare a Prefetch, and their scale row (1 KiB at
    // this width) is on the frontier at every layer, so exactly those stages
    // may carry a credit; whether they do is the model's answer, not a rule.
    bool const declared = codegen::ScalarPrefetchOperand(static_cast<codegen::TaskKind>(stage.kind)) >= 0;
    for (int n = prepared.graph.stage_offsets[s]; n < prepared.graph.stage_offsets[s + 1]; ++n) {
      Require(prepared.prefetch_ns[n] >= 0 && prepared.prefetch_ns[n] <= prepared.task_ns[n],
              "a frontier share outside its own task");
      if (!declared) Require(prepared.prefetch_ns[n] == 0, "credit on a body that prefetches nothing");
      if (prepared.prefetch_ns[n] > 0) { ++priced; share += prepared.prefetch_ns[n] / prepared.task_ns[n]; }
    }
  }
  Require(priced > 0, "no norm task in gqa2 prices its scale row");
  // A page the row does not fit is never issued by the executor, so it is
  // never credited by the solver either.
  options.prefetch_page_bytes = 512;
  auto small = dialect::PreparePlacementProblem(*module, options);
  for (double ns : small.prefetch_ns) Require(ns == 0, "credit for a prefetch the executor refuses");
  std::cout << "PIPELINE_PRICE tasks=" << prepared.task_ns.size() << " priced=" << priced
            << " mean_share=" << share / priced << " page512_priced=0\n";
}

/// Location 3: the overlap a pipelinable adjacency buys, in both bounds and in
/// the simulator, and the flags that travel to CG.
static void CheckBounds() {
  codegen::RuntimeTaskGraph graph{{0, 2}, {{}, {}}, {}, 0};
  SimulatorInput in;
  in.graph = &graph;
  in.task_ns = {10, 10};
  MaterializedPlan one, two;
  one.queue = {{{0, 0}, {0, 1}}};
  two.queue = {{{0, 0}}, {{0, 1}}};
  PreparedPlanBounds prepared;
  std::string error;
  PlanBounds plain, piped, split;
  Require(PreparePlanBounds(in, &prepared, &error), error.c_str());
  Require(EvaluatePlanBounds(prepared, one, &plain, &error), error.c_str());
  Require(plain.queue_lb_ns == 20 && plain.binding_path_ns == 20, "unpipelined queue");
  Require(PipelinedSlots(in, one).empty() ||
          PipelinedSlots(in, one) == std::vector<unsigned char>{0, 0}, "no frontier, no flags");

  in.prefetch_ns = {4, 4};
  Require(PreparePlanBounds(in, &prepared, &error), error.c_str());
  Require(EvaluatePlanBounds(prepared, one, &piped, &error), error.c_str());
  // The head of the queue has nothing to overlap with; the second slot hides
  // its whole 4 ns fetch under the first slot's body.
  Require(piped.queue_lb_ns == 16 && piped.binding_path_ns == 16, "pipelined queue");
  Require(PipelinedSlots(in, one) == std::vector<unsigned char>({0, 1}), "pipelined flags");
  // Same tasks, same frontier, one per worker: both are queue heads, so there
  // is nothing to overlap and the cost is the unpipelined one.
  Require(EvaluatePlanBounds(prepared, two, &split, &error), error.c_str());
  Require(split.binding_path_ns == 10 && PipelinedSlots(in, two) == std::vector<unsigned char>({0, 0}),
          "two workers cannot pipeline one queue");
  // An overlap can never exceed the body it hides under.
  in.task_ns = {1, 10};
  Require(PreparePlanBounds(in, &prepared, &error), error.c_str());
  Require(EvaluatePlanBounds(prepared, one, &piped, &error), error.c_str());
  Require(piped.queue_lb_ns == 10, "overlap clamped to the predecessor");

  SimulatorOptions sim_options;
  sim_options.observed_task_times = true;
  sim_options.flat_hop = true;
  HopCurve hop;
  SimulatorResult sim;
  in.task_ns = {10, 10};
  Require(SimulateExecution(in, one, sim_options, hop, &sim, &error), error.c_str());
  Require(sim.makespan_ns == 16, "simulated makespan must see the same overlap");
  in.prefetch_ns.clear();
  Require(SimulateExecution(in, one, sim_options, hop, &sim, &error), error.c_str());
  Require(sim.makespan_ns == 20, "a build without the mechanism pipelines nothing");
  std::cout << "PIPELINE_BOUNDS queue_lb binding_path makespan flags\n";
}

/// Location 3, written back: the table the dialect accepts and what it rejects.
static void CheckPlacementTable() {
  dialect::PlacementTable table;
  table.worker = {0, 0};
  table.slot = {0, 1};
  table.seq = 4;
  table.past = 3;
  table.grid = 1;
  std::string error;
  Require(dialect::ValidatePlacementTable(table, &error), error.c_str());
  table.pipeline = {0, 1};
  Require(dialect::ValidatePlacementTable(table, &error), error.c_str());
  table.pipeline = {1, 1};
  Require(!dialect::ValidatePlacementTable(table, &error), "a queue head cannot pipeline");
  table.pipeline = {0};
  Require(!dialect::ValidatePlacementTable(table, &error), "flags must cover every node");
  std::cout << "PIPELINE_TABLE accepted rejected-head rejected-length\n";
}

int main() try {
  analysis::IslContext isl;
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<dialect::CGDialect>();
  CheckReferenceModels(ctx);
  CheckPricing(ctx);
  CheckBounds();
  CheckPlacementTable();
  std::cout << "PIPELINE_SIGMA PASS frontier pricing bounds table\n";
  return 0;
} catch (std::exception const& e) {
  std::cerr << "pipeline sigma: " << e.what() << '\n';
  return 1;
}
