// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/CouplingCache.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Solver/ModelDramFloor.h>
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Solver/FlowPreparation.h>
#include <tilemega/Solver/StageFlowModel.h>
#include <tilemega/Solver/OperatorClasses.h>
#include <tilemega/Target/TargetSpec.h>
#include <mlir/IR/MLIRContext.h>

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <functional>

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: serving_import_test BRIDGE.json decode|prefill [TARGET.json]\n";
    return 2;
  }
  tilemega::analysis::IslContext isl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<tilemega::dialect::CGDialect>();
  context.getOrLoadDialect<tilemega::dialect::ExecDialect>();
  auto bridge = tilemega::frontend::ReadExportBridge(argv[1]);
  std::cerr << "SERVING_IMPORT_STEP bridge=" << bridge.nodes.size() << '\n';
  tilemega::frontend::ServingOptions serving;
  serving.phase = std::string(argv[2]) == "prefill"
      ? tilemega::frontend::ServingOptions::Phase::kPrefill
      : tilemega::frontend::ServingOptions::Phase::kDecode;
  serving.seq = serving.phase ==
      tilemega::frontend::ServingOptions::Phase::kDecode ? 1 : 64;
  serving.argmax_tile_n = 128;
  auto plan = tilemega::frontend::BuildModelPlan(
      bridge.nodes, bridge.inputs, bridge.outputs, serving);
  std::cerr << "SERVING_IMPORT_STEP plan=" << plan.stages.size() << '\n';
  tilemega::frontend::ImportOptions options;
  options.gemms.assign(plan.gemms.size(), {16, 128, 64, 2, 1});
  tilemega::frontend::ImportSummary summary;
  tilemega::frontend::TorchExportImporter importer;
  auto imported = importer.ImportSemantics(argv[1], plan, context);
  std::cerr << "SERVING_IMPORT_STEP semantics\n";
  tilemega::analysis::CouplingCache cache;
  auto module = importer.InstantiateForGranularity(
      imported, context, options, &cache, &summary);
  std::cerr << "SERVING_IMPORT_STEP instantiate\n";
  if (!module || summary.task_spaces < plan.stages.size())
    throw std::runtime_error("serving import omitted a stage");
  std::cout << "SERVING_IMPORT phase=" << argv[2]
            << " stages=" << summary.stages
            << " tasks=" << summary.task_spaces
            << " edges=" << summary.couplings
            << " cache_hit=" << cache.hits
            << " cache_miss=" << cache.misses << '\n';
  if (argc == 4) {
    tilemega::solver::ModelDims dims;
    dims.seq = serving.seq;
    dims.batch = 2;
    dims.past = serving.seq == 1 ? 64 : 0;
    dims.total = dims.seq + dims.past;
    auto model = tilemega::solver::ModelDescription::FromCouplingGraph(
        *module, dims, "serving_test");
    std::cerr << "SERVING_IMPORT_STEP model\n";
    auto target = tilemega::TargetSpec::FromJson(argv[3]);
    auto classes=tilemega::solver::BuildOperatorClasses(imported);
    for(std::size_t index=0;index<classes.size();++index) {
      auto domain=tilemega::solver::ServingClassCandidates(
          classes[index],imported,target,dims.batch,dims.seq);
      if(domain.candidates.empty())throw std::runtime_error("empty serving class domain");
      std::cout<<"SERVING_DOMAIN class="<<index<<" raw="<<domain.raw
               <<" r1="<<domain.removed_r1<<" r2="<<domain.removed_r2
               <<" kept="<<domain.candidates.size()<<'\n';
    }
    auto floor = tilemega::solver::DeriveModelDramFloor(
        *module, model, target, "");
    std::cerr << "SERVING_IMPORT_STEP floor\n";
    auto value = floor.Evaluate(model.MetricBindings());
    if (value.floor_ns <= 0) throw std::runtime_error("nonpositive serving floor");
    std::cout << "SERVING_FLOOR batch=2 past=" << dims.past
              << " dram_bytes=" << value.read_bytes + value.write_bytes
              << " floor_ns=" << value.floor_ns << '\n';
    auto problem = tilemega::solver::PrepareSymbolicProblem(
        *module, target, dims, target.res.num_sms, 1, 1, nullptr, false);
    if (problem.counts.size() < plan.stages.size())
      throw std::runtime_error("serving symbolic problem omitted a stage");
    std::cout << "SERVING_SYMBOLIC stages=" << problem.counts.size()
              << " tiles=" << problem.offsets.back() << '\n';
    tilemega::solver::FlowPreparationCache prepared;
    tilemega::solver::HopCurve hop;
    auto flow = tilemega::solver::PrepareFlow(
        problem, floor, target, 1, hop, cache, prepared, true);
    if (flow.flow.spaces.size() != problem.counts.size())
      throw std::runtime_error("serving flow omitted a task space");
    std::cout << "SERVING_FLOW spaces=" << flow.flow.spaces.size()
              << " edges=" << flow.flow.edges.size() << '\n';
    auto predicted=tilemega::solver::EvaluateFlow(flow.flow);
    if (!std::isfinite(predicted.makespan_ns) ||
        predicted.makespan_ns < value.dram_ns)
      throw std::runtime_error("serving flow prediction violates the DRAM floor");
    std::cout << "SERVING_PREDICTION ns=" << predicted.makespan_ns << '\n';
    std::vector<std::pair<double,std::string>> costs;
    for (auto const& space : flow.flow.spaces)
      costs.push_back({space.rank_ns * space.count, space.name});
    std::sort(costs.begin(), costs.end(), std::greater<>{});
    for (std::size_t i = 0; i < std::min<std::size_t>(8, costs.size()); ++i)
      std::cout << "SERVING_COST " << costs[i].second << " ns=" << costs[i].first << '\n';
  }
}
