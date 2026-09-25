// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/CouplingCache.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Solver/ModelDramFloor.h>
#include <tilemega/Target/TargetSpec.h>
#include <mlir/IR/MLIRContext.h>

#include <iostream>
#include <stdexcept>

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
    auto floor = tilemega::solver::DeriveModelDramFloor(
        *module, model, target, "");
    std::cerr << "SERVING_IMPORT_STEP floor\n";
    auto value = floor.Evaluate(model.MetricBindings());
    if (value.floor_ns <= 0) throw std::runtime_error("nonpositive serving floor");
    std::cout << "SERVING_FLOOR batch=2 past=" << dims.past
              << " dram_bytes=" << value.read_bytes + value.write_bytes
              << " floor_ns=" << value.floor_ns << '\n';
  }
}
