// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Frontend/SymbolicShapeBridge.h>

#include <mlir/IR/MLIRContext.h>

#include <cassert>
#include <set>
#include <stdexcept>
#include <string>

int main() {
  mlir::MLIRContext context;
  tilemega::frontend::ImportSummary summary;
  auto module = tilemega::frontend::TorchExportImporter{}.Import(
      std::string(TILEMEGA_SOURCE_DIR) +
          "/docs/experiments/E2E_GEN/raw/export_bridge.json",
      context, &summary);
  // Operator granularity: one task space per L-task node of the instantiated
  // OperatorGraph (34 for the two-layer fixture), one coupling per
  // (consumer, in-graph operand) pair.  The previous 179/222 counted FX nodes
  // and the placeholder edges between them.
  assert(summary.task_spaces == 34 && summary.couplings == 42);
  assert(summary.stages == 30 && summary.guards == 4);
  assert(module->getOperation()->getAttr("tilemega.model_plan"));
  auto aliases = module->getOperation()->getAttrOfType<mlir::DictionaryAttr>(
      "tilemega.symbol_aliases");
  assert(aliases.getAs<mlir::StringAttr>("s61").getValue() == "s14");
  assert(aliases.getAs<mlir::StringAttr>("s65").getValue() == "s14");
  // A task space now names the role the semantic lifting recognised, not the
  // FX target it came from, so `view`/`transpose` no longer appear as kinds.
  // What replaces that coverage: every recognised role must be present and
  // none may have degraded to `generic` on a model the frontend claims to
  // cover.
  std::set<std::string> kinds;
  for (auto task : module->getOps<tilemega::dialect::TaskSpaceOp>()) {
    auto kind = task.getWriteMap().getFields().getAs<mlir::StringAttr>("kind").getValue();
    assert(kind != "generic");
    kinds.insert(kind.str());
  }
  assert((kinds == std::set<std::string>{"attention", "elementwise", "gemm",
                                         "kvappend", "rmsnorm", "rope"}));
  // The gap this whole change closes: C used to be the constant
  // `{ [0] -> [0] }` for every edge.  No edge may carry a one-point relation.
  for (auto coupling : module->getOps<tilemega::dialect::CouplingOp>()) {
    std::string relation = coupling.getRelation().getMap().ToString();
    assert(relation.find("[0] -> [0]") == std::string::npos);
  }

  std::string cuda =
      tilemega::codegen::CouplingGraphToCUDA{}.Lower(*module);
  assert(cuda.find("constexpr ModelSpec kModel") != std::string::npos);
  assert(cuda.find("kStages, 30u") != std::string::npos);
  assert(cuda.find("TILEMEGA_GENERATED_WAIT_global") != std::string::npos);
  assert(cuda.find("TILEMEGA_GENERATED_NOTIFY_global") != std::string::npos);
  assert(cuda.find("TILEMEGA_GENERATED_RESIDENT_GRID") != std::string::npos);
  assert(cuda.find("ModelHarness.cuh") != std::string::npos);
  assert(cuda.find("GeneratedLlamaRuntime.cuh") == std::string::npos);
  assert(cuda.find("% 12") == std::string::npos);

  // P4.7: the cluster shape is a property of the whole launch, so the
  // generator's contract is all-or-nothing.  Flipping every coupling and every
  // placement together must produce a clustered kernel; flipping only one side
  // must be rejected rather than silently resolved to the other.
  {
    auto clustered = module->clone();
    auto sync = tilemega::dialect::SyncKindAttr::get(
        &context, mlir::StringAttr::get(&context, "cluster"));
    for (auto coupling : clustered.getOps<tilemega::dialect::CouplingOp>())
      coupling.setSyncKindAttr(sync);
    std::string half;
    try {
      half = tilemega::codegen::CouplingGraphToCUDA{}.Lower(clustered);
    } catch (std::invalid_argument const&) {
      half = "rejected";
    }
    assert(half == "rejected");
    for (auto placement : clustered.getOps<tilemega::dialect::PlacementOp>())
      placement.setCluster(2);
    std::string whole = tilemega::codegen::CouplingGraphToCUDA{}.Lower(clustered);
    assert(whole.find("#define TILEMEGA_GENERATED_CLUSTER_DIM 2") !=
           std::string::npos);
    // The flat kernel must keep saying nothing about clusters, so a default
    // build cannot pick the macro up by accident.
    assert(cuda.find("TILEMEGA_GENERATED_CLUSTER_DIM") == std::string::npos);
    clustered->erase();
  }

  // An uncovered operator degrades instead of being rejected; `degradation_test`
  // holds the rest of that contract.
  tilemega::frontend::ImportSummary degraded;
  (void)tilemega::frontend::TorchExportImporter{}.Import(
      std::string(TILEMEGA_SOURCE_DIR) +
          "/test/fixtures/export_unsupported.json", context, &degraded);
  assert(degraded.degraded.size() == 1 &&
         degraded.degraded.front() == "aten.imaginary.default");

  auto domain = tilemega::frontend::SymbolicShapeBridge{}.Parse(
      {{"s0", "VR[1, 1024]"}},
      {"L['flat_args'][0].size()[0] % 128 == 0",
       "L['flat_args'][0].size()[0] <= 512"},
      {{"s0"}});
  assert(domain.constraints.size() == 2);
  assert(domain.constraints[0].predicate ==
         tilemega::frontend::ShapeConstraint::Predicate::kDivisible);
  assert(domain.constraints[1].predicate ==
         tilemega::frontend::ShapeConstraint::Predicate::kLessEqual);
  return 0;
}
