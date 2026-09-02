// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Frontend/SymbolicShapeBridge.h>

#include <mlir/IR/MLIRContext.h>

#include <cassert>
#include <stdexcept>
#include <string>

int main() {
  mlir::MLIRContext context;
  tilemega::frontend::ImportSummary summary;
  auto module = tilemega::frontend::TorchExportImporter{}.Import(
      std::string(TILEMEGA_SOURCE_DIR) +
          "/docs/experiments/E2E_GEN/raw/export_bridge.json",
      context, &summary);
  assert(summary.task_spaces == 179 && summary.couplings == 222);
  assert(summary.stages == 30 && summary.guards == 4);
  assert(module->getOperation()->getAttr("tilemega.model_plan"));
  auto aliases = module->getOperation()->getAttrOfType<mlir::DictionaryAttr>(
      "tilemega.symbol_aliases");
  assert(aliases.getAs<mlir::StringAttr>("s61").getValue() == "s14");
  assert(aliases.getAs<mlir::StringAttr>("s65").getValue() == "s14");
  bool sawView = false, sawTranspose = false;
  for (auto task : module->getOps<tilemega::dialect::TaskSpaceOp>()) {
    auto kind = task.getWriteMap().getFields().getAs<mlir::StringAttr>("kind").getValue();
    sawView |= kind == "view";
    sawTranspose |= kind == "transpose";
  }
  assert(sawView && sawTranspose);

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

  bool rejected = false;
  try {
    (void)tilemega::frontend::TorchExportImporter{}.Import(
        std::string(TILEMEGA_SOURCE_DIR) +
            "/test/fixtures/export_unsupported.json", context);
  } catch (std::runtime_error const& error) {
    rejected = std::string(error.what()).find("aten.imaginary.default") !=
               std::string::npos;
  }
  assert(rejected);

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
