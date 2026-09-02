// SPDX-License-Identifier: BSD-3-Clause
//
// Skeleton §0.1: an operator no rule covers degrades to one conservative task
// space and is reported; it does not stop the import.
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Frontend/TorchExportImporter.h>

#include <mlir/IR/MLIRContext.h>

#include <cassert>
#include <cstdio>
#include <string>

int main() {
  mlir::MLIRContext context;
  tilemega::frontend::ImportSummary summary;
  auto module = tilemega::frontend::TorchExportImporter{}.Import(
      std::string(TILEMEGA_SOURCE_DIR) + "/test/fixtures/export_unsupported.json",
      context, &summary);
  assert(summary.degraded.size() == 1);
  assert(summary.degraded.front() == "aten.imaginary.default");
  assert(summary.task_spaces == 2);  // one per operator, nothing grouped
  assert(summary.stages == 0);       // no decoder layer, so no model plan
  assert(!module->getOperation()->getAttr("tilemega.model_plan"));

  int generic = 0, spaces = 0;
  for (auto task : module->getOps<tilemega::dialect::TaskSpaceOp>()) {
    ++spaces;
    generic += task.getWriteMap().getFields().getAs<mlir::StringAttr>("kind")
                   .getValue() == "generic";
  }
  assert(spaces == 2 && generic == 1);
  std::printf("DEGRADED ops=%zu task_spaces=%d generic=%d\n",
              summary.degraded.size(), spaces, generic);
  return 0;
}
