// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.1 torch.export ingestion (Phase 1 stub).
#pragma once
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <string>
#include <vector>
namespace tilemega::frontend {
struct ImportSummary {
  std::size_t task_spaces = 0;
  std::size_t couplings = 0;
  std::size_t stages = 0;
  std::size_t guards = 0;
  /// Operators no classification rule covers. They import as one generic task
  /// space each rather than being rejected.
  std::vector<std::string> degraded;
};
class TorchExportImporter {
 public:
  mlir::OwningOpRef<mlir::ModuleOp> Import(
      std::string const& stable_json_path, mlir::MLIRContext& context,
      ImportSummary* summary = nullptr) const;
};
}  // namespace tilemega::frontend
