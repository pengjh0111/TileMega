// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.1 torch.export ingestion (Phase 1 stub).
#pragma once
#include <string>
namespace tilemega::frontend {
struct ImportedProgram { std::string textual_ir; };
class TorchExportImporter {
 public:
  ImportedProgram Import(std::string const& exported_program_path) const;
};
}  // namespace tilemega::frontend
