// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §5.1 torch.export ingestion.
//
// The stable facts export_bridge.py writes, parsed and nothing more. Split out
// of the importer so the semantic lifting can be checked against a real
// exported graph without going through MLIR.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <tilemega/Frontend/ModelPlan.h>

namespace tilemega::frontend {

struct ExportBridge {
  std::vector<FxNodeRecord> nodes;  ///< every FX node, in order
  std::vector<FxNodeRecord> tasks;  ///< the call_function subset
  std::vector<SignatureInput> inputs;
  std::vector<std::string> outputs;
  std::unordered_map<std::string, std::string> range_texts;
  std::vector<std::string> guards;
  /// call_function targets no classification rule covers, sorted and unique.
  std::vector<std::string> unsupported;
};

ExportBridge ReadExportBridge(std::string const& path);

}  // namespace tilemega::frontend
