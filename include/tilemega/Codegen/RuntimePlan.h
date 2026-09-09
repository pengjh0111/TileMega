// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/DependencyForm.h>
#include <tilemega/Codegen/RuntimeOwnership.h>
#include <cstdint>
#include <vector>

namespace mlir { class ModuleOp; }
namespace tilemega::codegen {

struct DependencyRecord {
  std::uint32_t producer;
  std::uint32_t consumer;
  analysis::WaitWindow window;
};

struct GemmRuntimeRecord {
  std::uint16_t compiled_variant = 0;
  std::uint16_t split_k = 0;
  std::uint16_t tile_m = 0;
  std::uint16_t tile_n = 0;
  std::uint16_t tile_k = 0;
  std::uint16_t stages = 0;
};

/// Shared semantic-to-runtime seed, before host split rewrite. Codegen and
/// symbolic queue counting must use the same merged/reduced dependencies.
struct RuntimePlan {
  std::vector<DependencyRecord> dependencies;
  std::vector<GemmRuntimeRecord> gemms;
  std::uint32_t ownership_flags = 0;
  int cluster_dim = 0;
};

RuntimePlan ReadRuntimePlan(mlir::ModuleOp module);

}  // namespace tilemega::codegen
