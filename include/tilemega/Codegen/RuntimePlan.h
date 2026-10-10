// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/DependencyForm.h>
#include <tilemega/Analysis/DependencyTable.h>
#include <tilemega/Analysis/CountedDependencyForm.h>
#include <tilemega/Codegen/RuntimeOwnership.h>
#include <tilemega/Codegen/AttentionPlan.h>
#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include <optional>

namespace mlir { class ModuleOp; }
namespace tilemega::codegen {

struct CountedWaitRecord {
  analysis::CountedDependencyForm contributions;
  // I2 is used for queue ordering; counters wait for logical contributions.
  analysis::CouplingRelation conservative_relation;
  std::uint32_t producers = 0;
  std::string tensor;
  std::vector<unsigned> unit_axes;
};

struct DependencyRecord {
  std::uint32_t producer;
  std::uint32_t consumer;
  analysis::WaitWindow window;
  std::optional<analysis::WaitWindow> phase_window;
  int phase_tiles = 0;
  std::optional<analysis::DependencyTable> table;
  std::optional<CountedWaitRecord> counted;
  bool producer_main = false, consumer_done = false;
};

struct GemmRuntimeRecord {
  std::uint16_t compiled_variant = 0;
  std::uint16_t split_k = 0;
  std::uint16_t tile_m = 0;
  std::uint16_t tile_n = 0;
  std::uint16_t tile_k = 0;
  std::uint16_t stages = 0;
  std::uint16_t impl = 0;
};

/// Shared semantic-to-runtime seed, before host split rewrite. Codegen and
/// symbolic queue counting must use the same merged/reduced dependencies.
struct RuntimePlan {
  std::vector<DependencyRecord> dependencies;
  std::vector<GemmRuntimeRecord> gemms;
  std::vector<AttentionRuntimeRecord> attention;  ///< original-stage indexed; empty is direct
  std::uint32_t ownership_flags = 0;
  int cluster_dim = 0;
  bool resident_only = true;
  std::map<std::string,std::pair<long,long>> parameter_ranges;
  std::map<std::string,std::uint32_t> task_stages;
  analysis::ParamBinding task_binding;
};

RuntimePlan ReadRuntimePlan(mlir::ModuleOp module);
/// Original phase schedule retained by transactional L-task fusion. This is
/// input to the replacement projection, never the final fused schedule.
RuntimePlan ReadFusionSourcePlan(mlir::ModuleOp module);
std::vector<AttentionRuntimeRecord> ReadAttentionRuntime(mlir::ModuleOp module);

}  // namespace tilemega::codegen
