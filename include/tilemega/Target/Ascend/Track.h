// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <optional>
#include <map>
#include <string>
#include <vector>
#include <tilemega/Codegen/tasks/TaskBase.h>

namespace tilemega::ascend {
enum class TrackRole { kAic, kAiv };
enum class ImplementationRole { kAicOnly, kAivOnly, kMix1_2 };
enum class LaunchMode { kAicOnly, kAivOnly, kMix1_2 };
enum class HandoffRoute { kGmEvent, kLocalL0cToUb, kLocalUbToL1, kLocalControl };
enum class SyncKind { kUnverified, kGroupFlag, kStageBarrier, kGmEvent };
enum class CostLane { kAicCube, kAivVector, kFixpipe, kGm, kL2, kSync };

struct LocalResources {
  std::uint64_t l1 = 0, l0a = 0, l0b = 0, l0c = 0, ub = 0, registers = 0, ssbuffer = 0;
};
struct TrackClass {
  std::string name;
  TrackRole role = TrackRole::kAic;
  int count = 0;
  LocalResources local_resources;
  std::vector<ImplementationRole> available_impl_types;
  std::string grouping;
};
struct TargetCapability {
  std::string arch_tag, product_sku, cann_version, catlass_sha;
  std::string gm_event_evidence;
  std::vector<TrackClass> tracks;
  std::map<std::string, codegen::TaskKind> implementations;
  int max_verified_mix_groups = 0;
  bool aic_only_kernel = false, aiv_only_kernel = false, mix_1_2 = false;
  bool group_flag = false, cross_group_stage_barrier = false, gm_event = false;
  bool same_group_l0c_to_ub = false, same_group_ub_to_l1 = false, ssbuffer_control = false;
  int TrackCount(TrackRole role) const;
  LocalResources const& Budget(TrackRole role) const;
  void Validate() const;
};
struct LaunchTopology {
  LaunchMode mode = LaunchMode::kMix1_2;
  int num_groups = 0;
  int num_blocks = 0;
  int residency = 1;
};
struct WorkerPlacement {
  int global_worker_id = -1, group = -1;
  TrackRole role = TrackRole::kAic;
  int local_block_id = -1, sub_block_id = 0;
};
struct WorkerMap {
  LaunchTopology launch;
  std::vector<WorkerPlacement> workers;
  static WorkerMap Build(TargetCapability const&, LaunchMode, int blocks);
  WorkerPlacement const& At(int worker) const;
  std::vector<int> Domain(ImplementationRole role) const;
  void Validate(TargetCapability const&, std::vector<std::uint32_t> const& offsets,
                std::uint32_t task_refs) const;
};
struct TaskImplementation {
  codegen::TaskKind kind = codegen::TaskKind::kGemm;
  std::string impl_id, arch_tag = "ascend_3510", input_layout, output_layout;
  ImplementationRole required_role = ImplementationRole::kAicOnly;
  LocalResources aic_scratch, aiv_scratch;
  std::vector<unsigned> internal_flags;
  bool calibrated = false;
  void Validate(TargetCapability const&) const;
};
struct HandoffContract {
  HandoffRoute route = HandoffRoute::kGmEvent;
  SyncKind sync = SyncKind::kUnverified;
  int producer = -1, consumer = -1;
  std::uint64_t workspace_bytes = 0;
  std::string layout, visibility_evidence;
  unsigned buffer_epochs = 0;
  void Validate(TargetCapability const&, WorkerMap const&) const;
};

// Composite tasks occur in all participating queues; only owner publishes.
struct TaskAssignment {
  int owner = -1;
  TaskImplementation implementation;
  std::vector<int> participants;
  std::vector<unsigned> dependencies, publishes;
};
struct PlanAudit {
  std::vector<unsigned> topological_order, event_triggers;
  std::uint64_t fifo_edges = 0, dependency_edges = 0;
};
PlanAudit ValidatePlan(TargetCapability const&, WorkerMap const&,
                       std::vector<TaskAssignment> const&,
                       std::vector<std::vector<unsigned>> const& queues,
                       std::vector<unsigned> const& expected_event_triggers = {});
int Home(std::vector<int> const& legal_domain, int base, int tile);
std::vector<int> Spread(std::vector<int> const& legal_domain, int base, int tile, int width);
void ValidateColocation(WorkerMap const&, int producer, int consumer, bool same_worker_identity);
std::optional<double> SharedGmTimeNs(std::vector<std::uint64_t> const& task_bytes,
                                   std::optional<double> device_bytes_per_ns);
} // namespace tilemega::ascend
