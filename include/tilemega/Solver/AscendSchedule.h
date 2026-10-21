// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/Track.h>
#include <cstdint>
#include <string>
#include <vector>

namespace tilemega::solver {
struct SymbolicProblem;
struct PlanSkeleton;
struct AscendServicePrice {
  double service_ns = 0;
  std::uint64_t gm_bytes = 0;
  std::string provenance;
};
struct AscendScheduledTask {
  std::uint32_t stage = 0, logical_task = 0;
  ascend::ImplementationRole role = ascend::ImplementationRole::kAicOnly;
  AscendServicePrice price;
  std::vector<std::uint32_t> predecessors;
  std::vector<int> legal_workers;
  unsigned mix_aiv_lanes = 2;
};
struct AscendSchedulingCosts {
  double gm_bytes_per_ns = 0, cross_worker_ready_ns = 0;
  std::string provenance;
};
struct AscendFrontierWait {
  std::uint32_t producer_worker = 0, required_prefix = 0;
};
struct AscendSchedule {
  std::vector<std::uint32_t> owners, ordinals, offsets, queue;
  std::vector<std::vector<std::uint32_t>> participants;
  std::vector<std::vector<AscendFrontierWait>> waits;
  std::vector<double> start_ns, finish_ns;
  std::uint64_t semantic_edges = 0, fifo_edges = 0, lowered_waits = 0;
  double makespan_ns = 0, gm_floor_ns = 0;
  std::uint32_t epoch_stride = 0;
  std::string ToJson() const;
};

// Body service time already includes its local copies. The global GM calendar
// is a shared throughput constraint, not another copy of service time per role.
AscendSchedule ScheduleAscendTasks(std::vector<AscendScheduledTask> const&,
    ascend::WorkerMap const&, AscendSchedulingCosts const&);
// Bootstrap execution/calibration only. No fabricated service times or claimed
// optimization: canonical topological order and least-populated legal queues.
AscendSchedule DiagnosticAscendSchedule(std::vector<AscendScheduledTask> const&,
    ascend::WorkerMap const&);
void AuditAscendSchedule(std::vector<AscendScheduledTask> const&,
    ascend::WorkerMap const&, AscendSchedule const&);
std::uint32_t AscendProgressValue(std::uint64_t request_forward_epoch,
                                 std::uint32_t prefix, std::uint32_t stride);
std::vector<AscendScheduledTask> AscendTasksFromSymbolic(SymbolicProblem const&,
    PlanSkeleton const&, std::vector<AscendServicePrice> const&);
} // namespace tilemega::solver
