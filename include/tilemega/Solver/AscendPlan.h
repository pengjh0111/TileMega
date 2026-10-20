// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Target/Ascend/PlanABI.h>
#include <map>

namespace tilemega::solver {
struct AscendTaskHandoff {
  unsigned producer_task=0,consumer_task=0;
  ascend::HandoffContract contract;
};
struct AscendPlan {
  ascend::abi::Metadata metadata{};
  ascend::WorkerMap workers;
  std::vector<ascend::abi::TaskRef> tasks;
  std::vector<ascend::abi::TaskMeta> task_metadata;
  std::vector<uint32_t> schedule_offsets,schedule,dependencies;
  std::vector<ascend::TaskImplementation> stage_implementations;
  std::vector<AscendTaskHandoff> handoffs;
  ascend::PlanAudit audit;
  std::map<std::string,std::uint64_t> storage_bytes;
  bool stage_barrier=false;
  void SelectStageBarrier();
  std::string ToJson() const;
};
AscendPlan MaterializeAscendPlan(SymbolicProblem const&,PlanSkeleton const&,
    ascend::TargetCapability const&,std::vector<ascend::TaskImplementation> const&,
    unsigned epochs=1);
} // namespace tilemega::solver
