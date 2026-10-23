// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/LlmPlanABI.h>
#include <tilemega/Solver/AscendSchedule.h>
#include <mlir/IR/BuiltinOps.h>
#include <filesystem>

namespace tilemega::codegen {
struct AscendLlmPlan {
  ascend::llm::Metadata metadata{};
  std::vector<ascend::llm::Stage> stages;
  std::vector<ascend::llm::Task> tasks;
  std::vector<ascend::llm::Wait> waits;
  std::vector<ascend::llm::Allocation> allocations;
  std::vector<uint32_t> offsets,queue;
  std::vector<solver::AscendServicePrice> task_prices;
  std::string buffers_json;
  void Write(std::filesystem::path const&) const;
};
AscendLlmPlan MaterializeAscendLlm(mlir::ModuleOp,solver::SymbolicProblem const&,
    ascend::WorkerMap const&,std::vector<solver::AscendScheduledTask> const&,
    solver::AscendSchedule const&,bool diagnostic,unsigned attention_precision=0,bool local_tail_k7=false);
std::string LowerAscendLlm(AscendLlmPlan const&);
}
