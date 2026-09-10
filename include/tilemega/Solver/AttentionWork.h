// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Codegen/AttentionPlan.h>

#ifndef TILEMEGA_ATTENTION_PHASE_WORK
#define TILEMEGA_ATTENTION_PHASE_WORK 1
#endif

namespace tilemega::solver {
struct AttentionPhaseWork {
  codegen::AttentionPhase phase;
  analysis::CouplingRelation tasks;
  analysis::TaskAccesses accesses;
  std::map<std::string,int> element_bytes;
  analysis::QuasiPolynomial task_count,read_bytes,write_bytes,flops,transcendental;
  codegen::ScalarDataflow flow;
  int shared_bytes=0;
};
struct AttentionCostPlan {
  std::vector<codegen::AttentionRuntimeRecord> choices;
  std::map<int,std::vector<AttentionPhaseWork>> stages;
  analysis::QuasiPolynomial workspace_bytes;
  int shared_bytes=0;
};
void ApplyAttentionCostPlan(ModelDescription& model,
    std::vector<codegen::AttentionRuntimeRecord> const& choices,int threads);
// Original CG semantic dimensions supply all tensor extents; chunk is a
// candidate granularity. FP32 scratch and model-dtype tensors stay distinct.
std::vector<AttentionPhaseWork> DeriveAttentionPhaseWork(
    ModelDescription const& model,ModelTaskSemantics const& semantic,
    codegen::AttentionRuntimeRecord choice,int threads);
}  // namespace tilemega::solver
