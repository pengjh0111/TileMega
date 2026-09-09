// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Codegen/RuntimePlan.h>

namespace tilemega::solver {

struct RuntimeProjectionOptions {
  int grid = 0;
  int threads = 0;
  int kappa = 0;
  bool force_all_dependencies = false;
  bool cg_split_task_order = TILEMEGA_CG_SPLIT_TASK_ORDER;
};

struct ProjectedStage {
  int logical_stage = -1;
  bool combine = false;
  analysis::QuasiPolynomial task_count;
};

struct RuntimeProjection {
  std::vector<ProjectedStage> stages;
  analysis::CouplingRelation tasks;
  analysis::CouplingRelation waits;
  analysis::QuasiPolynomial runtime_task_refs;
  analysis::QuasiPolynomial runtime_wait_entries;
  analysis::QuasiPolynomial max_worker_task_refs;
};

/// The current stage-major queues use a bijection of worker labels. Cardinal
/// totals and maximum length are invariant under those permutations; a new
/// task-dependent placement must replace the worker projection explicitly.
RuntimeProjection ProjectRuntimeQueues(ModelDescription const& model,
                                      codegen::RuntimePlan const& plan,
                                      RuntimeProjectionOptions options);

}  // namespace tilemega::solver
