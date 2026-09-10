// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Analysis/OpArithmetic.h>
#include <tilemega/Analysis/FusionAccess.h>
#include <tilemega/Codegen/tasks/ScalarDataflow.h>

namespace tilemega::solver {
#ifndef TILEMEGA_SCALAR_TASK_WORK
#define TILEMEGA_SCALAR_TASK_WORK 1
#endif
analysis::OperatorGraph InstantiateModelTasks(ModelDescription const& model,
                                            std::vector<GemmConfig> const& configs);
std::vector<ModelCouplingMetrics> InstantiateModelCouplings(
    ModelDescription const& model,std::vector<GemmConfig> const& configs,
    std::optional<std::pair<int,int>> stage_pair=std::nullopt);
struct RuntimeScalarAccess {
  analysis::CouplingRelation ownership,writes;
  std::map<std::string,analysis::CouplingRelation> reads;
};
struct DerivedTaskInput {
  analysis::OperatorNode task;
  analysis::TaskWork work;
  analysis::OpArithmetic arithmetic;
  std::vector<std::string> cost_coordinates;
  std::optional<codegen::ScalarDataflow> scalar_flow;
  std::optional<RuntimeScalarAccess> scalar_access;
};
analysis::TaskAccesses DeriveModelTaskAccesses(ModelTaskSemantics const& semantic,
                                             DerivedTaskInput const& input);
struct ModelFusionCandidate {
  DerivedTaskInput producer, consumer;
  analysis::FusionAccesses accesses;
  analysis::MixedArithmetic arithmetic;
  analysis::TaskAccesses producer_accesses,consumer_accesses;
};
// L-task candidates are identified by semantic names, not runtime stage IDs.
// Runtime ownership is a later projection and cannot define fusion legality.
ModelFusionCandidate DeriveLogicalFusionCandidate(ModelDescription const& model,
    std::vector<GemmConfig> const& configs, std::string const& producer,
    std::string const& consumer);
ModelFusionCandidate DeriveModelFusionCandidate(ModelDescription const& model,
    std::vector<GemmConfig> const& configs, int producer_stage, int consumer_stage);
analysis::TaskWork DeriveRuntimeScalarWork(ModelDescription const& model,
    ModelTaskSemantics const& semantic,analysis::OperatorNode const& task,
    analysis::TaskWork work,int threads,RuntimeScalarAccess* accesses=nullptr);
DerivedTaskInput DeriveModelTaskInput(ModelDescription const& model,
                                    ModelTaskSemantics const& semantic,
                                    analysis::OperatorGraph const& graph,
                                    GemmConfig const* config,
                                    bool runtime_ownership=true);
}  // namespace tilemega::solver
