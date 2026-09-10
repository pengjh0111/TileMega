// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Analysis/OpArithmetic.h>

namespace tilemega::solver {
analysis::OperatorGraph InstantiateModelTasks(ModelDescription const& model,
                                            std::vector<GemmConfig> const& configs);
struct DerivedTaskInput {
  analysis::OperatorNode task;
  analysis::TaskWork work;
  analysis::OpArithmetic arithmetic;
};
DerivedTaskInput DeriveModelTaskInput(ModelDescription const& model,
                                    ModelTaskSemantics const& semantic,
                                    analysis::OperatorGraph const& graph,
                                    GemmConfig const* config);
}  // namespace tilemega::solver
