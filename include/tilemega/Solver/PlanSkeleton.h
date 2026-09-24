// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingCache.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/SolverTiming.h>
#include <tilemega/Solver/TaskModel.h>
#include <mlir/IR/BuiltinOps.h>

namespace tilemega::solver {
struct SymbolicProblem {
  codegen::RuntimePlan runtime;
  ModelDescription model;
  std::vector<GemmConfig> geometry;
  RuntimeProjection projection;
  std::vector<int> counts,offsets;
  std::vector<double> task_ns,prefetch_ns;
  int threads=0;
};
SymbolicProblem PrepareSymbolicProblem(mlir::ModuleOp module,TargetSpec const& target,
    ModelDims dims,int grid,int residency,int kappa);
struct SkeletonSpace {
  int count=0,offset=0,base=0,width=0,order=0;
  double task_ns=0,load_ns=0;
};
struct SkeletonEdge {
  int producer=0,consumer=0;
  std::shared_ptr<analysis::OraclePair> oracle;
  bool all_producer=false;
};
struct PlanSkeleton {
  int grid=0,residency=0;
  std::vector<SkeletonSpace> spaces;
  std::vector<SkeletonEdge> edges;
  std::vector<std::vector<int>> incoming,outgoing;
  std::vector<int> stage_order;
  std::vector<double> task_ns;
  analysis::ParamBinding theta;
  std::vector<int> Spread(int stage,int tile) const;
};
PlanSkeleton BuildPlanSkeleton(SymbolicProblem const& problem,int grid,int residency,
    int k_base,bool all_workers,analysis::CouplingCache& cache,SolverTiming* timing=nullptr);
void WritePlanSkeleton(mlir::ModuleOp module,PlanSkeleton const& skeleton);
}
