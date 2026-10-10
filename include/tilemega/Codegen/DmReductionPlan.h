// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/DmReductionProof.h>
#include <tilemega/Codegen/tasks/ModelRuntime.h>

namespace tilemega::codegen {
inline DmReductionPlan BuildDmReductionPlan(std::vector<StageDesc>& stages,
    std::vector<StageDependency> const& dependencies,std::vector<unsigned> const& tasks,
    RuntimeDependencyInterval const* intervals,bool pooling,bool moe,unsigned moe_mask=3) {
  auto visit=[&](auto const& dep,unsigned c,unsigned p,auto const& callback) {
    return VisitStageDependencyIntervals(dep,intervals,c,p,callback);
  };
  return BuildDmReductionProof(stages,dependencies,tasks,visit,pooling,moe,moe_mask);
}
} // namespace tilemega::codegen
