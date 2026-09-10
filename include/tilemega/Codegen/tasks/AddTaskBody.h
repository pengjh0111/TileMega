// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
#include <tilemega/Codegen/tasks/Placement.cuh>
#include <tilemega/Codegen/tasks/TaskResources.h>

namespace tilemega::codegen {
template <class Arch,class SmemUnion,int Threads>
struct AddTaskBody {
  using ResourceTraits=SimtTaskResources<TaskKind::kAdd,Threads>;
  using SharedStorage=typename ResourceTraits::SharedStorage;
  static constexpr int kSmemBytes=sizeof(SharedStorage);
  static constexpr int kNumThreads=Threads,kStages=0;
  static constexpr bool kLegal=true;
  __device__ static TaskOwnership Ownership(Params const& p,StageDesc const& stage) {
    auto const& g=static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
    return {OwnershipOf(TaskKind::kAdd),g.tiles_m*g.tiles_n};
  }
  __device__ static void RunTask(Params const& p,StageDesc const& stage,SmemUnion&,int task) {
    auto const& g=static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
    int m0=(task/g.tiles_n)*g.tile_m,n0=(task%g.tiles_n)*g.tile_n;
    for (int offset=threadIdx.x;offset<g.tile_m*g.tile_n;offset+=blockDim.x) {
      int m=m0+offset/g.tile_n,n=n0+offset%g.tile_n;
      if (m>=p.dims.seq || n>=int(stage.extent)) continue;
      int index=m*int(stage.extent)+n;
      p.buffers[stage.operand[2]][index]=ModelElement(
          float(p.buffers[stage.operand[0]][index])+float(p.buffers[stage.operand[1]][index]));
    }
  }
  __device__ void operator()(Params const& p,StageDesc const& stage,SmemUnion& smem) const {
    for (int task=PlacedBlock();task<Ownership(p,stage).count;task+=gridDim.x)
      RunTask(p,stage,smem,task);
  }
};
}  // namespace tilemega::codegen
