// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#include <tilemega/Codegen/DmReductionPlan.h>
#include <cassert>

using namespace tilemega::codegen;
StageDesc Stage(TaskKind kind) {StageDesc stage{};stage.kind=kind;return stage;}
StageDependency All(unsigned p,unsigned c) {return {p,c,StageDependency::Map::kAll,1,0,0,1};}
StageDependency Identity(unsigned p,unsigned c) {return {p,c,StageDependency::Map::kWindow,1,1,0,1};}
int main() {
  {
    auto stages=std::vector<StageDesc>{Stage(TaskKind::kGemm),Stage(TaskKind::kGlobalPoolReduce)};
    RuntimeDependencyInterval rows[]={{0,2},{1,2}};
    StageDependency dep{0,1,StageDependency::Map::kTable,1,1,0,1};dep.table_rows=2;dep.table_stride=1;
    auto plan=BuildDmReductionPlan(stages,{dep},{3,2},rows,true,true);
    assert(plan.selected==1 && plan.tickets==2 && stages[1].handoff_elided);
    assert(stages[0].dm_reduce_stage==1 && plan.arrivals.size()==4);
    assert(plan.offsets[0]==0 && plan.offsets[1]==1 && plan.offsets[2]==3 && plan.offsets[3]==4);
    for(auto a:plan.arrivals)assert(a.expected==2);
  }
  {
    auto stages=std::vector<StageDesc>{Stage(TaskKind::kGemm),Stage(TaskKind::kMoETopK),
        Stage(TaskKind::kMoETopK),Stage(TaskKind::kMoETopK),Stage(TaskKind::kMoETopK)};
    stages[1].moe.step=DmMoeStep::kSelect;stages[2].moe.step=DmMoeStep::kHistogram;
    stages[3].moe.step=DmMoeStep::kPrefix;stages[4].moe.step=DmMoeStep::kScatter;
    auto plan=BuildDmReductionPlan(stages,{Identity(0,1),Identity(1,2),All(2,3),
        All(3,4),Identity(1,4)},{2,2,2,1,2},nullptr,false,true);
    assert(plan.selected==4);
    for(unsigned i=1;i<5;++i)assert(stages[i].handoff_elided && stages[i-1].dm_reduce_stage==i);
  }
  {
    auto stages=std::vector<StageDesc>{Stage(TaskKind::kGemm),Stage(TaskKind::kMoETopK),
        Stage(TaskKind::kGemm),Stage(TaskKind::kGemm),Stage(TaskKind::kMoECombine)};
    // Dispatch has no incoming router here, so is independently queued.
    stages[1].moe.step=DmMoeStep::kScatter;
    StageDependency counted{3,4,StageDependency::Map::kCounted,1,0,0,8};counted.table_rows=2;
    auto plan=BuildDmReductionPlan(stages,{All(0,2),All(1,2),Identity(2,3),
        All(0,4),All(1,4),counted},{1,1,4,4,2},nullptr,false,true);
    assert(plan.selected==1 && stages[4].handoff_elided && plan.stages[3].counted_edge==5);
    assert(plan.tickets==2);
    for(unsigned mask:{0u,1u,2u,3u}) {
      stages[3].dm_reduce_stage=kDmNoIndex;stages[4].handoff_elided=false;
      auto selected=BuildDmReductionPlan(stages,{All(0,2),All(1,2),Identity(2,3),
          All(0,4),All(1,4),counted},{1,1,4,4,2},nullptr,false,true,mask);
      assert(selected.selected==unsigned(bool(mask&2)));
    }
  }
  {
    auto stages=std::vector<StageDesc>{Stage(TaskKind::kGemm),Stage(TaskKind::kGemm),
        Stage(TaskKind::kGlobalPoolReduce)};
    auto plan=BuildDmReductionPlan(stages,{All(0,2),All(1,2)},{1,1,1},nullptr,true,false);
    assert(!plan.selected && !stages[2].handoff_elided);
    // An independent extra input cannot be dismissed by topological order.
    auto disabled=BuildDmReductionPlan(stages,{All(0,2)},{1,1,1},nullptr,false,false);
    assert(!disabled.selected);
  }
  return 0;
}
