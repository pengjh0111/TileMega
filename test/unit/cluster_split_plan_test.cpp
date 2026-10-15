// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/executor/ClusterPlacement.h>
#include <tilemega/Solver/PlanMaterialize.h>
#include <cassert>
#include <cstdio>
#include <numeric>

int main() {
  using namespace tilemega;
  unsigned checks=0;
  for(unsigned c:{2u,4u})for(int clusters:{1,3,7})for(int tiles:{1,2,8,23}) {
    solver::PlanRequest r;
    r.mode=dialect::PlacementMode::kTemplate;
    r.params={std::int64_t(dialect::PlacementTemplate::kClusterSplit)};
    r.grid=c*clusters;r.counts={3,tiles*int(c),2,(tiles+1)*int(c)};
    r.stage_order={0,1,2,3};r.cluster_dim=c;r.cluster_chunks={0,c,0,c};
    r.physical_worker.resize(r.grid);std::iota(r.physical_worker.begin(),r.physical_worker.end(),0);
    auto graph=codegen::MaterializeRuntimeTaskGraph(r.counts,{},r.grid);r.graph=&graph;
    solver::MaterializedPlan p;std::string error;
    assert(solver::MaterializePlanPlacement(r,&p,&error));
    assert(solver::CheckPlanLegality(graph,p,&error));
    struct Task{unsigned stage,logical_task;};
    std::vector<Task> tasks;std::vector<std::uint32_t> offsets={0};
    for(auto const& queue:p.queue) {
      for(auto const& item:queue)tasks.push_back({item.stage,unsigned(item.logical)});
      offsets.push_back(tasks.size());
    }
    std::vector<unsigned> counts(r.counts.begin(),r.counts.end());
    codegen::executor::ValidateClusterPlacement(c,r.grid,counts,offsets,tasks);
    codegen::executor::ValidateClusterEpisodeOrder(c,r.grid,r.cluster_chunks,offsets,tasks);++checks;
    auto reject=[&](auto mutate) {
      auto bad=r;mutate(bad);solver::MaterializedPlan discarded;
      assert(!solver::MaterializePlanPlacement(bad,&discarded,&error));++checks;
    };
    reject([](auto& bad){bad.cluster_dim=1;});
    reject([](auto& bad){bad.cluster_chunks.pop_back();});
    reject([](auto& bad){bad.cluster_chunks[1]=1;});
    reject([](auto& bad){++bad.counts[1];});
    reject([](auto& bad){std::swap(bad.physical_worker[0],bad.physical_worker[1]);});
    reject([](auto& bad){bad.stage_order[0]=bad.stage_order[1];});
    auto bad=tasks;
    for(auto& task:bad)if(task.stage==1 && task.logical_task%c==1){task.logical_task-=1;break;}
    bool rejected=false;try {
      codegen::executor::ValidateClusterEpisodeOrder(c,r.grid,r.cluster_chunks,offsets,bad);
    } catch(std::invalid_argument const&){rejected=true;}
    assert(rejected);++checks;
  }
  std::printf("PASS cluster_split materialization/episode ordering %u checks\n",checks);
}
