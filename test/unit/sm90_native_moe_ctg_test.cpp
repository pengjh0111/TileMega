// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativeFlow.h>
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <tilemega/Solver/Sm90NativeMoeDomain.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace tilemega;
using namespace tilemega::solver;
int main(int argc,char** argv) {
  if(argc!=6)return 2;
  analysis::IslContext isl;mlir::MLIRContext context;
  auto target=TargetSpec::FromJson(argv[1]);
  auto constants=Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]),argv[2]);
  Sm90DenseCostModel cost(target,constants);
  auto c=Sm90DenseCoordinates::FromJson(json::ParseFile(argv[4]));
  int batch=std::stoi(argv[5]);assert(c.moe && c.expert_fuse);
  auto imported=ImportSm90Dense(argv[3],c,context);
  auto work=Sm90DenseWorkloadFromPlan(imported.plan,batch);
  Sm90NativeCtgCache cache;
  for(auto route:{Sm90DenseCoordinates::Route::kLocal,Sm90DenseCoordinates::Route::kLastArriver,
                  Sm90DenseCoordinates::Route::kStage}) {
  c.route=route;
  for(bool fused:{true,false}) {
  if(!fused && !c.GlobalRoute())continue;
  c.expert_fuse=fused;
  auto ctg=InstantiateSm90NativeCtg(imported,c,batch,context,cache,cost.Grid(c.residency));
  assert(ctg.approximation.At("kind").AsString("kind")==kSm90NativeMoeDomainApproximation);
  auto flow=Sm90NativeCtgScore(imported,work,c,cost,context,cache);
  assert(std::isfinite(flow.step_ns) && flow.step_ns>0);
  int routes=0,combines=0,gateups=0,downs=0;
  int gt=work.shapes[6][0]/c.gemms[6].tile_n,dt=work.shapes[7][0]/c.gemms[7].tile_n;
  int gu=Sm90NativeExpectedExpertTasks(work.experts,work.top_k,batch,gt);
  int dn=Sm90NativeExpectedExpertTasks(work.experts,work.top_k,batch,dt);
  for(unsigned i=0;i<flow.nodes.size();++i) {
    auto const& node=flow.nodes[i];auto const& space=flow.problem.spaces[i];
    assert(node.price.tasks==space.count && space.count==ctg.spaces[i].count);
    if(node.price.name==std::string("route_")+c.RouteName()) {
      ++routes;assert(space.count==(c.GlobalRoute()?1:cost.Grid(c.residency)));
      assert(space.handoff_reducer==(route==Sm90DenseCoordinates::Route::kLastArriver));
    }
    if(node.price.name=="moe_combine") {++combines;assert(space.count==dt && space.handoff_reducer);}
    if(node.price.name=="gemm6") {++gateups;assert(space.count==gu && space.pages_per_worker==0);}
    if(node.price.name=="gemm7") {++downs;assert(space.count==dn && node.kappa==(c.expert_fuse?gt:1));}
  }
  assert(routes==work.layers && combines==work.layers && gateups==work.layers && downs==work.layers);
  int counted=0;std::vector<int> incoming(flow.nodes.size());
  for(auto const& edge:flow.problem.edges) {
    assert(edge.sorted && !edge.sorted->empty());++incoming[edge.consumer];
    auto const& p=flow.nodes[edge.producer].price.name;
    auto const& q=flow.nodes[edge.consumer].price.name;
    if(p=="gemm6" && q=="gemm7") {
      ++counted;
      assert(edge.all_producer==!c.expert_fuse && edge.kappa==(c.expert_fuse?gt:gu));
      assert(edge.hop_ns==(c.expert_fuse?0:constants.Barrier(flow.problem.workers)));
      for(auto [last,task]:*edge.sorted)assert(last==(c.expert_fuse?std::min(gu,(task/dt+1)*gt)-1:gu-1));
    }
    if(p==std::string("route_")+c.RouteName()) {
      assert(edge.all_producer==c.GlobalRoute() && edge.kappa==1);
      assert(edge.hop_ns==(c.GlobalRoute()?constants.Barrier(flow.problem.workers):0));
      for(auto [last,task]:*edge.sorted)
        assert(last==(c.GlobalRoute()?0:(task+(q=="gemm7"?gu:0))%flow.problem.workers));
    }
    if(q=="gemm0" && edge.producer>0)assert(p=="moe_combine");
  }
  assert(counted==work.layers && incoming.front()==0);
  for(unsigned i=1;i<incoming.size();++i)assert(incoming[i]>0);
  auto isolated=Sm90NativeCtgScore(imported,work,c,cost,context,cache,true);
  double sum=0;
  for(auto const& node:isolated.nodes) {
    for(auto const* price:{&node.price.middle,&node.price.head,&node.price.tail,&node.price.head_and_tail})
      assert(price->no_producer_dram_bytes==0);
    sum+=EvaluateFlow(Sm90NativeSingleStageProblem(node.price,cost,c.residency)).makespan_ns;
  }
  assert(std::abs(sum-isolated.step_ns)<1e-6);
  auto cached=Sm90NativeCtgScore(imported,work,c,cost,context,cache);
  assert(cached.step_ns==flow.step_ns && cache.space_hits>0 && cache.release_hits>0);
  std::vector<int> offsets(flow.nodes.size());
  for(unsigned i=1;i<flow.nodes.size();++i)if(c.expert_fuse && flow.nodes[i].price.name=="gemm7")offsets[i]=gu;
  auto replay=RefineAndReplaySm90Native(flow.problem,Sm90NativeCandidateTracks(flow.problem,offsets));
  assert(std::isfinite(replay.replay.makespan_ns) && replay.replay.makespan_ns>0);
  std::cout<<json::Value(json::Object{{"passed",true},{"batch",batch},{"layers",work.layers},
      {"spaces",int(flow.nodes.size())},{"pool_ns",flow.step_ns},{"replay_ns",replay.replay.makespan_ns},
      {"isolated_ns",isolated.step_ns},{"approximation",ctg.approximation}}).Dump()<<'\n';
  }
  }
}
