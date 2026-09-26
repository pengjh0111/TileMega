// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/StageFlowModel.h>
#include <tilemega/Solver/DramFluid.h>
#include <tilemega/Solver/ExecutionSimulator.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
using namespace tilemega::solver;
static void Near(double a,double b,char const* why){if(std::abs(a-b)>1e-7)throw std::runtime_error(std::string(why)+": "+std::to_string(a)+" != "+std::to_string(b));}
static FlowSpace Space(int n,TaskPriceParts parts){FlowSpace s;s.count=n;s.pieces={{n,parts}};s.piece_of_task.assign(n,0);return s;}
static void CheckFluidClock() {
  struct Item {double left,cap,rate;int count;};std::vector<Item> dense;
  DramFluidServer shared(100);std::mt19937 random(90109);double delivered=0;
  for(int step=0;step<1000;++step) {
    if(step<100 || step%3==0){double bytes=1+random()%1000,cap=1+random()%7;int count=1+random()%8;dense.push_back({bytes,cap,0,count});shared.Add(bytes,cap,count);}
    std::vector<int> active;long count=0;
    for(int i=0;i<int(dense.size());++i)if(dense[i].count){active.push_back(i);count+=dense[i].count;}
    if(active.empty())break;
    std::sort(active.begin(),active.end(),[&](int a,int b){return dense[a].cap<dense[b].cap;});
    double bandwidth=100,next=std::numeric_limits<double>::infinity();
    for(int i:active){auto& x=dense[i];x.rate=std::min(x.cap,bandwidth/count);bandwidth-=x.rate*x.count;count-=x.count;next=std::min(next,x.left/x.rate);}
    Near(shared.Next(),next,"cap clock versus independent dense water filling");
    double dt=next*(step%4==0?.25:1);std::vector<int> expected;
    for(int i:active){auto& x=dense[i];double sent=std::min(x.left,x.rate*dt);x.left-=sent;delivered+=sent*x.count;
      if(x.left<=1e-6){delivered+=x.left*x.count;x.left=0;x.count=0;expected.push_back(i);}}
    std::sort(expected.begin(),expected.end());
    if(shared.Advance(dt)!=expected)throw std::runtime_error("fluid completion set differs from dense reference");
    Near(shared.Delivered(),delivered,"cap clock conserves staggered cohort bytes");
  }
}
static void CheckInflightClock() {
  struct Item {double left,cap,q,rate;int count;};
  std::vector<Item> dense;
  InflightDramServer shared(10,{1,2},{10,10},{1,10},{20,20});
  std::mt19937 random(1472026);double delivered=0;
  for(int step=0;step<400;++step) {
    if(step<20 || step%3==0) {
      double bytes=1+random()%1000,cap=1+random()%7,q=1<<random()%3;
      int count=1+random()%8;dense.push_back({bytes,cap,q,0,count});
      shared.Add(bytes,cap,q,count);
    }
    std::vector<int> active;double weight=0;
    for(int i=0;i<int(dense.size());++i)if(dense[i].count){active.push_back(i);weight+=dense[i].q*dense[i].count;}
    if(active.empty())break;
    std::sort(active.begin(),active.end(),[&](int a,int b){
      double left=dense[a].cap/dense[a].q,right=dense[b].cap/dense[b].q;
      return left==right?a<b:left<right;
    });
    double bandwidth=10,next=std::numeric_limits<double>::infinity();
    for(int i:active) {
      auto& x=dense[i];x.rate=std::min(x.cap,bandwidth*x.q/weight);
      bandwidth-=x.rate*x.count;weight-=x.q*x.count;
      next=std::min(next,x.left/x.rate);
    }
    Near(shared.Next(),next,"in-flight class clock versus dense weighted filling");
    double dt=next*(step%4==0?.25:1);std::vector<int> expected;
    for(int i:active) {
      auto& x=dense[i];double sent=std::min(x.left,x.rate*dt);
      x.left-=sent;delivered+=sent*x.count;
      if(x.left<=1e-6){delivered+=x.left*x.count;x.left=0;x.count=0;expected.push_back(i);}
    }
    std::sort(expected.begin(),expected.end());
    if(shared.Advance(dt)!=expected)
      throw std::runtime_error("in-flight completion differs from dense reference");
    Near(shared.Delivered(),delivered,"in-flight class clock conserves bytes");
  }
}
int main() try {
  CheckFluidClock();
  CheckInflightClock();
  InflightDramServer in_flight(10,{10,20},{4,8},{10,20},{10,10});
  in_flight.Add(100,10,10);Near(in_flight.DeviceRate(),4,"one CTA in-flight rate");
  in_flight.Add(100,10,10);Near(in_flight.DeviceRate(),8,"two CTA in-flight rate");
  Near(in_flight.Next(),25,"weighted in-flight completion");
  if(in_flight.Advance(25)!=std::vector<int>({0,1}))throw std::runtime_error("in-flight completion set");
  Near(in_flight.Delivered(),200,"in-flight byte conservation");
  FlowProblem streamed;streamed.workers=2;streamed.dram_gbps=10;
  streamed.inflight_dram=true;streamed.inflight_curve_bytes={10,20};
  streamed.inflight_curve_gbps={4,8};streamed.cta_stream_curve_bytes={10,20};
  streamed.cta_stream_curve_gbps={10,10};
  streamed.spaces={Space(2,{0,0,100,10,100,10})};
  streamed.dram_floor_ns=streamed.floor_ns=20;streamed.all_external_miss=true;
  Near(EvaluateFlow(streamed).makespan_ns,25,"Level 1 measured in-flight model");
  tilemega::codegen::RuntimeTaskGraph streamed_graph;
  streamed_graph.stage_offsets={0,2};streamed_graph.successors={{},{}};
  MaterializedPlan streamed_plan;streamed_plan.queue={{{0,0}},{{0,1}}};
  SimulatorInput streamed_input;streamed_input.graph=&streamed_graph;
  streamed_input.task_price_parts.assign(2,{0,0,100,10,100,10});
  SimulatorOptions streamed_options;streamed_options.dram_fluid=true;
  streamed_options.dram_gbps=10;streamed_options.inflight_dram=true;
  streamed_options.inflight_curve_bytes=streamed.inflight_curve_bytes;
  streamed_options.inflight_curve_gbps=streamed.inflight_curve_gbps;
  streamed_options.cta_stream_curve_bytes=streamed.cta_stream_curve_bytes;
  streamed_options.cta_stream_curve_gbps=streamed.cta_stream_curve_gbps;
  streamed_options.all_external_miss=true;streamed_options.dram_floor_ns=20;
  SimulatorResult streamed_result;std::string streamed_error;
  if(!SimulateExecution(streamed_input,streamed_plan,streamed_options,{},
      &streamed_result,&streamed_error))throw std::runtime_error(streamed_error);
  Near(streamed_result.makespan_ns,25,"Level 1 and tile simulator share the in-flight physics");
  DramFluidServer server(10);server.Add(10,2);server.Add(80,20);Near(server.Next(),5,"capped water filling");auto done=server.Advance(5);if(done!=std::vector<int>{0})throw std::runtime_error("wrong fluid completion");Near(server.Next(),4,"redistribution after release");server.Advance(4);Near(server.Delivered(),90,"byte conservation");
  FlowProblem p;p.workers=2;p.dram_gbps=10;p.spaces={Space(2,{0,0,100,100})};p.dram_floor_ns=p.floor_ns=20;p.all_external_miss=true;
  Near(EvaluateFlow(p).makespan_ns,20,"shared bandwidth must not double");
  p.spaces={Space(2,{3,10,50,5})};p.dram_floor_ns=p.floor_ns=10;
  Near(EvaluateFlow(p).makespan_ns,13,"compute and DRAM overlap");
  p.workers=1;Near(EvaluateFlow(p).makespan_ns,26,"global worker serialization");
  p.workers=2;p.spaces={Space(2,{2,3,0,0}),Space(2,{2,3,0,0})};p.spaces[1].order=1;
  auto releases=std::make_shared<std::vector<std::pair<int,int>> const>(std::vector<std::pair<int,int>>{{0,0},{1,1}});
  p.edges={{0,1,1,false,false,releases}};p.publication_ns=7;p.consumer_wait_ns=11;p.hop_ns=13;p.dram_floor_ns=p.floor_ns=0;
  Near(EvaluateFlow(p).makespan_ns,41,"publication wait and hop counted once");
  p.edges[0].colocated=true;Near(EvaluateFlow(p).makespan_ns,10,"kappa one colocated synchronization omitted");
  p.edges[0].kappa=2;Near(EvaluateFlow(p).makespan_ns,41,"kappa groups retain synchronization");
  auto d=DecomposeFlow(p);for(auto const& link:d.original.critical_links)Near(link.wait_ns+link.fixed_ns+link.mainloop_ns+link.publication_ns,link.end_ns-link.start_ns,"critical link segment closure");Near(d.synchronization+d.fixed+d.contention+d.chain,d.original.makespan_ns-p.floor_ns,"counterfactual closure");
  if(CoarsenRelease(4,7,4)!=6 || CoarsenRelease(0,7,4)!=3)throw std::runtime_error("coarsening tail mismatch");
  bool rejected=false;p.dram_floor_ns=100;try{EvaluateFlow(p);}catch(std::runtime_error const&){rejected=true;}if(!rejected)throw std::runtime_error("physical floor not enforced");
  FlowOptions no_external;no_external.no_external=true;EvaluateFlow(p,no_external);
  tilemega::codegen::RuntimeTaskGraph graph;graph.stage_offsets={0,2,4};graph.successors={{2},{3},{},{}};
  MaterializedPlan plan;plan.queue={{{0,0},{1,0}},{{0,1},{1,1}}};
  SimulatorInput input;input.graph=&graph;input.task_price_parts.assign(4,TaskPriceParts{2,3,20,4});
  SimulatorOptions opts;opts.dram_fluid=true;opts.dram_gbps=8;opts.flat_hop=true;opts.all_external_miss=true;opts.dram_floor_ns=10;
  SimulatorResult simulated;std::string error;
  if(!SimulateExecution(input,plan,opts,{},&simulated,&error))throw std::runtime_error(error);
  Near(simulated.makespan_ns,14,"FIFO fluid pipeline");
  input.prefetch_ns={1,1,1,1};if(SimulateExecution(input,plan,opts,{},&simulated,&error))throw std::runtime_error("fluid path accepted prefetch");
  input.prefetch_ns.clear();input.fluid_forced_local_hops={{0,2},{1,3}};
  HopCurve local_event;local_event.c0=5;
  if(!SimulateExecution(input,plan,opts,local_event,&simulated,&error))throw std::runtime_error(error);
  Near(simulated.makespan_ns,19,"group event retains same-worker visibility hop");
  std::cout<<"STAGE_FLOW waterfill=PASS byte_conservation=PASS overlap=PASS coarsen=PASS kappa_sync=PASS counterfactuals=PASS floor_assertion=PASS\n";
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
