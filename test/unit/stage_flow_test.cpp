// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/StageFlowModel.h>
#include <tilemega/Solver/DramFluid.h>
#include <tilemega/Solver/ExecutionSimulator.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace tilemega::solver;
static void Near(double a,double b,char const* why){if(std::abs(a-b)>1e-7)throw std::runtime_error(std::string(why)+": "+std::to_string(a)+" != "+std::to_string(b));}
static FlowSpace Space(int n,TaskPriceParts parts){FlowSpace s;s.count=n;s.pieces={{n,parts}};s.piece_of_task.assign(n,0);return s;}
int main() try {
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
