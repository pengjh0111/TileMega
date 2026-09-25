// SPDX-License-Identifier: BSD-3-Clause
// The same source is compiled against the baseline and current headers/library.
// Historical trace durations are fixed inputs, not fresh GPU performance claims.
#include "../SIMULATOR/cell_inputs.h"
#include <tilemega/Solver/ExecutionSimulator.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace tilemega;
using namespace tilemega::solver;
using namespace tilemega::experiments;

void Bits(double value) {
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  std::cout << '\t' << std::hex << std::setw(16) << std::setfill('0') << bits << std::dec;
}

int main(int argc, char** argv) try {
  if (argc != 2) throw std::runtime_error("usage: simulator_identity REPO");
  std::string root=argv[1], error;
  HopCurve hop;
  if (!HopCurve::FromTsv(root+"/docs/experiments/SIMULATOR/hop_ns.tsv", &hop, &error))
    throw std::runtime_error(error);
  for (auto model : {"gqa2", "mha4"}) for (int seq : {4,128}) {
    std::string cell=std::string(model)+"_s"+std::to_string(seq);
    std::string raw=root+"/docs/experiments/SIMULATOR/raw/";
    auto dump=ReadCellDump(raw+"run/"+model+"_p5_s"+std::to_string(seq)+".out");
    auto deps=ReadDependencies(root+"/docs/experiments/PLAN_CONTRACT/legacy_identity/plan/"+model+".cu",0);
    auto graph=codegen::MaterializeRuntimeTaskGraph(dump.counts,deps,dump.grid);
    SimulatorInput input;
    input.graph=&graph;
    input.task_ns.assign(graph.stage_offsets.back(),-1);
    std::ifstream trace(raw+"dump/"+cell+"_p5/slots.tsv");
    std::string line;
    std::getline(trace,line);
    auto header=Split(line,'\t');
    auto col=[&](char const* name) {
      auto it=std::find(header.begin(),header.end(),name);
      if(it==header.end()) throw std::runtime_error("missing trace column");
      return std::size_t(it-header.begin());
    };
    auto stage=col("stage"), logical=col("logical_task"), begin=col("run_begin"), end=col("run_end");
    while(std::getline(trace,line)) {
      auto r=Split(line,'\t');
      input.task_ns.at(graph.stage_offsets.at(std::stoi(r[stage]))+std::stoi(r[logical]))=
          double(std::stoull(r[end])-std::stoull(r[begin]));
    }
    for(double value:input.task_ns) if(value<0) throw std::runtime_error("unobserved task");
    input.worker_sm=ReadWorkerSm(raw+"dump/"+cell+"_p5/slots.tsv",dump.grid);
    PlanRequest request;
    request.grid=dump.grid;request.counts=dump.counts;request.stage_order=dump.stage_order;
    request.graph=&graph;request.physical_worker.resize(dump.grid);
    std::iota(request.physical_worker.begin(),request.physical_worker.end(),0);
    for(auto mode:{dialect::PlacementMode::kLegacyGridStride,dialect::PlacementMode::kRotate,
                   dialect::PlacementMode::kBalanced}) {
      request.mode=mode;MaterializedPlan plan;
      if(!MaterializePlanPlacement(request,&plan,&error) || !CheckPlanLegality(graph,plan,&error))
        throw std::runtime_error(error);
      for(int arm=0;arm<2;++arm) {
        SimulatorOptions options;
        options.sms=dump.num_sms;options.ctas_per_sm=dump.ctas_per_sm;
        options.flat_hop=arm;options.proportional_sharing=true;
        options.observed_task_times=arm;options.publication_ns=arm?1074.125:0;
        options.consumer_wait_ns=arm?1764.375:0;
        SimulatorResult result;
        if(!SimulateExecution(input,plan,options,hop,&result,&error)) throw std::runtime_error(error);
        std::cout<<cell<<'\t'<<int(mode)<<'\t'<<arm<<"\tsummary";
        for(double x:{result.makespan_ns,result.total_work_ns,result.solo_work_ns,
                      result.total_block_ns,result.busiest_worker_ns,result.critical_path_ns}) Bits(x);
        std::cout<<'\t'<<result.busiest_worker<<'\t'<<result.cross_worker_edges<<'\t'<<result.same_worker_edges<<'\n';
        for(std::size_t n=0;n<result.tasks.size();++n) {
          auto const& task=result.tasks[n];
          std::cout<<cell<<'\t'<<int(mode)<<'\t'<<arm<<'\t'<<n<<'\t'<<task.worker;
          Bits(task.start_ns);Bits(task.end_ns);Bits(task.block_ns);Bits(task.stretch);
          std::cout<<'\n';
        }
      }
    }
  }
} catch(std::exception const& e) { std::cerr<<e.what()<<'\n';return 1; }
