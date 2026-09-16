// SPDX-License-Identifier: BSD-3-Clause
#include "../SIMULATOR/cell_inputs.h"
#include <tilemega/Solver/ExecutionSimulator.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <set>
using namespace tilemega; using namespace tilemega::solver;
using namespace tilemega::experiments;
using Row=std::map<std::string,std::string>;
std::vector<Row> table(std::string path) {
  std::ifstream f(path); if(!f) throw std::runtime_error(path);
  std::string line; std::getline(f,line);auto keys=Split(line,'\t');
  std::vector<Row> rows;
  while(std::getline(f,line)) {if(!line.empty()&&line.back()=='\r')line.pop_back();
    auto values=Split(line,'\t');Row r;for(std::size_t i=0;i<keys.size();++i)r[keys[i]]=values.at(i);rows.push_back(r);}
  return rows;
}
int main(int argc,char**argv) try {
  if(argc!=7)throw std::runtime_error("replay REPO MANIFEST OUT PUB WAIT HOP");
  std::string repo=argv[1],error;std::ofstream out(argv[3]);
  out<<std::setprecision(12)<<"dump\twindow\tmeasured_ns\tpredicted_ns\terror_ns\trelative_error\tevaluate_us\n";
  for(auto entry:table(argv[2])) {
    std::string dir=repo+"/"+entry.at("dump"); auto rows=table(dir+"/slots.tsv");
    Row meta;for(auto r:table(dir+"/meta.tsv"))meta[r.at("key")]=r.at("value");
    int grid=std::stoi(meta.at("grid"));std::vector<int> counts(std::stoi(meta.at("stage_count")),0);
    for(auto r:rows)counts.at(std::stoi(r.at("stage")))=std::max(counts.at(std::stoi(r.at("stage"))),std::stoi(r.at("logical_task"))+1);
    auto graph=codegen::MaterializeRuntimeTaskGraph(counts,ReadDependencies(repo+"/"+entry.at("source"),0),grid);
    std::set<int> active;for(auto e:table(dir+"/events.tsv"))if(std::stoi(e.at("fanin"))>0)active.insert(std::stoi(e.at("stage")));
    SimulatorInput in;in.graph=&graph;in.task_ns.resize(rows.size());in.publication_required.resize(rows.size());in.consumer_wait_required.resize(rows.size());
    MaterializedPlan plan;plan.queue.resize(grid);plan.owner.resize(counts.size());plan.slot.resize(counts.size());
    for(std::size_t s=0;s<counts.size();++s){plan.owner[s].resize(counts[s]);plan.slot[s].resize(counts[s]);}
    // Window traces replay their actual execution order as a FIFO projection.
    // This is explicitly a W>1 approximation, never a window simulator claim.
    std::stable_sort(rows.begin(),rows.end(),[](Row const&a,Row const&b){return std::stoull(a.at("run_begin"))<std::stoull(b.at("run_begin"));});
    for(auto r:rows) {int s=std::stoi(r.at("stage")),t=std::stoi(r.at("logical_task")),n=graph.stage_offsets[s]+t,w=std::stoi(r.at("worker"));
      in.task_ns[n]=std::stod(r.at("run_end"))-std::stod(r.at("run_begin"));
      in.publication_required[n]=active.count(s);in.consumer_wait_required[n]=std::stoi(r.at("wait_count"))>0;
      plan.owner[s][t]=w;plan.slot[s][t]=plan.queue[w].size();plan.queue[w].push_back({static_cast<std::uint32_t>(s),t});}
    if(!CheckPlanLegality(graph,plan,&error))throw std::runtime_error(dir+": "+error);
    SimulatorOptions options;options.observed_task_times=true;options.publication_ns=std::stod(argv[4]);options.consumer_wait_ns=std::stod(argv[5]);options.flat_hop=true;
    HopCurve hop;hop.c0=std::stod(argv[6]);SimulatorResult result;
    auto start=std::chrono::steady_clock::now();
    if(!SimulateExecution(in,plan,options,hop,&result,&error))throw std::runtime_error(error);
    double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
    double measured=std::stod(meta.at("l2_ms"))*1e6;
    out<<entry.at("dump")<<'\t'<<entry.at("window")<<'\t'<<measured<<'\t'<<result.makespan_ns<<'\t'<<result.makespan_ns-measured<<'\t'<<result.makespan_ns/measured-1<<'\t'<<us<<'\n';out.flush();
    std::cout<<"REPLAY "<<entry.at("dump")<<std::endl;
  }
}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}
