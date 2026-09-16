// SPDX-License-Identifier: BSD-3-Clause
#include "../SIMULATOR/cell_inputs.h"
#include <tilemega/Solver/JointSearch.h>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace tilemega::solver;using namespace tilemega::experiments;
using Row=std::map<std::string,std::string>;
std::vector<Row> read(std::string path){
  std::ifstream f(path);if(!f)throw std::runtime_error(path);std::string line;std::getline(f,line);auto keys=Split(line,'\t');std::vector<Row> rows;
  while(std::getline(f,line)){auto values=Split(line,'\t');Row r;for(std::size_t i=0;i<keys.size();++i)r[keys[i]]=values.at(i);rows.push_back(r);}return rows;
}
int main(int argc,char**argv)try{
  if(argc!=2)throw std::runtime_error("rank CELL_DIR");std::string dir=argv[1];
  std::vector<JointCandidate> candidates;
  for(auto r:read(dir+"/project_manifest.tsv")){
    JointCandidate c;c.config={std::stoi(r["m"]),std::stoi(r["n"]),std::stoi(r["k"]),std::stoi(r["stages"]),std::stoi(r["split"])};
    c.kappa=std::stoi(r["kappa"]);c.ctas_per_sm=std::stoi(r["residency"]);c.work_lb_ns=std::stod(r["work_lb_ns"]);c.cp_lb_ns=std::stod(r["cp_lb_ns"]);c.priority_ns=std::stod(r["priority_ns"]);c.key=r["key"];candidates.push_back(c);
  }
  JointSearchStats stats;
  auto result=SearchL2Configurations(candidates,candidates.size(),[&](JointCandidate const&c){
    std::vector<JointEvaluation> evaluated;
    try{for(auto r:read(dir+"/plans/"+c.key+"/predicted.tsv")){
      JointEvaluation e;e.placement=r["placement"];e.status=r["status"];e.simulated=r["simulated"]=="1";e.makespan_ns=std::stod(r["makespan_ns"]);e.floor_ns=std::stod(r["floor_ns"]);evaluated.push_back(e);
    }}catch(std::exception const&e){JointEvaluation failed;failed.status="projection_failed";evaluated.push_back(failed);}
    return evaluated;
  },&stats);
  std::ofstream out(dir+"/solver_rank.tsv");out<<std::setprecision(12)<<"key\tplacement\tstatus\tsimulated\tmakespan_ns\tfloor_ns\n";
  for(auto const&r:result)out<<r.candidate.key<<'\t'<<r.placement<<'\t'<<r.status<<'\t'<<r.simulated<<'\t'<<r.makespan_ns<<'\t'<<r.floor_ns<<'\n';
  std::cout<<"JOINT_RANK considered="<<stats.considered<<" evaluated="<<stats.evaluated<<" bound_pruned="<<stats.pruned<<" capacity_deferred="<<stats.capacity_deferred<<'\n';
}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}
