// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Solver/OperatorClasses.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <mlir/Parser/Parser.h>
#include <isl/map.h>
#include <fstream>
#include <iostream>
#include <set>
using namespace tilemega;
int main(int argc,char** argv) {
 try {
  if(argc!=5 && argc!=7)throw std::invalid_argument("usage: tilemega-skeleton-audit CG.mlir target.json seq past [EFT.tsv hop.tsv]");
  analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
  context.getOrLoadDialect<dialect::ExecDialect>();
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);
  if(!module)throw std::invalid_argument("cannot parse CG");
  int seq=std::stoi(argv[3]),past=std::stoi(argv[4]);auto target=TargetSpec::FromJson(argv[2]);
  auto problem=solver::PrepareSymbolicProblem(*module,target,{seq,past,seq+past},target.res.num_sms,1,1);
  analysis::CouplingCache cache;auto skeleton=solver::BuildPlanSkeleton(problem,target.res.num_sms,1,8,false,cache);
  std::map<std::string,int> structures,kinds;int comparisons=0;
  std::cout<<"producer\tconsumer\tdirection\toracle\tcoordinate\texpected\tactual\tequal\n";
  for(auto const& edge:skeleton.edges){++structures[analysis::ToString(edge.oracle->structure)];
    for(int direction=0;direction<2;++direction){
      auto const& oracle=direction?edge.oracle->reverse:edge.oracle->forward;
      ++kinds[analysis::ToString(oracle.kind())];int n=skeleton.spaces[direction?edge.consumer:edge.producer].count;
      for(int t:std::set<int>{0,n/2,n-1}){if(t<0)continue;
        auto* map=isl_map_read_from_str(isl.raw(),oracle.relation().c_str());
        map=isl_map_fix_si(map,isl_dim_in,0,t);char* raw=isl_map_to_str(map);isl_map_free(map);
        auto relation=analysis::CouplingRelation::FromIslText(raw).BindParams(skeleton.theta);free(raw);
        std::set<std::vector<long>> expected,actual;
        for(auto const& [a,b]:relation.Points())expected.insert(b);
        oracle.Query({t},skeleton.theta).ForEach([&](auto const& point){actual.insert(point);});
        bool equal=expected==actual;++comparisons;
        std::cout<<edge.producer<<'\t'<<edge.consumer<<'\t'<<direction<<'\t'<<analysis::ToString(oracle.kind())<<'\t'<<t<<'\t'<<expected.size()<<'\t'<<actual.size()<<'\t'<<equal<<'\n';
        if(!equal)throw std::runtime_error("Oracle differs from exact local expansion");
      }
    }
  }
  for(auto const& [name,count]:structures)std::cout<<"STRUCTURE "<<name<<" edges="<<count<<'\n';
  for(auto const& [name,count]:kinds)std::cout<<"ORACLE_KIND "<<name<<" directions="<<count<<'\n';
  std::cout<<"ORACLE_SET_EQUAL comparisons="<<comparisons<<" PASS\n";
  if(argc==7) {
    dialect::PlacementSolveOptions options;options.target=target;
    options.dims={seq,past,seq+past};
    auto integer=[&](char const* name,int fallback){auto a=(*module)->getAttrOfType<mlir::IntegerAttr>(name);return a?int(a.getInt()):fallback;};
    options.residency=integer("tmexec.solved_residency",1);
    options.verified_resident_limit=options.residency;
    options.requested_grid=integer("tmexec.solved_grid",0);
    options.kappa=integer("tmexec.solved_kappa",1);
    auto prepared=dialect::PreparePlacementProblem(*module,options);
    solver::EftRequest request;request.graph=&prepared.graph;request.task_ns=prepared.task_ns;
    request.grid=prepared.grid;request.sms=prepared.grid;request.ctas_per_sm=1;
    solver::EftSchedule schedule;std::string error;
    if(!solver::HopCurve::FromTsv(argv[6],&request.hop,&error))throw std::runtime_error(error);
    if(!solver::ScheduleByEarliestFinish(request,&schedule,&error))throw std::runtime_error(error);
    std::ofstream out(argv[5]);out<<"node\tstage\ttile\tworker\tslot\tstart_ns\tend_ns\n";
    for(std::size_t s=0;s<prepared.counts.size();++s)for(int t=0;t<prepared.counts[s];++t) {
      int n=prepared.graph.stage_offsets[s]+t;
      out<<n<<'\t'<<s<<'\t'<<t<<'\t'<<schedule.worker[n]<<'\t'<<schedule.slot[n]<<'\t'<<schedule.start_ns[n]<<'\t'<<schedule.end_ns[n]<<'\n';
    }
    std::cout<<"CONTROL_EFT plan="<<argv[5]<<" nodes="<<schedule.worker.size()<<'\n';
  }
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
