// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Solver/OperatorClasses.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <mlir/Parser/Parser.h>
#include <isl/map.h>
#include <iostream>
#include <set>
using namespace tilemega;
int main(int argc,char** argv) {
 try {
  if(argc!=5)throw std::invalid_argument("usage: tilemega-skeleton-audit CG.mlir target.json seq past");
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
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
