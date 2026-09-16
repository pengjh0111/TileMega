// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/ParametricPlacementIO.h>
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <mlir/Parser/Parser.h>
#include <fstream>
#include <filesystem>
#include <iostream>
using namespace tilemega;
static void save(solver::MaterializedPlan const& p,std::filesystem::path path){std::ofstream f(path);f<<"stage\ttask\tworker\tslot\n";for(std::size_t s=0;s<p.owner.size();++s)for(std::size_t t=0;t<p.owner[s].size();++t)f<<s<<'\t'<<t<<'\t'<<p.owner[s][t]<<'\t'<<p.slot[s][t]<<'\n';}
static long differences(solver::MaterializedPlan const& a,solver::MaterializedPlan const& b){if(a.owner.size()!=b.owner.size())throw std::invalid_argument("stage count changed across grid");long n=0;for(std::size_t s=0;s<a.owner.size();++s){if(a.owner[s].size()!=b.owner[s].size())throw std::invalid_argument("task count changed across grid");for(std::size_t t=0;t<a.owner[s].size();++t)n+=(a.owner[s][t]!=b.owner[s][t] || a.slot[s][t]!=b.slot[s][t]);}return n;}
int main(int argc,char** argv)try{
 if(argc!=5)throw std::invalid_argument("cross_grid SYMBOLIC_CG TARGET HOP OUT_DIR");
 analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
 auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);if(!module)throw std::invalid_argument("invalid CG");
 auto attrs=module->getOperation()->getAttrOfType<mlir::ArrayAttr>("tilemega.parametric_alternatives");if(!attrs || attrs.size()!=4)throw std::invalid_argument("four symbolic families required");
 auto integer=[&](char const* key){auto a=module->getOperation()->getAttrOfType<mlir::IntegerAttr>(key);if(!a)throw std::invalid_argument(key);return int(a.getInt());};
 auto selected=module->getOperation()->getAttrOfType<mlir::StringAttr>("tilemega.solved_placement");
 dialect::PlacementSolveOptions options;options.target=TargetSpec::FromJson(argv[2]);options.residency=integer("tilemega.solved_residency");options.kappa=integer("tilemega.solved_kappa");options.verified_resident_limit=options.residency;options.dims={4,3,7};
 std::string error;if(!solver::HopCurve::FromTsv(argv[3],&options.hop,&error))throw std::runtime_error(error);
 std::filesystem::path root=argv[4];std::filesystem::create_directories(root);std::ofstream rows(root/"comparisons.tsv"),catalog(root/"catalog.tsv");
 rows<<"grid\tfamily\treference\treference_family\tdifferent_entries\ttemplate_file\treference_file\n";catalog<<"grid\tplacement\tfloor_ns\tpredicted_ns\n";
 for(int grid:{256,340}){
  options.requested_grid=grid;auto clone=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(module->clone()));auto solved=dialect::SolveAndWritePlacement(*clone,options);
  std::vector<solver::PlacementEvaluation const*> valid;
  for(auto const& c:solved.candidates){if(!c.error.empty())throw std::runtime_error(c.name+": "+c.error);valid.push_back(&c);catalog<<grid<<'\t'<<c.name<<'\t'<<c.bounds.lower_bound_ns<<'\t'<<c.predicted_ns<<'\n';save(c.plan,root/("g"+std::to_string(grid)+"_"+c.name+".tsv"));}
  auto best=*std::min_element(valid.begin(),valid.end(),[](auto a,auto b){return std::tie(a->bounds.lower_bound_ns,a->predicted_ns)<std::tie(b->bounds.lower_bound_ns,b->predicted_ns);});
  auto chosen=std::find_if(valid.begin(),valid.end(),[&](auto c){return c->name==selected.getValue();});if(chosen==valid.end())throw std::invalid_argument("selected family absent");
  std::vector<int> counts;for(auto const& v:best->plan.owner)counts.push_back(v.size());
  for(auto attr:attrs){auto p=dialect::ReadParametricPlacement(llvm::cast<mlir::DictionaryAttr>(attr));analysis::ParamBinding theta;theta.Bind("S",4).Bind("G",grid);auto plan=solver::EvaluateParametricPlacement(p,theta,counts,grid);
   auto prefix="g"+std::to_string(grid)+"_"+p.family;save(plan,root/(prefix+"_template.tsv"));
   for(auto const& ref:std::vector<std::pair<std::string,solver::PlacementEvaluation const*>>{{"selected",*chosen},{"resolved",best}})rows<<grid<<'\t'<<p.family<<'\t'<<ref.first<<'\t'<<ref.second->name<<'\t'<<differences(plan,ref.second->plan)<<'\t'<<prefix<<"_template.tsv\t"<<"g"<<grid<<'_'<<ref.second->name<<".tsv\n";
   auto same=std::find_if(valid.begin(),valid.end(),[&](auto c){return c->name==p.family || (p.family=="band" && c->name=="balanced");});
   if(p.family!="band" && same!=valid.end() && differences(plan,(*same)->plan)!=0)throw std::runtime_error("template differs from native family "+p.family);
  }
  std::cout<<"CROSS_GRID "<<grid<<" candidates=6 winner="<<best->name<<" CPU_ONLY\n"<<std::flush;
 }
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
