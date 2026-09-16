// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <mlir/Parser/Parser.h>
#include <iostream>
using namespace tilemega;
int main(int argc,char** argv)try{
 if(argc!=3)throw std::invalid_argument("check_prepared_prices CG TARGET");
 analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
 auto original=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);
 dialect::PlacementSolveOptions options;options.target=TargetSpec::FromJson(argv[2]);options.dims={4,3,7};options.kappa=1;options.verified_resident_limit=2;
 auto cache=std::make_shared<dialect::PlacementTaskPriceCache>();int compared=0;
 for(int r:{1,2}){
  options.residency=r;options.task_price_cache=cache;
  auto copy=[&]{return mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(original->clone()));};
  auto a=copy();auto cached=dialect::SolveAndWritePlacement(*a,options);
  options.task_price_cache.reset();auto b=copy();auto direct=dialect::SolveAndWritePlacement(*b,options);
  if(cached.candidates.size()!=direct.candidates.size())throw std::runtime_error("candidate count");
  for(std::size_t i=0;i<cached.candidates.size();++i){
   auto const& x=cached.candidates[i];auto const& y=direct.candidates[i];
   if(x.name!=y.name || x.bounds.lower_bound_ns!=y.bounds.lower_bound_ns || x.predicted_ns!=y.predicted_ns || x.plan.owner!=y.plan.owner || x.plan.slot!=y.plan.slot)throw std::runtime_error("cache changed plan or cost");++compared;
  }
 }
 if(cache->prices.size()!=1)throw std::runtime_error("residency incorrectly invalidated immutable task work");
 std::cout<<"PREPARED_PRICES PASS independent_catalog_comparisons="<<compared<<" cache_entries="<<cache->prices.size()<<'\n';
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
