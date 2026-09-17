// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/VisitFiniteRelation.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <mlir/Parser/Parser.h>
#include <mlir/IR/BuiltinOps.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
using namespace tilemega;
using Point=std::array<long,6>;
using Clock=std::chrono::steady_clock;
struct Sink {int arity;std::vector<Point> points;};
static isl_stat reference_point(isl_point* p,void* opaque) {
  auto& s=*static_cast<Sink*>(opaque);Point values{};
  for(int i=0;i<s.arity;++i) {auto v=isl_point_get_coordinate_val(p,isl_dim_set,i);values[i]=isl_val_get_num_si(v);isl_val_free(v);}
  isl_point_free(p);s.points.push_back(values);return isl_stat_ok;
}
static void compare(analysis::IslContext& ctx,std::string const& name,std::string const& text,int arity) {
  Sink old{arity,{}},fast{arity,{}};
  auto start=Clock::now();auto map=isl_map_read_from_str(ctx.raw(),text.c_str());auto set=isl_map_wrap(map);
  auto status=isl_set_foreach_basic_set(set,[](isl_basic_set* b,void* p){
    auto s=isl_set_from_basic_set(b);auto status=isl_set_foreach_point(s,reference_point,p);isl_set_free(s);return status;
  },&old);isl_set_free(set);
  if(status!=isl_stat_ok)throw std::runtime_error("reference enumeration failed");
  auto old_us=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
  start=Clock::now();analysis::VisitFiniteRelation(ctx,text,arity,[&](long const* p){Point value{};std::copy_n(p,arity,value.begin());fast.points.push_back(value);});
  auto fast_us=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
  std::sort(old.points.begin(),old.points.end());std::sort(fast.points.begin(),fast.points.end());
  if(old.points!=fast.points)throw std::runtime_error("emitted point multiset differs: "+name);
  std::cout<<name<<'\t'<<old.points.size()<<'\t'<<old_us<<'\t'<<fast_us<<"\t1\n"<<std::flush;
}
int main(int argc,char** argv)try {
  analysis::IslContext isl;mlir::MLIRContext ctx;ctx.getOrLoadDialect<dialect::CGDialect>();
  std::cout<<"relation\tpoints\treference_us\texact_box_us\tidentical\n";
  compare(isl,"rectangular_1M","{ [i] -> [j] : 0<=i<1024 and 0<=j<1024 }",2);
  for(int arg=1;arg<argc;++arg) {
    auto m=mlir::parseSourceFile<mlir::ModuleOp>(argv[arg],&ctx);if(!m)throw std::runtime_error("invalid CG");
    auto attr=[&](char const* key){return int(m->getOperation()->getAttrOfType<mlir::IntegerAttr>(key).getInt());};
    int seq=attr("tilemega.solved_seq"),past=attr("tilemega.solved_past");
    auto runtime=codegen::ReadRuntimePlan(*m);auto model=solver::ModelDescription::FromCouplingGraph(*m,{seq,past,seq+past},"relation-audit");
    solver::RuntimeProjectionOptions options{attr("tilemega.solved_grid"),128,attr("tilemega.solved_kappa")};options.count_wait_entries=false;
    auto projection=solver::ProjectRuntimeQueues(model,runtime,options);
    compare(isl,std::string(argv[arg])+":dependencies",projection.dependencies.ToString(),4);
    compare(isl,std::string(argv[arg])+":events",projection.requested_events.ToString(),6);
  }
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
