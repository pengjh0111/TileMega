// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/VisitFiniteRelation.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/ParametricTemplates.h>
#include <tilemega/Solver/VariantSchedule.h>
#include <mlir/Parser/Parser.h>
#include <fstream>
#include <filesystem>
#include <iostream>
using namespace tilemega;using namespace tilemega::solver;
static void save(MaterializedPlan const& p,std::filesystem::path path){std::ofstream f(path);f<<"stage\ttask\tworker\tslot\n";for(std::size_t s=0;s<p.owner.size();++s)for(std::size_t t=0;t<p.owner[s].size();++t)f<<s<<'\t'<<t<<'\t'<<p.owner[s][t]<<'\t'<<p.slot[s][t]<<'\n';}
int main(int argc,char** argv)try{
 if(argc!=3)throw std::invalid_argument("champion_fit CHOSEN_CG OUT_DIR");
 analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);if(!module)throw std::invalid_argument("invalid CG");
 auto attr=[&](char const* key){auto a=module->getOperation()->getAttrOfType<mlir::IntegerAttr>(key);if(!a)throw std::invalid_argument(key);return int(a.getInt());};
 int seq=attr("tilemega.solved_seq"),past=attr("tilemega.solved_past"),grid=attr("tilemega.solved_grid"),kappa=attr("tilemega.solved_kappa");
 auto runtime=codegen::ReadRuntimePlan(*module);auto model=ModelDescription::FromCouplingGraph(*module,{seq,past,seq+past},"fit-audit");RuntimeProjectionOptions options{grid,128,kappa};options.count_wait_entries=false;auto p=ProjectRuntimeQueues(model,runtime,options);
 std::vector<int> counts;AffineCountPiece piece;piece.begin=piece.end=seq;for(auto const& s:p.stages){int n=s.task_count.Eval({});counts.push_back(n);piece.counts.push_back(std::to_string(n));piece.band_width.push_back(std::max(1,(n+grid-1)/grid));}
 auto graph=codegen::MaterializeRuntimeTaskGraph(counts,{},grid);analysis::VisitFiniteRelation(isl,p.dependencies.ToString(),4,[&](long const* e){graph.successors.at(graph.stage_offsets.at(e[2])+e[3]).push_back(graph.stage_offsets.at(e[0])+e[1]);});
 auto edges=runtime.dependencies;std::stable_sort(edges.begin(),edges.end(),[](auto const&a,auto const&b){return a.consumer<b.consumer;});std::vector<std::uint32_t> order;
 for(auto const& e:BuildVariantStageSchedule(edges,model.stages.size()).schedule)for(std::size_t s=0;s<p.stages.size();++s)if(p.stages[s].logical_stage==int(e.stage))order.push_back(s);
 std::vector<int> first(model.stages.size(),-1),last(first),level(p.stages.size());std::vector<std::vector<int>> predecessors(level.size());
 for(std::size_t s=0;s<p.stages.size();++s){int l=p.stages[s].logical_stage;if(first[l]<0)first[l]=s;last[l]=s;}
 for(auto const& e:edges)predecessors[first[e.consumer]].push_back(last[e.producer]);
 for(std::size_t s=1;s<p.stages.size();++s)if(p.stages[s].logical_stage==p.stages[s-1].logical_stage)predecessors[s].push_back(s-1);
 for(int s:order)for(int parent:predecessors[s])level[s]=std::max(level[s],level[parent]+1);
 PlanRequest request;request.grid=grid;request.counts=counts;request.graph=&graph;request.stage_order=order;request.physical_worker.resize(grid);std::iota(request.physical_worker.begin(),request.physical_worker.end(),0);
 auto placement=*module->getOps<dialect::PlacementOp>().begin();auto mode=placement->getAttrOfType<mlir::StringAttr>("mode");if(!mode || !dialect::ParsePlacementMode(mode.getValue().data(),mode.getValue().size(),&request.mode))throw std::invalid_argument("unknown selected mode");
 if(auto params=placement->getAttrOfType<mlir::DenseI64ArrayAttr>("params"))for(auto v:params.asArrayRef())request.params.push_back(v);
 MaterializedPlan selected;std::string error;
 if(request.mode==dialect::PlacementMode::kEft){auto table=module->getOperation()->getAttrOfType<mlir::DictionaryAttr>(dialect::kPlacementTableAttr);auto workers=llvm::cast<mlir::DenseI64ArrayAttr>(table.get("worker")).asArrayRef();auto slots=llvm::cast<mlir::DenseI64ArrayAttr>(table.get("slot")).asArrayRef();selected.owner.resize(counts.size());selected.slot.resize(counts.size());std::size_t n=0;for(std::size_t s=0;s<counts.size();++s)for(int t=0;t<counts[s];++t){selected.owner[s].push_back(workers[n]);selected.slot[s].push_back(slots[n++]);}if(n!=workers.size() || !BuildPlanQueues(counts,grid,&selected,&error))throw std::invalid_argument("selected table shape "+error);}
 else if(!MaterializePlanPlacement(request,&selected,&error))throw std::invalid_argument(error);
 std::filesystem::path out=argv[2];std::filesystem::create_directories(out);save(selected,out/"selected.tsv");std::ofstream rows(out/"fit.tsv");rows<<"family\tseq\tgrid\tnodes\tdifferent_entries\ttemplate_file\tselected_file\n";
 for(std::string family:{"legacy_grid_stride","rotate","band","wavefront"}){auto templ=BuildParametricTemplate(p.tasks,p.dependencies,order,{piece},{grid},grid,family,level);analysis::ParamBinding theta;theta.Bind("S",seq).Bind("G",grid);auto evaluated=EvaluateParametricPlacement(templ,theta,counts,grid);
  auto native_request=request;native_request.params.clear();
  if(family=="legacy_grid_stride")native_request.mode=dialect::PlacementMode::kLegacyGridStride;
  else if(family=="rotate")native_request.mode=dialect::PlacementMode::kRotate;
  else {native_request.mode=dialect::PlacementMode::kTemplate;native_request.params={family=="band" ? 0 : 1};}
  MaterializedPlan native;if(!MaterializePlanPlacement(native_request,&native,&error))throw std::invalid_argument(error);
  if(evaluated.owner!=native.owner || evaluated.slot!=native.slot)throw std::invalid_argument("symbolic/native family disagrees: "+family);
  save(native,out/(family+"_native.tsv"));long different=0,nodes=0;for(std::size_t s=0;s<counts.size();++s)for(int t=0;t<counts[s];++t){++nodes;different+=evaluated.owner[s][t]!=selected.owner[s][t] || evaluated.slot[s][t]!=selected.slot[s][t];}save(evaluated,out/(family+".tsv"));rows<<family<<'\t'<<seq<<'\t'<<grid<<'\t'<<nodes<<'\t'<<different<<'\t'<<family<<".tsv\tselected.tsv\n";rows.flush();}
 std::cout<<"CHAMPION_FIT grid="<<grid<<" seq="<<seq<<" families=4\n";
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
