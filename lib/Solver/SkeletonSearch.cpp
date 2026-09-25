// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Solver/ModelDramFloor.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Analysis/ExactMemo.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
namespace tilemega::solver {
namespace {
std::string ConfigKey(std::vector<GemmConfig> const& config,int kappa,int residency) {
  std::ostringstream out;for(auto const& g:config)out<<g.tile_m<<'x'<<g.tile_n<<'x'<<g.tile_k<<'s'<<g.stages<<'k'<<g.split_k<<';';
  out<<"kappa="<<kappa<<";residency="<<residency;return out.str();
}
struct SearchContext {
  frontend::ImportedSemantics imported;
  frontend::TorchExportImporter importer;
  std::vector<OperatorClass> classes;
  analysis::CouplingCache cache;
  FlowPreparationCache flow_cache;
  VariantResourceCache resources;
  mlir::MLIRContext& context;
  SkeletonSearchOptions options;
  std::optional<analysis::DramFloor> floor;
  std::optional<SymbolicProblem> base,last_structure;
  std::string last_geometry;
  mlir::Attribute floor_attribute;
  ScalarType dtype;
  SearchContext(frontend::ImportedSemantics input,mlir::MLIRContext& ctx,SkeletonSearchOptions const& opts)
      :imported(std::move(input)),classes(BuildOperatorClasses(imported)),resources(opts.variant_probe,opts.common.timing),context(ctx),options(opts),
       dtype(imported.lifted.sem.ops.front().dtype==analysis::ScalarType::kBF16?ScalarType::kBF16:ScalarType::kF32) {}
  SkeletonSolvedPoint Prepare(std::vector<GemmConfig> const& config,int kappa,int residency,int actual=0,bool materialize=false) {
    auto* timing=options.common.timing;auto const& target=options.common.placement.target;
    if(timing)timing->candidate=ConfigKey(config,kappa,residency);
    auto estimate=resources.Estimate(classes,config,target,dtype);
    int limit=actual?actual:estimate.resident_limit;if(limit<1)throw std::invalid_argument("no resident CTA for geometry");
    residency=std::min(residency,limit);
    SkeletonSolvedPoint point;point.candidate.config=config;point.candidate.kappa=kappa;point.candidate.residency=residency;
    point.candidate.key=ConfigKey(config,kappa,residency);point.candidate.estimated_limit=estimate.resident_limit;point.candidate.actual_limit=actual;
    auto granularity=ClassGranularity(imported,classes,config);
    if(!base || materialize) {
      point.module=importer.InstantiateForGranularity(imported,context,granularity,&cache,nullptr,timing);
      {SolverPhase phase(timing,"prepare_relations");point.problem=PrepareSymbolicProblem(*point.module,target,options.common.placement.dims,target.res.num_sms*residency,residency,kappa,nullptr,false);}
      if(!base)base=point.problem;
      if(!floor){floor=DeriveModelDramFloor(*point.module,point.problem.model,target,options.fixture);floor_attribute=(*point.module)->getAttr("tmexec.dram_floor");}
      else (*point.module)->setAttr("tmexec.dram_floor",floor_attribute);
    } else {
      std::vector<GemmConfig> geometry;for(auto const& g:granularity.gemms)geometry.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
      SolverPhase phase(timing,"instantiate_and_derive");auto geometry_key=ConfigKey(config,0,0);
      if(last_structure && last_geometry==geometry_key){point.problem=*last_structure;point.problem.projection.options.grid=target.res.num_sms*residency;point.problem.projection.options.kappa=kappa;}
      else{point.problem=PrepareFlowStructure(*base,geometry,target.res.num_sms*residency,kappa,cache,&flow_cache);last_structure=point.problem;last_geometry=std::move(geometry_key);}
    }
    {SolverPhase phase(timing,"piece_pricing_and_release");point.flow=PrepareFlow(point.problem,*floor,target,residency,options.common.placement.hop,cache,flow_cache);}
    return point;
  }
  SkeletonCandidate Evaluate(std::vector<GemmConfig> const& config,int kappa,int residency,int actual=0) {
    auto point=Prepare(config,kappa,residency,actual);
    {SolverPhase phase(options.common.timing,"flow");point.candidate.score=EvaluateFlow(point.flow->flow).makespan_ns;}
    return point.candidate;
  }
  SkeletonSolvedPoint Materialize(SkeletonCandidate const& candidate,bool pure,int actual=0) {
    auto point=Prepare(candidate.config,candidate.kappa,candidate.residency,actual,true);point.candidate.score=candidate.score;
    auto const& target=options.common.placement.target;
    ApplyFlowPrices(point.problem,*point.flow,target,point.candidate.residency);
    point.skeleton=BuildPlanSkeleton(point.problem,target.res.num_sms*point.candidate.residency,point.candidate.residency,options.k_base,options.all_workers,cache,options.common.timing);
    SkeletonRequest request;request.skeleton=&point.skeleton;request.hop=options.common.placement.hop;request.sms=point.skeleton.grid;request.pure_template=pure;
    std::string error;{SolverPhase phase(options.common.timing,"materialize");if(!ScheduleBySkeleton(request,&point.schedule,&point.candidate.placement,&error))throw std::runtime_error(error);}
    return point;
  }
};
std::vector<SkeletonCandidate> CoordinateDescent(SearchContext& search,int& rounds,std::ostream& out) {
  auto const& options=search.options;
  if(options.passes<1 || options.passes>3)throw std::invalid_argument("coordinate descent supports P=1..3");
  std::vector<SkeletonCandidate> evaluated;std::map<std::string,std::size_t> seen;
  auto evaluate=[&](std::vector<GemmConfig> const& config,int kappa,int residency)->std::size_t {
    auto key=ConfigKey(config,kappa,residency);auto old=seen.find(key);if(old!=seen.end())return old->second;
    SkeletonCandidate candidate;candidate.config=config;candidate.key=key;candidate.kappa=kappa;candidate.residency=residency;
    try{candidate=search.Evaluate(config,kappa,residency);}catch(std::exception const& e){candidate.error=e.what();}
    auto canonical=seen.find(candidate.key);if(canonical!=seen.end()){seen.emplace(key,canonical->second);return canonical->second;}
    std::size_t index=evaluated.size();seen.emplace(key,index);seen.emplace(candidate.key,index);evaluated.push_back(std::move(candidate));
    auto const& c=evaluated.back();out<<"EVALUATE\t"<<index<<'\t'<<c.key<<'\t'<<c.score<<'\t'<<c.residency<<'\t'<<c.estimated_limit<<'\t'<<c.error<<'\n';out.flush();return index;
  };
  std::vector<std::vector<GemmConfig>> domains;
  for(auto const& cls:search.classes) {
    auto domain=ClassCandidates(cls,search.imported,options.common.placement.target,search.dtype);
    if(!options.common.geometry_domain.empty())domain.erase(std::remove_if(domain.begin(),domain.end(),[&](auto const& g){return std::none_of(options.common.geometry_domain.begin(),options.common.geometry_domain.end(),[&](auto const& a){return std::tie(g.tile_m,g.tile_n,g.tile_k,g.stages)==std::tie(a.tile_m,a.tile_n,a.tile_k,a.stages);});}),domain.end());
    out<<"DOMAIN\t"<<domains.size()<<'\t'<<domain.size()<<'\n';domains.push_back(std::move(domain));
  }
  std::vector<GemmConfig> seed(search.classes.size(),options.seed);
  auto legacy=evaluate(seed,options.kappa,options.seed_residency);std::size_t uniform=legacy;
  // A uniform configuration must be legal for every operator class.
  for(auto const& g:domains.front()) {
    bool legal=true;for(auto const& domain:domains)legal &= std::any_of(domain.begin(),domain.end(),[&](auto const& other){return ClassGeometryKey(g)==ClassGeometryKey(other);});
    if(!legal)continue;std::vector<GemmConfig> config(search.classes.size(),g);
    auto limit=search.resources.Estimate(search.classes,config,options.common.placement.target,search.dtype).resident_limit;
    for(int k:{1,2,4})for(int r=1;r<=limit;++r){auto i=evaluate(config,k,r);if(evaluated[i].score<evaluated[uniform].score)uniform=i;}
  }
  for(std::size_t start:{legacy,uniform}) {
    auto incumbent=start;if(!std::isfinite(evaluated[incumbent].score))throw std::runtime_error("flow seed has no valid score: "+evaluated[incumbent].error);
    for(int pass=0;pass<options.passes;++pass){bool moved=false;++rounds;
      for(std::size_t c=0;c<search.classes.size();++c){auto fixed=evaluated[incumbent];int improvements=0;
        for(auto const& g:domains[c]){auto config=fixed.config;config[c]=g;auto i=evaluate(config,fixed.kappa,fixed.residency);if(evaluated[i].score<evaluated[incumbent].score){incumbent=i;moved=true;++improvements;}}
        out<<"COORDINATE\t"<<start<<'\t'<<pass<<'\t'<<c<<'\t'<<domains[c].size()<<'\t'<<improvements<<'\t'<<evaluated[incumbent].score<<'\n';out.flush();
      }
      auto fixed=evaluated[incumbent];for(int k:{1,2,4}){auto i=evaluate(fixed.config,k,fixed.residency);if(evaluated[i].score<evaluated[incumbent].score){incumbent=i;moved=true;}}
      fixed=evaluated[incumbent];for(int r=1;r<=fixed.estimated_limit;++r){auto i=evaluate(fixed.config,fixed.kappa,r);if(evaluated[i].score<evaluated[incumbent].score){incumbent=i;moved=true;}}
      if(!moved)break;
    }
  }
  std::stable_sort(evaluated.begin(),evaluated.end(),[](auto const& a,auto const& b){return a.score<b.score;});return evaluated;
}
}
SkeletonSearchResult SolveSkeletonExport(std::string const& path,mlir::MLIRContext& context,
    SkeletonSearchOptions const& options,frontend::ImportSummary* summary,std::ostream& evidence) {
  SolverPhase total(options.common.timing,"total");analysis::ScopedExactAnalysisMemo memo;frontend::TorchExportImporter importer;
  if(options.jobs!=1)throw std::invalid_argument("flow search is single-threaded; ISL and price caches belong to its thread");
  auto plan=[&]{SolverPhase phase(options.common.timing,"bridge_and_plan");auto b=frontend::ReadExportBridge(path);return frontend::BuildModelPlan(b.nodes,b.inputs,b.outputs);}();
  auto imported=[&]{SolverPhase phase(options.common.timing,"import");return importer.ImportSemantics(path,plan,context);}();
  return SolveSkeletonImported(imported,context,options,summary,evidence);
}
SkeletonSearchResult SolveSkeletonImported(frontend::ImportedSemantics const& imported,
    mlir::MLIRContext& context,SkeletonSearchOptions const& options,
    frontend::ImportSummary* summary,std::ostream& evidence) {
  analysis::ScopedExactAnalysisMemo memo;
  if(options.jobs!=1)throw std::invalid_argument("flow search is single-threaded");
  SearchContext search(imported,context,options);SkeletonSearchResult result;result.classes=search.classes;
  evidence<<std::setprecision(17);result.evaluated=CoordinateDescent(search,result.rounds,evidence);
  if(options.search_only) {
    std::ofstream floor(options.artifact_prefix+".floor.tsv");
    auto value=search.floor->Evaluate(search.base->model.MetricBindings());
    floor<<std::setprecision(17)<<"dram_ns\tcompute_ns\tfloor_ns\n"
         <<value.dram_ns<<'\t'<<value.compute_ns<<'\t'<<value.floor_ns<<'\n';
    if(auto* t=options.common.timing){t->Add("cache_hit",0,search.cache.hits);t->Add("cache_miss",0,search.cache.misses);t->Add("search_evaluations",0,result.evaluated.size());t->Add("search_rounds",0,result.rounds);}
    return result;
  }
  struct Materialized {CompilerSearchResult::ShortlistEntry entry;SkeletonCandidate candidate;bool pure;};std::vector<Materialized> materialized;
  std::ofstream table(options.artifact_prefix+".materializations.tsv");table<<"rank\tkey\tpure_ns\teft_ns\tpure_selected\tmoved_fraction\n";
  int rank=0;for(auto const& candidate:result.evaluated) {
    if(!candidate.error.empty())continue;if(rank==options.top_m)break;++rank;
    auto opts=options;opts.kappa=candidate.kappa;
    auto a=search.Materialize(candidate,true);auto a_stats=a.candidate.placement;
    auto ea=FinalizeSkeletonPoint(std::move(a),opts,options.artifact_prefix+".m"+std::to_string(rank)+"A");
    auto b=search.Materialize(candidate,false);auto b_stats=b.candidate.placement;
    auto eb=FinalizeSkeletonPoint(std::move(b),opts,options.artifact_prefix+".m"+std::to_string(rank)+"B");
    bool pure=options.pure_template || ea.evaluation.makespan_ns<=eb.evaluation.makespan_ns;
    table<<rank<<'\t'<<candidate.key<<'\t'<<ea.evaluation.makespan_ns<<'\t'<<eb.evaluation.makespan_ns<<'\t'<<pure<<'\t'<<(b_stats.placed?1.-double(b_stats.home)/b_stats.placed:0)<<'\n';table.flush();
    auto c=candidate;c.placement=pure?a_stats:b_stats;
    materialized.push_back({pure?std::move(ea):std::move(eb),c,pure});
  }
  std::stable_sort(materialized.begin(),materialized.end(),[](auto const& a,auto const& b){return a.entry.evaluation.makespan_ns<b.entry.evaluation.makespan_ns;});
  if(materialized.size()>3)materialized.resize(3);
  std::ofstream resources(options.artifact_prefix+".resources.tsv");resources<<"rank\tkey\testimated\tactual\tre_solved\tresidency\tflow_ns\tsimulated_ns\n";
  rank=0;for(auto& item:materialized) {
    auto& c=item.candidate;int actual;
    if(!options.common.query_residency)throw std::invalid_argument("top-3 requires real queryResidency");
    {SolverPhase phase(options.common.timing,"megakernel_compile");actual=options.common.query_residency(*item.entry.module,c.kappa);}
    if(actual<1)throw std::runtime_error("top-3 compiled with zero residency");
    bool changed=actual!=c.estimated_limit;
    if(changed) {
      // Correct occupancy can expose a better residency as well as invalidate one.
      auto best=search.Evaluate(c.config,c.kappa,1,actual);
      for(int r=2;r<=actual;++r) {
        auto trial=search.Evaluate(c.config,c.kappa,r,actual);
        if(trial.score<best.score)best=std::move(trial);
      }
      c=std::move(best);auto opts=options;opts.kappa=c.kappa;
      auto a=search.Materialize(c,true,actual),b=search.Materialize(c,false,actual);
      auto astats=a.candidate.placement,bstats=b.candidate.placement;
      auto ea=FinalizeSkeletonPoint(std::move(a),opts,options.artifact_prefix+".verified"+std::to_string(rank)+"A");
      auto eb=FinalizeSkeletonPoint(std::move(b),opts,options.artifact_prefix+".verified"+std::to_string(rank)+"B");
      item.pure=options.pure_template || ea.evaluation.makespan_ns<=eb.evaluation.makespan_ns;
      c.placement=item.pure?astats:bstats;item.entry=item.pure?std::move(ea):std::move(eb);
    }
    c.actual_limit=actual;resources<<++rank<<'\t'<<c.key<<'\t'<<c.estimated_limit<<'\t'<<actual<<'\t'<<changed<<'\t'<<c.residency<<'\t'<<c.score<<'\t'<<item.entry.evaluation.makespan_ns<<'\n';resources.flush();
    result.top.push_back(c);result.compiled.shortlist.push_back(std::move(item.entry));
  }
  if(result.top.empty())throw std::runtime_error("no admitted flow candidate");
  std::stable_sort(result.compiled.shortlist.begin(),result.compiled.shortlist.end(),[](auto const& a,auto const& b){return a.evaluation.makespan_ns<b.evaluation.makespan_ns;});
  auto& best=result.compiled.shortlist.front();result.compiled.module=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(best.module->clone()));result.compiled.winner=best.evaluation.candidate;
  auto selected=std::find_if(result.top.begin(),result.top.end(),[&](auto const& c){return c.key==best.evaluation.candidate.key;});result.compiled.winner_resident_limit=selected->actual_limit;result.compiled.stats.evaluated=result.evaluated.size();
  std::ofstream classes(options.artifact_prefix+".classes.tsv");classes<<"class\tgemm\top\ttile_m\ttile_n\ttile_k\tstages\tsplit_k\n";
  for(std::size_t c=0;c<search.classes.size();++c)for(std::size_t j=0;j<search.classes[c].gemms.size();++j){auto const& g=selected->config[c];classes<<c<<'\t'<<search.classes[c].gemms[j]<<'\t'<<search.classes[c].operators[j]<<'\t'<<g.tile_m<<'\t'<<g.tile_n<<'\t'<<g.tile_k<<'\t'<<g.stages<<'\t'<<g.split_k<<'\n';}
  if(summary){*summary={};summary->stages=search.imported.plan.stages.size();for(auto op:result.compiled.module->getOps<dialect::TileSpaceOp>())++summary->task_spaces;for(auto op:result.compiled.module->getOps<dialect::CouplingOp>())++summary->couplings;}
  if(auto* t=options.common.timing){t->Add("cache_hit",0,search.cache.hits);t->Add("cache_miss",0,search.cache.misses);t->Add("price_cache_hit",0,search.flow_cache.prices.hits);t->Add("release_cache_hit",0,search.flow_cache.release_hits);t->Add("search_evaluations",0,result.evaluated.size());t->Add("search_rounds",0,result.rounds);}
  return result;
}
}
