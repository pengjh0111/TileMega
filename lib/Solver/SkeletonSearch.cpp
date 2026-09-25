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
  struct FlowSnapshot {
    std::vector<GemmConfig> config;
    int residency=0;
    PreparedFlow prepared;
  };
  std::map<int,FlowSnapshot> recent_flows;
  mlir::Attribute floor_attribute;
  ScalarType dtype;
  SearchContext(frontend::ImportedSemantics input,mlir::MLIRContext& ctx,SkeletonSearchOptions const& opts)
      :imported(std::move(input)),classes(BuildOperatorClasses(imported)),resources(opts.variant_probe,opts.common.timing),context(ctx),options(opts),
       dtype(imported.lifted.sem.ops.front().dtype==analysis::ScalarType::kBF16?ScalarType::kBF16:ScalarType::kF32) {}
  SkeletonSolvedPoint Prepare(std::vector<GemmConfig> const& config,int kappa,int residency,int actual=0,bool materialize=false,int past_override=-1) {
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
      SolverPhase phase(timing,"instantiate_and_derive");auto geometry_key=ConfigKey(config,0,0)+":"+std::to_string(past_override);
      if(last_structure && last_geometry==geometry_key){point.problem=*last_structure;point.problem.projection.options.grid=target.res.num_sms*residency;point.problem.projection.options.kappa=kappa;}
      else{
        auto source=*base;
        if(past_override>=0) {
          source.model.dims.past=past_override;
          source.model.dims.total=source.model.dims.seq+past_override;
        }
        point.problem=PrepareFlowStructure(source,geometry,target.res.num_sms*residency,kappa,cache,&flow_cache);
        last_structure=point.problem;last_geometry=std::move(geometry_key);
      }
    }
    PreparedFlow const* prior=nullptr;
    std::vector<bool> reusable(imported.plan.stages.size(),false);
    int const past_key=past_override>=0?past_override:-1;
    {
      SolverPhase phase(timing,"incremental_prepare");
      auto previous=recent_flows.find(past_key);
      if(options.incremental_prepare && !materialize &&
         previous!=recent_flows.end() &&
         previous->second.residency==residency &&
         previous->second.config.size()==config.size()) {
        prior=&previous->second.prepared;
        std::fill(reusable.begin(),reusable.end(),true);
        for(std::size_t c=0;c<classes.size();++c)
          if(ClassGeometryKey(previous->second.config[c])!=ClassGeometryKey(config[c]))
            for(std::size_t s=0;s<imported.plan.stages.size();++s) {
              int gemm=imported.plan.stages[s].gemm;
              if(gemm>=0 && std::find(classes[c].gemms.begin(),classes[c].gemms.end(),
                  std::size_t(gemm))!=classes[c].gemms.end())reusable[s]=false;
            }
      }
    }
    {SolverPhase phase(timing,"piece_pricing_and_release");point.flow=PrepareFlow(
        point.problem,*floor,target,residency,options.common.placement.hop,
        cache,flow_cache,true,estimate.shared_bytes,prior,prior?&reusable:nullptr);}
    if(options.incremental_prepare && !materialize)
      recent_flows[past_key]={config,residency,*point.flow};
    return point;
  }
  SkeletonCandidate Evaluate(std::vector<GemmConfig> const& config,int kappa,int residency,int actual=0) {
    auto point=Prepare(config,kappa,residency,actual);
    {SolverPhase phase(options.common.timing,"flow");point.candidate.score=EvaluateFlow(point.flow->flow).makespan_ns;}
    if(options.serving_past_lo>=0 && options.serving_past_hi>=options.serving_past_lo) {
      auto low=Prepare(config,kappa,residency,actual,false,options.serving_past_lo);
      auto high=Prepare(config,kappa,residency,actual,false,options.serving_past_hi);
      SolverPhase phase(options.common.timing,"flow");
      point.candidate.score=(EvaluateFlow(low.flow->flow).makespan_ns+
          4*point.candidate.score+EvaluateFlow(high.flow->flow).makespan_ns)/6;
    }
    return point.candidate;
  }
  SkeletonSolvedPoint Materialize(SkeletonCandidate const& candidate,bool pure,int actual=0) {
    auto point=Prepare(candidate.config,candidate.kappa,candidate.residency,actual,true);point.candidate.score=candidate.score;
    if(options.serving_past_lo>=0 &&
       options.serving_past_hi>=options.serving_past_lo) {
      for(int past:{options.serving_past_lo,options.serving_past_hi}) {
        auto endpoint=Prepare(candidate.config,candidate.kappa,
            candidate.residency,actual,false,past);
        for(std::size_t s=0;s<point.problem.counts.size();++s)
          if(point.problem.counts[s]!=endpoint.problem.counts[s])
            throw std::runtime_error("serving interval changes a task-space count");
        point.interval_flows.emplace_back(past,std::move(*endpoint.flow));
      }
    }
    auto const& target=options.common.placement.target;
    ApplyFlowPrices(point.problem,*point.flow,target,point.candidate.residency);
    point.skeleton=BuildPlanSkeleton(point.problem,target.res.num_sms*point.candidate.residency,point.candidate.residency,options.k_base,options.all_workers,cache,options.common.timing);
    SkeletonRequest request;request.skeleton=&point.skeleton;request.hop=options.common.placement.hop;request.sms=point.skeleton.grid;request.pure_template=pure;
    std::string error;{SolverPhase phase(options.common.timing,"materialize");if(!ScheduleBySkeleton(request,&point.schedule,&point.candidate.placement,&error))throw std::runtime_error(error);}
    return point;
  }
};
GemmConfig ServingSeed(OperatorClass const& cls,
    frontend::ImportedSemantics const& imported,
    TargetSpec const& target,int batch,int seq,
    std::vector<GemmConfig> const& domain) {
  if(domain.empty())throw std::invalid_argument("serving class has no legal geometry");
  auto id=cls.gemms.front();
  auto stage=std::find_if(imported.plan.stages.begin(),imported.plan.stages.end(),
      [&](auto const& s){return s.kind==frontend::PlanTaskKind::kGemm && s.gemm==id;});
  if(stage==imported.plan.stages.end())throw std::invalid_argument("serving seed has no GEMM stage");
  int rows=stage->batch_rows?batch:batch*seq;
  int columns=int(imported.plan.gemms.at(id).n);
  int worker_limit=target.res.num_sms*
      std::max(1,target.res.max_threads_per_sm/kServingBF16Threads);
  int const output_tiles=((rows+15)/16)*((columns+127)/128);
  for(int tile_k:{128,64}) {
    int chosen_split=0;
    for(auto const& g:domain)
      if(g.tile_m==16 && g.tile_n==128 && g.tile_k==tile_k &&
         output_tiles*g.split_k*2>=worker_limit &&
         (chosen_split==0 || g.split_k<chosen_split))chosen_split=g.split_k;
    if(!chosen_split)for(auto const& g:domain)
      if(g.tile_m==16 && g.tile_n==128 && g.tile_k==tile_k)
        chosen_split=std::max(chosen_split,g.split_k);
    if(!chosen_split)continue;
    auto best=domain.end();
    for(auto it=domain.begin();it!=domain.end();++it)
      if(it->tile_m==16 && it->tile_n==128 && it->tile_k==tile_k &&
         it->split_k==chosen_split &&
         (best==domain.end() || it->stages>best->stages))best=it;
    if(best!=domain.end())return *best;
  }
  return domain.front();
}
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
    std::vector<GemmConfig> domain;
    if(search.imported.plan.serving) {
      auto pruned=ServingClassCandidates(cls,search.imported,
          options.common.placement.target,options.common.placement.dims.batch,
          options.common.placement.dims.seq);
      domain=std::move(pruned.candidates);
      out<<"PRUNING\t"<<domains.size()<<'\t'<<pruned.raw<<'\t'
         <<pruned.removed_r1<<'\t'<<pruned.removed_r2<<'\t'
         <<pruned.removed_r3<<'\t'<<domain.size()<<'\n';
    }else domain=ClassCandidates(cls,search.imported,options.common.placement.target,search.dtype);
    if(!options.common.geometry_domain.empty())domain.erase(std::remove_if(domain.begin(),domain.end(),[&](auto const& g){return std::none_of(options.common.geometry_domain.begin(),options.common.geometry_domain.end(),[&](auto const& a){return std::tie(g.tile_m,g.tile_n,g.tile_k,g.stages)==std::tie(a.tile_m,a.tile_n,a.tile_k,a.stages);});}),domain.end());
    out<<"DOMAIN\t"<<domains.size()<<'\t'<<domain.size()<<'\n';domains.push_back(std::move(domain));
  }
  std::vector<GemmConfig> seed(search.classes.size(),options.seed);
  if(search.imported.plan.serving)
    for(std::size_t c=0;c<seed.size();++c)
      seed[c]=ServingSeed(search.classes[c],search.imported,
          options.common.placement.target,
          options.common.placement.dims.batch,
          options.common.placement.dims.seq,domains[c]);
  int seed_residency=options.seed_residency;
  if(search.imported.plan.serving)seed_residency=std::max(1,
      search.resources.Estimate(search.classes,seed,
          options.common.placement.target,search.dtype).resident_limit);
  auto legacy=evaluate(seed,search.imported.plan.serving?1:options.kappa,
      seed_residency);std::size_t uniform=legacy;
  // A uniform configuration must be legal for every operator class.
  if(!search.imported.plan.serving)for(auto const& g:domains.front()) {
    bool legal=true;for(auto const& domain:domains)legal &= std::any_of(domain.begin(),domain.end(),[&](auto const& other){return ClassGeometryKey(g)==ClassGeometryKey(other);});
    if(!legal)continue;std::vector<GemmConfig> config(search.classes.size(),g);
    auto limit=search.resources.Estimate(search.classes,config,options.common.placement.target,search.dtype).resident_limit;
    for(int k:{1,2,4})for(int r=1;r<=limit;++r){auto i=evaluate(config,k,r);if(evaluated[i].score<evaluated[uniform].score)uniform=i;}
  }
  std::vector<std::size_t> starts{legacy};
  if(uniform!=legacy)starts.push_back(uniform);
  for(std::size_t start:starts) {
    auto incumbent=start;if(!std::isfinite(evaluated[incumbent].score))throw std::runtime_error("flow seed has no valid score: "+evaluated[incumbent].error);
    for(int pass=0;pass<options.passes;++pass){bool moved=false;++rounds;
      for(std::size_t c=0;c<search.classes.size();++c){auto fixed=evaluated[incumbent];int improvements=0;
        for(auto const& g:domains[c]){auto config=fixed.config;config[c]=g;
          int residency=fixed.residency;
          int kappa=fixed.kappa;
          if(search.imported.plan.serving) {
            auto limit=search.resources.Estimate(search.classes,config,
                options.common.placement.target,search.dtype).resident_limit;
            if(limit<1)continue;
            residency=limit;
            kappa=1;
          }
          auto i=evaluate(config,kappa,residency);
          if(evaluated[i].score<evaluated[incumbent].score){incumbent=i;moved=true;++improvements;}}
        out<<"COORDINATE\t"<<start<<'\t'<<pass<<'\t'<<c<<'\t'<<domains[c].size()<<'\t'<<improvements<<'\t'<<evaluated[incumbent].score<<'\n';out.flush();
      }
      auto fixed=evaluated[incumbent];
      auto serving_order=MakeServingSearchOrderR4(fixed.estimated_limit);
      for(int k:serving_order.kappa_scan){auto i=evaluate(fixed.config,k,fixed.residency);if(evaluated[i].score<evaluated[incumbent].score){incumbent=i;moved=true;}}
      fixed=evaluated[incumbent];
      serving_order=MakeServingSearchOrderR4(fixed.estimated_limit);
      for(int r:serving_order.residency_scan){auto i=evaluate(fixed.config,fixed.kappa,r);if(evaluated[i].score<evaluated[incumbent].score){incumbent=i;moved=true;}}
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
  evidence<<std::setprecision(17);
  if(!options.evaluation_cases.empty()) {
    // A fixed baseline creates the symbolic ModelDescription once.  Random
    // cases then change only their explicit class coordinates, just as the
    // production coordinate-descent path does after its seed evaluation.
    std::vector<GemmConfig> seed(search.classes.size(),options.seed);
    search.Evaluate(seed,options.kappa,options.seed_residency);
  }
  if(options.evaluation_cases.empty())
    result.evaluated=CoordinateDescent(search,result.rounds,evidence);
  else for(std::size_t i=0;i<options.evaluation_cases.size();++i) {
    auto const& test=options.evaluation_cases[i];
    try {result.evaluated.push_back(search.Evaluate(test.config,test.kappa,test.residency));}
    catch(std::exception const& e) {
      SkeletonCandidate failed;failed.config=test.config;failed.kappa=test.kappa;
      failed.residency=test.residency;failed.error=e.what();result.evaluated.push_back(std::move(failed));
    }
    auto const& candidate=result.evaluated.back();
    evidence<<"EVALUATE\t"<<i<<'\t'<<candidate.key<<'\t'
            <<candidate.score<<'\t'<<candidate.residency<<'\t'
            <<candidate.estimated_limit<<'\t'<<candidate.error<<'\n';
    evidence.flush();
  }
  if(options.search_only) {
    std::ofstream floor(options.artifact_prefix+".floor.tsv");
    auto value=search.floor->Evaluate(search.base->model.MetricBindings());
    floor<<std::setprecision(17)<<"dram_ns\tcompute_ns\tfloor_ns\n"
         <<value.dram_ns<<'\t'<<value.compute_ns<<'\t'<<value.floor_ns<<'\n';
    if(auto* t=options.common.timing){
      t->Add("cache_hit",0,search.cache.hits);t->Add("cache_miss",0,search.cache.misses);
      t->Add("space_cache_hit",0,search.flow_cache.space_hits);
      t->Add("space_cache_miss",0,search.flow_cache.space_misses);
      t->Add("price_cache_hit",0,search.flow_cache.prices.hits);
      t->Add("price_cache_miss",0,search.flow_cache.prices.misses);
      t->Add("release_cache_hit",0,search.flow_cache.release_hits);
      t->Add("prepare_spaces",search.flow_cache.spaces_ms,0);
      t->Add("prepare_graph",search.flow_cache.graph_ms,0);
      t->Add("prepare_derive",search.flow_cache.derive_ms,0);
      t->Add("prepare_price",search.flow_cache.price_ms,0);
      t->Add("prepare_piece_map",search.flow_cache.piece_map_ms,0);
      t->Add("prepare_edges",search.flow_cache.edges_ms,0);
      t->Add("search_evaluations",0,result.evaluated.size());t->Add("search_rounds",0,result.rounds);
    }
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
    bool pure=options.pure_template ||
        (search.imported.plan.serving
          ? eb.evaluation.makespan_ns>0.98*ea.evaluation.makespan_ns
          : ea.evaluation.makespan_ns<=eb.evaluation.makespan_ns);
    table<<rank<<'\t'<<candidate.key<<'\t'<<ea.evaluation.makespan_ns<<'\t'<<eb.evaluation.makespan_ns<<'\t'<<pure<<'\t'<<(b_stats.placed?double(b_stats.moved_from_home)/b_stats.placed:0)<<'\n';table.flush();
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
      item.pure=options.pure_template ||
          (search.imported.plan.serving
            ? eb.evaluation.makespan_ns>0.98*ea.evaluation.makespan_ns
            : ea.evaluation.makespan_ns<=eb.evaluation.makespan_ns);
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
