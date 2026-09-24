// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>

namespace tilemega::solver {
namespace {
std::string ConfigKey(std::vector<GemmConfig> const& config) {
  std::ostringstream out;for(auto const& g:config)out<<g.tile_m<<'x'<<g.tile_n<<'x'<<g.tile_k<<'s'<<g.stages<<'k'<<g.split_k<<';';return out.str();
}
// Alignment is established on symbolic couplings, before any concrete tile
// is visited. A split combiner retains its parent's semantic class.
int AlignedEdges(frontend::ImportedSemantics const& imported,std::vector<OperatorClass> const& classes,
    std::vector<GemmConfig> const& config,std::size_t changed,analysis::CouplingCache& cache) {
  auto options=ClassGranularity(imported,classes,config);auto lifted=imported.lifted;
  for(auto& op:lifted.ops)if(op.role==frontend::OpRole::kRoPE || op.role==frontend::OpRole::kKVAppend || op.role==frontend::OpRole::kActivation)op.ownership=frontend::OwnershipKind::kTilePerBlock;
  auto g=frontend::LaunchGranularity(lifted,imported.plan,options.gemms);
  auto graph=analysis::Instantiate(lifted.sem,g);analysis::ParamBinding known;
  known.Bind("Tm",config.front().tile_m).Bind("Tn",config.front().tile_n).Bind("Tkv",128);
  auto edges=cache.Derive(lifted.sem,graph,g,known);
  auto member=[&](std::string const& name){for(auto const& op:classes[changed].operators)if(name==op || name.rfind(op+".",0)==0)return true;return false;};
  int count=0;for(auto const& edge:edges)if(member(edge.src.name) || member(edge.dst.name))
    count+=cache.OracleFor(edge.C.Reverse().ToString())->structure==analysis::EdgeStructure::OneToOne;
  return count;
}
struct SearchContext {
  frontend::ImportedSemantics imported;
  frontend::TorchExportImporter importer;
  std::vector<OperatorClass> classes;
  analysis::CouplingCache cache;
  VariantResourceCache resources;
  mlir::MLIRContext& context;
  SkeletonSearchOptions const& options;
  ScalarType dtype;
  SearchContext(frontend::ImportedSemantics input,mlir::MLIRContext& ctx,SkeletonSearchOptions const& opts)
      :imported(std::move(input)),classes(BuildOperatorClasses(imported)),resources(opts.variant_probe,opts.common.timing),context(ctx),options(opts),
       dtype(imported.lifted.sem.ops.front().dtype==analysis::ScalarType::kBF16?ScalarType::kBF16:ScalarType::kF32) {}
  SkeletonSolvedPoint Evaluate(std::vector<GemmConfig> const& config,int fixed_limit=0) {
    auto* timing=options.common.timing;std::string key=ConfigKey(config);if(timing)timing->candidate=key;
    auto estimate=resources.Estimate(classes,config,options.common.placement.target,dtype);
    int limit=fixed_limit?fixed_limit:estimate.resident_limit;
    if(limit<1)throw std::invalid_argument("no resident CTA for geometry");
    auto module=importer.InstantiateForGranularity(imported,context,ClassGranularity(imported,classes,config),&cache,nullptr,timing);
    SymbolicPriceCache prices;
    SkeletonSolvedPoint best;best.candidate.config=config;best.candidate.key=key;best.candidate.estimated_limit=estimate.resident_limit;best.candidate.actual_limit=fixed_limit;
    for(int residency=1;residency<=limit;++residency) {
      int grid=options.common.placement.target.res.num_sms*residency;
      auto problem=[&]{SolverPhase phase(timing,"task_pricing");return PrepareSymbolicProblem(*module,
        options.common.placement.target,options.common.placement.dims,grid,residency,options.kappa,&prices);}();
      auto skeleton=BuildPlanSkeleton(problem,grid,residency,options.k_base,options.all_workers,cache,timing);
      SkeletonRequest request;request.skeleton=&skeleton;request.hop=options.common.placement.hop;
      // Prices already include residency, exactly as the legacy catalog's
      // observed-task input does. Avoid charging co-residency twice.
      request.sms=grid;request.ctas_per_sm=1;
      EftSchedule schedule;SkeletonPlacementStats stats;std::string error;
      auto started=std::chrono::steady_clock::now();
      {SolverPhase phase(timing,"level2");if(!ScheduleBySkeleton(request,&schedule,&stats,&error))throw std::runtime_error(error);}
      std::cerr<<"SKELETON_RESIDENCY key="<<key<<" residency="<<residency<<" nodes="<<skeleton.task_ns.size()
        <<" level2_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()
        <<" requeues="<<stats.lazy_requeues<<" score_ns="<<schedule.makespan_ns<<'\n';
      if(schedule.makespan_ns<best.candidate.score) {
        best.candidate.score=schedule.makespan_ns;best.candidate.residency=residency;best.candidate.placement=stats;
        best.problem=std::move(problem);best.skeleton=std::move(skeleton);best.schedule=std::move(schedule);
      }
    }
    best.module=std::move(module);return best;
  }
};
std::vector<SkeletonCandidate> CoordinateDescent(SearchContext& search,int& rounds,std::ostream& out) {
  auto const& options=search.options;
  if(options.passes<1 || options.passes>3)throw std::invalid_argument("coordinate descent supports P=1..3");
  std::vector<GemmConfig> current(search.classes.size(),options.seed);
  std::vector<SkeletonCandidate> evaluated;std::map<std::string,std::size_t> seen;
  auto evaluate=[&](std::vector<GemmConfig> const& g)->std::size_t {
    auto key=ConfigKey(g);auto found=seen.find(key);if(found!=seen.end())return found->second;
    SkeletonCandidate candidate;candidate.key=key;candidate.config=g;
    try {candidate=search.Evaluate(g).candidate;}catch(std::exception const& e){candidate.error=e.what();}
    std::size_t index=evaluated.size();seen.emplace(key,index);evaluated.push_back(candidate);
    out<<"EVALUATE\t"<<index<<'\t'<<key<<'\t'<<candidate.score<<'\t'<<candidate.residency<<'\t'<<candidate.estimated_limit<<'\t'<<candidate.error<<'\n';out.flush();return index;
  };
  std::size_t incumbent=evaluate(current);
  if(!evaluated[incumbent].error.empty())throw std::runtime_error("legacy seed rejected by skeleton: "+evaluated[incumbent].error);
  std::vector<std::vector<GemmConfig>> domains;
  for(auto const& cls:search.classes){auto domain=ClassCandidates(cls,search.imported,options.common.placement.target,search.dtype);
    // An explicit diagnostic domain is disclosed in the result; production
    // R9 measurements omit it and enumerate CandidateGenerator's whole domain.
    if(!options.common.geometry_domain.empty())domain.erase(std::remove_if(domain.begin(),domain.end(),[&](auto const& g){
      return std::none_of(options.common.geometry_domain.begin(),options.common.geometry_domain.end(),[&](auto const& allowed){
        return std::tie(g.tile_m,g.tile_n,g.tile_k,g.stages)==std::tie(allowed.tile_m,allowed.tile_n,allowed.tile_k,allowed.stages);});}),domain.end());
    domains.push_back(std::move(domain));}
  for(int pass=0;pass<options.passes;++pass){bool moved=false;++rounds;
    for(std::size_t c=0;c<search.classes.size();++c){
      std::vector<std::pair<int,GemmConfig>> ordered;
      {SolverPhase phase(options.common.timing,"derive_alignment");
        for(auto const& g:domains[c]){auto trial=current;trial[c]=g;int alignment=0;
          try {alignment=AlignedEdges(search.imported,search.classes,trial,c,search.cache);}catch(std::exception const& e){out<<"ALIGNMENT_REJECT\t"<<ConfigKey(trial)<<'\t'<<e.what()<<'\n';}
          ordered.emplace_back(alignment,g);}}
      std::stable_sort(ordered.begin(),ordered.end(),[](auto const& a,auto const& b){return a.first>b.first;});
      int improvements=0,aligned=0;
      // Keep all other classes fixed while sweeping this coordinate.
      auto fixed=current;
      for(auto const& [alignment,g]:ordered){auto trial=fixed;trial[c]=g;auto index=evaluate(trial);
        if(evaluated[index].score<evaluated[incumbent].score){incumbent=index;current=trial;moved=true;++improvements;aligned+=alignment>0;}}
      out<<"COORDINATE\t"<<pass<<'\t'<<c<<'\t'<<ordered.size()<<'\t'<<improvements<<'\t'<<aligned<<'\t'<<evaluated[incumbent].score<<'\n';out.flush();
    }
    if(!moved)break;
  }
  std::stable_sort(evaluated.begin(),evaluated.end(),[](auto const& a,auto const& b){return a.score<b.score;});return evaluated;
}
}
SkeletonSearchResult SolveSkeletonExport(std::string const& path,mlir::MLIRContext& context,
    SkeletonSearchOptions const& options,frontend::ImportSummary* summary,std::ostream& evidence) {
  SolverPhase total(options.common.timing,"total");frontend::TorchExportImporter importer;
  auto plan=[&]{SolverPhase phase(options.common.timing,"bridge_and_plan");auto b=frontend::ReadExportBridge(path);return frontend::BuildModelPlan(b.nodes,b.inputs,b.outputs);}();
  auto imported=[&]{SolverPhase phase(options.common.timing,"import");return importer.ImportSemantics(path,plan,context);}();
  SearchContext search(std::move(imported),context,options);SkeletonSearchResult result;result.classes=search.classes;
  evidence<<std::setprecision(17);result.evaluated=CoordinateDescent(search,result.rounds,evidence);
  auto write_classes=[&](std::string const& path,std::vector<GemmConfig> const& config){
    std::ofstream classes(path);classes<<"class\tgemm\top\ttile_m\ttile_n\ttile_k\tstages\tsplit_k\tseed_m\tseed_n\tseed_k\tseed_stages\tseed_split\n";
  for(std::size_t c=0;c<search.classes.size();++c)for(std::size_t j=0;j<search.classes[c].gemms.size();++j){auto const& g=config[c];auto const& seed=options.seed;
    classes<<c<<'\t'<<search.classes[c].gemms[j]<<'\t'<<search.classes[c].operators[j]<<'\t'<<g.tile_m<<'\t'<<g.tile_n<<'\t'<<g.tile_k<<'\t'<<g.stages<<'\t'<<g.split_k<<'\t'<<seed.tile_m<<'\t'<<seed.tile_n<<'\t'<<seed.tile_k<<'\t'<<seed.stages<<'\t'<<seed.split_k<<'\n';}
  };
  std::ofstream resources(options.artifact_prefix+".resources.tsv");resources<<"rank\tkey\testimated\tactual\tre_solved\tresidency\tlevel2_ns\tsimulated_ns\n";
  for(auto const& candidate:result.evaluated) {
    if(!candidate.error.empty())continue;if(result.top.size()==5)break;
    auto point=search.Evaluate(candidate.config);
    if(!options.common.query_residency)throw std::invalid_argument("top-K requires real queryResidency");
    int actual=[&]{SolverPhase phase(options.common.timing,"megakernel_compile");return options.common.query_residency(*point.module,options.kappa);}();
    bool changed=actual!=point.candidate.estimated_limit;
    if(actual<1)throw std::runtime_error("top-K compiled with zero residency");
    if(changed)point=search.Evaluate(candidate.config,actual);
    point.candidate.actual_limit=actual;int rank=int(result.top.size())+1;
    result.top.push_back(point.candidate);
    write_classes(options.artifact_prefix+".final"+std::to_string(rank)+".classes.tsv",candidate.config);
    auto entry=FinalizeSkeletonPoint(std::move(point),options,options.artifact_prefix+".final"+std::to_string(rank));
    resources<<rank<<'\t'<<candidate.key<<'\t'<<candidate.estimated_limit<<'\t'<<actual<<'\t'<<changed<<'\t'<<result.top.back().residency<<'\t'<<result.top.back().score<<'\t'<<entry.evaluation.makespan_ns<<'\n';resources.flush();
    result.compiled.shortlist.push_back(std::move(entry));
  }
  if(result.top.empty())throw std::runtime_error("no admitted Skeleton candidate");
  std::stable_sort(result.compiled.shortlist.begin(),result.compiled.shortlist.end(),[](auto const& a,auto const& b){return a.evaluation.makespan_ns<b.evaluation.makespan_ns;});
  if(result.compiled.shortlist.size()>3)result.compiled.shortlist.resize(3);
  auto& best=result.compiled.shortlist.front();result.compiled.module=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(best.module->clone()));
  result.compiled.winner=best.evaluation.candidate;
  auto selected=std::find_if(result.top.begin(),result.top.end(),[&](auto const& c){return c.key==best.evaluation.candidate.key;});
  result.compiled.winner_resident_limit=selected->actual_limit;
  result.compiled.stats.evaluated=result.evaluated.size();
  if(summary){*summary={};summary->stages=search.imported.plan.stages.size();
    for(auto op:result.compiled.module->getOps<dialect::TileSpaceOp>())++summary->task_spaces;
    for(auto op:result.compiled.module->getOps<dialect::CouplingOp>())++summary->couplings;
  }
  write_classes(options.artifact_prefix+".classes.tsv",selected->config);
  if(options.common.timing){auto* t=options.common.timing;t->phases["cache_hit"].count=search.cache.hits;t->phases["cache_miss"].count=search.cache.misses;
    for(auto const& [key,pair]:search.cache.oracle_entries)for(auto const* oracle:{&pair->forward,&pair->reverse}){
      std::string kind=analysis::ToString(oracle->kind());std::transform(kind.begin(),kind.end(),kind.begin(),::tolower);
      t->Add("oracle_"+kind,oracle->query_ms(),oracle->queries());
      t->Add("oracle",oracle->query_ms(),oracle->queries());}
    t->Add("search_evaluations",0,result.evaluated.size());t->Add("search_rounds",0,result.rounds);
  }
  return result;
}
}
