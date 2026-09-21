// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Solver/JointPlacement.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/VariantSchedule.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <limits>
#include <set>
#include <tilemega/Analysis/VisitFiniteRelation.h>
#include <tilemega/Solver/RelationBounds.h>

namespace tilemega::dialect {
// Scoped to one compile invocation. Keys include the complete immutable CG,
// target calibration, bound theta and compiled residency.
struct PlacementTaskPriceCache { std::map<std::string,std::vector<double>> prices; };
struct PlacementSolveOptions {
  TargetSpec target;
  solver::ModelDims dims;
  int residency=1;
  int kappa=1;
  /// Per-producer-stage coarsening over the projected stages (§6 B2). Empty
  /// keeps every stage on `kappa` and writes no per-stage attribute, so a
  /// solve that does not use the dimension emits what it always emitted.
  std::vector<int> stage_kappa;
  int verified_resident_limit=0;
  int requested_grid=0; // Zero uses the full compiled resident grid.
  /// The executor's `TILEMEGA_PREFETCH_PAGE_BYTES`; a prefetch that does not
  /// fit it is never issued, so it is never credited either.
  int prefetch_page_bytes=1024;
  solver::HopCurve hop;
  std::shared_ptr<PlacementTaskPriceCache> task_price_cache;
};
struct PlacementSolveResult {
  std::vector<solver::PlacementEvaluation> candidates;
  std::vector<solver::GemmConfig> geometry;
  int grid=0,kappa=1;
};
inline void WriteSolvedPlacement(mlir::ModuleOp module,
    solver::PlacementEvaluation const& selected,PlacementSolveOptions const& options) {
  mlir::OpBuilder b(module.getContext());
  int grid=int(selected.plan.queue.size());
  module->removeAttr(kPlacementTableAttr);
  if (selected.mode==PlacementMode::kEft) {
    std::vector<std::int64_t> worker,slot,pipeline;
    for (std::size_t s=0;s<selected.plan.owner.size();++s)
      for (std::size_t t=0;t<selected.plan.owner[s].size();++t) {
        worker.push_back(selected.plan.owner[s][t]);slot.push_back(selected.plan.slot[s][t]);
      }
    for (unsigned char flag:selected.pipeline) pipeline.push_back(flag);
    std::vector<mlir::NamedAttribute> fields{
      b.getNamedAttr("worker",b.getDenseI64ArrayAttr(worker)),b.getNamedAttr("slot",b.getDenseI64ArrayAttr(slot)),
      b.getNamedAttr("seq",b.getI64IntegerAttr(options.dims.seq)),b.getNamedAttr("past",b.getI64IntegerAttr(options.dims.past)),
      b.getNamedAttr("grid",b.getI64IntegerAttr(grid))};
    // A plan that pipelines nothing writes no field at all, so a model without
    // a read-only frontier keeps byte-identical CG (H2).
    if (std::find(pipeline.begin(),pipeline.end(),1)!=pipeline.end())
      fields.push_back(b.getNamedAttr("pipeline",b.getDenseI64ArrayAttr(pipeline)));
    module->setAttr(kPlacementTableAttr,b.getDictionaryAttr(fields));
  }
  for (auto placement:module.getOps<PlacementOp>()) {
    placement->removeAttr("mapping_mode");placement->removeAttr("params_map");
    placement->setAttr("mode",b.getStringAttr(PlacementModeName(selected.mode)));
    placement->setAttr("params",b.getDenseI64ArrayAttr(selected.params));
    placement->setAttr("window",b.getI64IntegerAttr(1));
    placement->setAttr("policy",b.getStringAttr(kPlacementPolicyAot));
    placement->setAttr("resident_only",b.getBoolAttr(true));
    auto function=analysis::CouplingRelation::FromIslText("[S,past] -> { [] -> ["+
        std::to_string(grid)+"] : S="+std::to_string(options.dims.seq)+
        " and past="+std::to_string(options.dims.past)+" }");
    placement->setAttr("grid_map",CouplingMapAttr::get(module.getContext(),function));
    placement->setAttr("resident_limit_map",CouplingMapAttr::get(module.getContext(),function));
  }
  module->setAttr("tmexec.solved_placement",b.getStringAttr(selected.name));
  // R8 BE-1: the architecture the Plan was priced and compiled for travels
  // with the Plan. Codegen turns it into the arch tag the TaskBodies are
  // instantiated on, and the harness refuses a device that disagrees.
  module->setAttr("tmexec.solved_arch",b.getStringAttr(options.target.arch_tag));
  module->setAttr("tmexec.solved_kappa",b.getI64IntegerAttr(options.kappa));
  if (!options.stage_kappa.empty())
    module->setAttr("tmexec.solved_stage_kappa",b.getDenseI64ArrayAttr(
        std::vector<std::int64_t>(options.stage_kappa.begin(),options.stage_kappa.end())));
  module->setAttr("tmexec.solved_residency",b.getI64IntegerAttr(options.residency));
  module->setAttr("tmexec.solved_seq",b.getI64IntegerAttr(options.dims.seq));
  module->setAttr("tmexec.solved_past",b.getI64IntegerAttr(options.dims.past));
  module->setAttr("tmexec.solved_grid",b.getI64IntegerAttr(grid));
  module->setAttr("tmexec.solved_floor_ns",b.getF64FloatAttr(selected.bounds.lower_bound_ns));
  if (mlir::failed(mlir::verify(module))) throw std::invalid_argument("solved placement failed CG verification");
}

struct PreparedPlacementProblem {
  codegen::RuntimePlan runtime;
  solver::ModelDescription model;
  std::vector<solver::GemmConfig> geometry;
  int grid=0,threads=0;
  solver::RuntimeProjection projection;
  std::vector<int> counts;
  codegen::RuntimeTaskGraph graph;
  std::vector<double> task_ns,prefetch_ns;
};
/// With `bounds`, the caller wants only work and the semantic critical path,
/// so the dense edge set is never materialized: the relation's own intervals
/// carry the longest-path relaxation and `graph.successors` stays empty. Every
/// other field is what the materializing path produces.
inline PreparedPlacementProblem PreparePlacementProblem(mlir::ModuleOp module,
    PlacementSolveOptions const& options,solver::RelationBounds* bounds=nullptr) {
  using namespace solver;
  if (!module || mlir::failed(mlir::verify(module)) || options.dims.seq<=0 ||
      options.dims.past<0 || options.residency<=0 || options.kappa<=0)
    throw std::invalid_argument("placement solve requires verified CG, bound theta and positive kappa");
  auto runtime=codegen::ReadRuntimePlan(module);
  auto model=ModelDescription::FromCouplingGraph(module,options.dims,"placement-pass");
  PlacementSolveResult result;
  int resident_grid=options.target.res.num_sms*options.residency;
  if(options.requested_grid<0 || options.requested_grid>resident_grid)
    throw std::invalid_argument("requested grid exceeds the compiled resident grid");
  result.grid=options.requested_grid ? options.requested_grid : resident_grid;
  for (auto const& g:runtime.gemms) result.geometry.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
  int threads=0,max_shared=0;
  for (std::size_t s=0;s<model.stages.size();++s) {
    auto const& stage=model.stages[s];
    auto const& g=result.geometry.at(stage.IsCollective() ? stage.gemm : 0);
    auto traits=ModelTaskTraits(model,int(s),g);
    threads=std::max(threads,traits.threads);max_shared=std::max(max_shared,traits.smem_bytes);
  }
  // Higher residency must come from the compiler's whole-kernel occupancy
  // query, never from the maximum of separate TaskBody resource estimates.
  if (options.residency>std::max(1,options.verified_resident_limit) ||
      max_shared>options.target.res.max_dynamic_smem_per_cta)
    throw std::invalid_argument("resident grid requires compiler-confirmed resource metadata");
  RuntimeProjectionOptions po{result.grid,threads,options.kappa};po.count_wait_entries=false;
  po.stage_kappa=options.stage_kappa;
  auto projection=ProjectRuntimeQueues(model,runtime,po);
  std::vector<int> counts;
  for (auto const& stage:projection.stages) counts.push_back(int(stage.task_count.Eval({})));
  // The table is indexed by projected stage, and a gemm expands into more than
  // one of those, so a table sized to the model's stages would coarsen the
  // wrong producers and leave the tail at the global value. Refuse it.
  if (!options.stage_kappa.empty() && options.stage_kappa.size()!=counts.size())
    throw std::invalid_argument("per-stage kappa table does not cover the projected stages");
  auto graph=codegen::MaterializeRuntimeTaskGraph(counts,{},result.grid);
  auto node=[&](long stage,long task) {
    if (stage<0 || stage>=long(counts.size()) || task<0 || task>=counts[stage])
      throw std::invalid_argument("projected relation outside task domain");
    return graph.stage_offsets[stage]+task;
  };
  if (!bounds) {
    analysis::VisitFiniteRelation(analysis::SharedIslContext(),
        projection.dependencies.ToString(),4,[&](long const* e) {
          graph.successors[node(e[2],e[3])].push_back(node(e[0],e[1]));
        });
    for (auto& row:graph.successors) {
      std::sort(row.begin(),row.end());row.erase(std::unique(row.begin(),row.end()),row.end());
    }
  }
  std::string price_key;
  std::vector<double> const* cached_prices=nullptr;
  if (options.task_price_cache) {
    llvm::raw_string_ostream text(price_key);module.print(text);text.flush();
    price_key+=options.target.ToJson()+"|"+std::to_string(options.dims.seq)+"|"+
        std::to_string(options.dims.past)+"|"+std::to_string(options.dims.total)+"|"+std::to_string(threads)+"|"+std::to_string(options.residency)+"|"+std::to_string(options.prefetch_page_bytes);
    for (int n:counts)price_key+="|"+std::to_string(n);
    auto found=options.task_price_cache->prices.find(price_key);
    if(found!=options.task_price_cache->prices.end())cached_prices=&found->second;
  }
  std::optional<analysis::OperatorGraph> semantic_graph;
  if(!cached_prices)semantic_graph=InstantiateModelTasks(model,result.geometry);
  SimulatorInput input;input.graph=&graph;input.task_ns.resize(graph.successors.size());
  input.prefetch_ns.resize(graph.successors.size());
  CostModel cost(options.target,model.dtype);
  auto theta=model.MetricBindings();
  for (std::size_t s=0;s<counts.size();++s) {
    auto const& projected=projection.stages[s];auto const& stage=model.stages[projected.logical_stage];
    auto const& g=result.geometry.at(stage.IsCollective() ? stage.gemm : 0);
    if(cached_prices) {
      auto const nodes=std::ptrdiff_t(input.task_ns.size());
      std::copy(cached_prices->begin()+graph.stage_offsets[s],cached_prices->begin()+graph.stage_offsets[s+1],
                input.task_ns.begin()+graph.stage_offsets[s]);
      std::copy(cached_prices->begin()+nodes+graph.stage_offsets[s],
                cached_prices->begin()+nodes+graph.stage_offsets[s+1],
                input.prefetch_ns.begin()+graph.stage_offsets[s]);continue;
    }
    if (projected.combine) {
      auto task=DeriveCombineTaskInput(model,projected.logical_stage,g,*semantic_graph,threads,
          runtime.ownership_flags & codegen::kCombinerTileOwnership,cost.options().fp32_partials);
      auto resources=codegen::ReadSimtTaskResources(codegen::TaskKind::kGemmCombine,threads);
      BackendTraits traits;traits.threads=resources.threads;traits.smem_bytes=resources.shared_bytes;traits.shape_legal=true;
      std::vector<analysis::ParamBinding> coordinates(counts[s]);
      for (int t=0;t<counts[s];++t) coordinates[t].Bind("q",t);
      if(task.work.task_count.Eval(theta)!=counts[s])
        throw std::invalid_argument("combine access ownership disagrees with projected count");
      std::vector<double> prefetch;PrefetchPricing pricing{options.prefetch_page_bytes,&prefetch};
      auto prices=PriceTaskInstances(cost,task,traits,{options.residency},model,1,coordinates,options.residency,&pricing);
      std::copy(prices.begin(),prices.end(),input.task_ns.begin()+graph.stage_offsets[s]);
      std::copy(prefetch.begin(),prefetch.end(),input.prefetch_ns.begin()+graph.stage_offsets[s]);
      continue;
    }
    auto found=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),[&](auto const& x) {
      return x.stage==projected.logical_stage && (!stage.IsCollective() || x.op.kind==analysis::OperatorKind::kMatmul);
    });
    if (found==model.task_semantics.end()) throw std::invalid_argument("stage lacks derived semantic task costs");
    auto task=DeriveModelTaskInput(model,*found,*semantic_graph,stage.IsCollective() ? &g : nullptr);
    auto traits=ModelTaskTraits(model,projected.logical_stage,g);
    int chunks=stage.IsCollective() ? cost.Chunks(model.gemms.at(stage.gemm),g) : 1;
    std::vector<analysis::ParamBinding> coordinates(counts[s]);
    if (task.scalar_access) {
      for (int t=0;t<counts[s];++t) coordinates[t].Bind("q",t);
    } else {
      auto ownership=ProjectTaskOwnership(*found,task.task,stage,threads).BindParams(theta);
      auto names=ownership.RangeDimNames();
      for (auto const& [physical,logical]:ownership.Points()) {
        if (physical.size()!=1 || physical[0]<0 || physical[0]>=counts[s])
          throw std::invalid_argument("task cost ownership disagrees with projected count");
        for (std::size_t i=0;i<names.size();++i) coordinates[physical[0]].Bind(names[i],logical[i]);
      }
    }
    std::vector<double> prefetch;PrefetchPricing pricing{options.prefetch_page_bytes,&prefetch};
    auto prices=PriceTaskInstances(cost,task,traits,{options.residency},model,chunks,coordinates,options.residency,&pricing);
    std::copy(prices.begin(),prices.end(),input.task_ns.begin()+graph.stage_offsets[s]);
    std::copy(prefetch.begin(),prefetch.end(),input.prefetch_ns.begin()+graph.stage_offsets[s]);
  }
  if(options.task_price_cache && !cached_prices) {
    auto entry=input.task_ns;
    entry.insert(entry.end(),input.prefetch_ns.begin(),input.prefetch_ns.end());
    options.task_price_cache->prices.emplace(std::move(price_key),std::move(entry));
  }
  if (bounds) {
    std::string error;
    if (!solver::PrepareRelationBounds(graph.stage_offsets,counts,input.task_ns,
            projection.dependencies.ToString(),bounds,&error))
      throw std::invalid_argument(error);
  }
  return {std::move(runtime),std::move(model),std::move(result.geometry),result.grid,threads,
          std::move(projection),std::move(counts),std::move(graph),std::move(input.task_ns),
          std::move(input.prefetch_ns)};
}

inline PlacementSolveResult SolveAndWritePlacement(mlir::ModuleOp module,
    PlacementSolveOptions const& options) {
  using namespace solver;
  auto prepared=PreparePlacementProblem(module,options);
  auto const& runtime=prepared.runtime;auto const& model=prepared.model;
  auto const& projection=prepared.projection;auto const& counts=prepared.counts;
  auto& graph=prepared.graph;
  PlacementSolveResult result;result.grid=prepared.grid;result.geometry=prepared.geometry;
  SimulatorInput input;input.graph=&graph;input.task_ns=std::move(prepared.task_ns);
  input.prefetch_ns=std::move(prepared.prefetch_ns);
  auto node=[&](long stage,long task) {
    if (stage<0 || stage>=long(counts.size()) || task<0 || task>=counts[stage])
      throw std::invalid_argument("projected relation outside task domain");
    return graph.stage_offsets[stage]+task;
  };
  PlanRequest request;request.graph=&graph;request.grid=result.grid;request.counts=counts;
  request.physical_worker.resize(result.grid);std::iota(request.physical_worker.begin(),request.physical_worker.end(),0);
  auto edges=runtime.dependencies;
  std::stable_sort(edges.begin(),edges.end(),[](auto const& a,auto const& b){return a.consumer<b.consumer;});
  for (auto const& entry:BuildVariantStageSchedule(edges,model.stages.size()).schedule)
    for (std::size_t s=0;s<projection.stages.size();++s)
      if (projection.stages[s].logical_stage==int(entry.stage)) request.stage_order.push_back(std::uint32_t(s));
  SimulatorOptions sim;sim.observed_task_times=true;sim.flat_hop=true;
  auto const& rates=options.target.EventCalibrationFor(model.dtype==ScalarType::kBF16 ? "bf16" : "f32");
  sim.publication_ns=rates.task_publication.ns.value_or(0.0);
  sim.consumer_wait_ns=rates.task_wait.ns.value_or(0.0);
  module->setAttr("tilemega.event_cost_calibrated",mlir::BoolAttr::get(module.getContext(),
      rates.task_publication.ns.has_value() && rates.task_wait.ns.has_value()));
  // Group readiness depends on the selected queue: singleton polls can be
  // elided only after ownership is known. Validate every inner candidate.
  auto price_events=[&](MaterializedPlan const& plan,codegen::RuntimeTaskGraph& grouped,
                        SimulatorInput& priced,std::vector<int>& changed_rows) {
    std::vector<unsigned char> changed(input.task_ns.size(),0);
    std::vector<std::set<std::pair<int,int>>> desired(input.task_ns.size());
    std::set<int> publishing;
    priced.publication_required.assign(input.task_ns.size(),0);
    priced.consumer_wait_required.assign(input.task_ns.size(),0);
    analysis::VisitFiniteRelation(analysis::SharedIslContext(),
        projection.requested_events.ToString(),6,[&](long const* e) {
      int cn=node(e[0],e[1]),ps=e[3],kind=e[4],group=e[5];
      if (ps<0 || ps>=int(counts.size()) || kind<0 || kind>2)
        throw std::invalid_argument("invalid projected event");
      if (kind==2 && plan.owner.at(ps).at(group)==plan.owner.at(e[0]).at(e[1])) return;
      publishing.insert(ps);
      if (!desired[cn].insert({ps,kind==0 ? -1 : group}).second) return;
      int const ek=solver::ProducerKappa(projection.options,ps);
      int begin=kind==0 ? 0 : group*ek;
      int end=kind==0 ? counts[ps] : std::min(counts[ps],begin+ek);
      for (int pt=begin;pt<end;++pt) {
        int producer=node(ps,pt);auto const& semantic=graph.successors[producer];
        bool contiguous=!semantic.empty() && semantic.back()-semantic.front()+1==int(semantic.size());
        bool present=contiguous ? cn>=semantic.front() && cn<=semantic.back()
                                : std::binary_search(semantic.begin(),semantic.end(),cn);
        if(present)continue;
        if(!changed[producer]) {changed[producer]=1;changed_rows.push_back(producer);}
        grouped.successors[producer].push_back(cn);
      }
    });
    for (int producer:changed_rows) {
      auto& row=grouped.successors[producer];
      std::sort(row.begin(),row.end());row.erase(std::unique(row.begin(),row.end()),row.end());
    }
    for (int ps:publishing)
      std::fill(priced.publication_required.begin()+graph.stage_offsets[ps],
                priced.publication_required.begin()+graph.stage_offsets[ps+1],1);
    for (auto const& queue:plan.queue) {
      std::set<std::pair<int,int>> seen;
      for (auto task:queue) for (auto event:desired[node(task.stage,task.logical)])
        if (seen.insert(event).second) priced.consumer_wait_required[node(task.stage,task.logical)]=1;
    }
  };
  result.kappa=options.kappa;
  result.candidates=SolvePlacementCatalog(input,request,sim,options.hop,price_events);
  WriteSolvedPlacement(module,result.candidates.front(),options);
  return result;
}


// A finite integer theta interval is solved exhaustively. Materialized winners
// are retained at every point; this is a bounded table, not an extrapolated
// template or a claim of portability to an unmeasured resident grid.
inline void SolveAndWritePlacementInterval(mlir::ModuleOp module,
    PlacementSolveOptions const& options,int begin,int end,std::ostream* evidence=nullptr) {
  if(begin<1 || end<begin || options.dims.past<0)
    throw std::invalid_argument("invalid placement theta interval");
  mlir::OpBuilder b(module.getContext());std::vector<mlir::Attribute> entries;
  auto seed=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(module->clone()));
  for(int seq=begin;seq<=end;++seq) {
    auto point=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>((*seed)->clone()));
    auto bound=options;bound.dims={seq,options.dims.past,seq+options.dims.past};
    auto solved=SolveAndWritePlacement(*point,bound);
    auto const& winner=solved.candidates.front();
    if(!winner.error.empty())throw std::runtime_error("no legal interval placement at seq="+std::to_string(seq));
    auto materialized=winner;materialized.mode=PlacementMode::kEft;materialized.params.clear();
    WriteSolvedPlacement(*point,materialized,bound);
    auto table=(*point)->getAttrOfType<mlir::DictionaryAttr>(kPlacementTableAttr);
    mlir::NamedAttrList fields(table);
    fields.set("selected_family",b.getStringAttr(winner.name));
    fields.set("floor_ns",b.getF64FloatAttr(winner.bounds.lower_bound_ns));
    entries.push_back(fields.getDictionary(module.getContext()));
    if(evidence)for(auto const& choice:solved.candidates)
      *evidence<<seq<<'\t'<<choice.name<<'\t'<<choice.bounds.lower_bound_ns<<'\t'
          <<choice.predicted_ns<<'\t'<<choice.error<<'\n';
    if(seq==begin) {
      module->setAttrs((*point)->getAttrs());
      for(auto placement:module.getOps<PlacementOp>()) {
        placement->setAttr("mode",b.getStringAttr("eft"));
        placement->setAttr("params",b.getDenseI64ArrayAttr({}));
        placement->setAttr("window",b.getI64IntegerAttr(1));
        placement->setAttr("policy",b.getStringAttr(kPlacementPolicyAot));
        placement->setAttr("resident_only",b.getBoolAttr(true));
        placement->removeAttr("params_map");placement->removeAttr("mapping_mode");
        auto map=analysis::CouplingRelation::FromIslText("[S,past] -> { [] -> ["+
            std::to_string(solved.grid)+"] : "+std::to_string(begin)+"<=S<="+
            std::to_string(end)+" and past="+std::to_string(options.dims.past)+" }");
        placement->setAttr("grid_map",CouplingMapAttr::get(module.getContext(),map));
        placement->setAttr("resident_limit_map",CouplingMapAttr::get(module.getContext(),map));
      }
    }
  }
  mlir::NamedAttrList table(llvm::cast<mlir::DictionaryAttr>(entries.front()));
  table.set("interval",b.getArrayAttr(entries));
  module->setAttr(kPlacementTableAttr,table.getDictionary(module.getContext()));
  module->removeAttr("tmexec.solved_seq");
  module->setAttr("tmexec.solved_seq_begin",b.getI64IntegerAttr(begin));
  module->setAttr("tmexec.solved_seq_end",b.getI64IntegerAttr(end));
  module->setAttr("tmexec.solved_placement",b.getStringAttr("finite_theta_interval"));
  if(mlir::failed(mlir::verify(module)))throw std::invalid_argument("interval CG verification failed");
}

struct PlacementSolvePass : mlir::PassWrapper<PlacementSolvePass,mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlacementSolvePass)
  PlacementSolvePass()=default;
  PlacementSolvePass(PlacementSolvePass const& other):PassWrapper(other) {}
  mlir::Pass::Option<std::string> target{*this,"target",llvm::cl::desc("Calibrated TargetSpec JSON")};
  mlir::Pass::Option<int> seq{*this,"seq",llvm::cl::init(4)};
  mlir::Pass::Option<int> seq_end{*this,"seq-end",llvm::cl::init(0)};
  mlir::Pass::Option<int> past{*this,"past",llvm::cl::init(3)};
  mlir::Pass::Option<int> kappa{*this,"kappa",llvm::cl::init(1)};
  llvm::StringRef getArgument() const final {return "tilemega-solve-placement";}
  llvm::StringRef getDescription() const final {return "Solve the placement catalog and write the selected Plan into CG";}
  void runOnOperation() override {
    try {
      PlacementSolveOptions options;options.target=TargetSpec::FromJson(target);
      options.dims={seq,past,seq+past};options.kappa=kappa;
      if(seq_end)SolveAndWritePlacementInterval(getOperation(),options,seq,seq_end);
      else SolveAndWritePlacement(getOperation(),options);
    } catch (std::exception const& e) {getOperation().emitError(e.what());signalPassFailure();}
  }
};
inline void RegisterPlacementSolvePass() {mlir::PassRegistration<PlacementSolvePass>();}
} // namespace tilemega::dialect
