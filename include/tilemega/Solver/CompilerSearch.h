// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/CandidateGenerator.h>
#include <tilemega/Solver/JointSearch.h>
#include <ostream>

namespace tilemega::solver {
struct CompilerSearchOptions {
  dialect::PlacementSolveOptions placement;
  std::size_t capacity=8;
  // Optional externally evidenced backend domain for a disclosed reduced
  // search. Split and kappa remain decisions of the solver.
  std::vector<GemmConfig> geometry_domain;
  std::vector<std::pair<GemmConfig,std::string>> numerical_rejections;
  /// Refine the winning uniform kappa into a per-producer-stage table (§6 B2).
  /// Off by default: with it off no per-stage attribute is written, so the
  /// generated text, and therefore the SASS, is what it was (H2).
  bool per_stage_kappa=false;
  /// A table pinned by the caller is solved on the winner in place of the
  /// descent. It cannot ride the outer search: the projected stage count
  /// changes with split-K, so no one table covers every candidate.
  std::vector<int> stage_kappa;
  std::function<int(mlir::ModuleOp,int)> query_residency;
  /// Keep the best plan of every evaluated candidate, not just the top three
  /// (§6 C1-c needs the measured set to cover what the search looked at).
  /// Off by default: it holds one module per candidate alive to the end.
  bool keep_evaluated=false;
};
struct CompilerSearchResult {
  struct ShortlistEntry {
    JointEvaluation evaluation;
    mlir::OwningOpRef<mlir::ModuleOp> module;
  };
  mlir::OwningOpRef<mlir::ModuleOp> module;
  JointSearchStats stats;
  std::vector<JointEvaluation> ranking;
  std::vector<JointCandidate> outer_candidates;
  std::vector<ShortlistEntry> shortlist;
  /// The configuration behind `module`, and the resident CTAs its kernel
  /// compiled to. A segmented interval reuses both to keep the launch fixed.
  JointCandidate winner;
  int winner_resident_limit=0;
  /// One entry per evaluated candidate, its own best plan, when the caller
  /// asked for them. The shortlist above is the first three of these by key.
  std::vector<ShortlistEntry> evaluated;
  /// Empty unless the per-stage refinement ran and improved on uniform kappa.
  std::vector<int> stage_kappa;
  /// Projected stages on the winner, which is the length any pinned table must
  /// have. It is not knowable before the search: split-K adds combine stages.
  std::size_t stage_count=0;
  int stage_kappa_moves=0;
  double uniform_ns=0,per_stage_ns=0;
};
/// The finite budget is disclosed as deferred work, never an optimality proof.
/// Reimport each geometry so ownership and CG relations change together.
inline CompilerSearchResult SolveExport(std::string const& path,
    mlir::MLIRContext& context,CompilerSearchOptions const& options,
    frontend::ImportSummary* summary,std::ostream& evidence) {
  // The plan is the pattern match over the exported graph and does not read
  // ImportOptions, so it is the same object for every granularity below.
  // Rebuilding it per candidate is the outer loop's dominant cost (F-229).
  auto bridge=frontend::ReadExportBridge(path);
  auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
  auto seed=frontend::TorchExportImporter{}.ImportPlan(path,plan,context);
  auto model=ModelDescription::FromCouplingGraph(*seed,options.placement.dims,"compile-search");
  CandidateGenerator generator(options.placement.target,model.dtype,{256,64,5});
  CompilerSearchResult result;
  std::vector<JointCandidate> candidates;
  for (auto const& backend:generator.Enumerate()) {
    auto const& t=backend.traits();
    if (!options.geometry_domain.empty() && std::none_of(options.geometry_domain.begin(),
        options.geometry_domain.end(),[&](auto const& g) {
          return std::tie(g.tile_m,g.tile_n,g.tile_k,g.stages)==
                 std::tie(t.tile_m,t.tile_n,t.tile_k,t.stages);
        })) continue;
    for (int split:{1,2,4,8,16,32}) {
      GemmConfig g{t.tile_m,t.tile_n,t.tile_k,t.stages,split};
      frontend::ImportOptions import;
      import.gemms.assign(model.gemms.size(),{g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
      import.rope_tile_per_block=import.kv_tile_per_block=true;
      import.activation_tile_per_block=import.combiner_tile_per_block=true;
      auto coarse_module=frontend::TorchExportImporter{}.ImportPlan(path,plan,context,nullptr,import);
      auto optimistic=options.placement;optimistic.residency=1;optimistic.verified_resident_limit=1;
      optimistic.requested_grid=0;optimistic.kappa=1;
      // Bounds only: this pass asks for work and the critical path and nothing
      // else, so the dense edge set is never built (§6 C1-b).
      RelationBounds bounds;
      auto problem=dialect::PreparePlacementProblem(*coarse_module,optimistic,&bounds);
      // Before compiling occupancy, the hardware thread limit is an upper
      // bound on any legal grid. One-CTA task service is a lower bound on
      // every residency priced by this monotone resource model.
      int max_ctas=std::max(1,options.placement.target.res.max_threads_per_sm/problem.threads);
      int grid_upper=options.placement.target.res.num_sms*max_ctas;
      double work=bounds.work_ns/grid_upper,queue=0;
      for(std::size_t stage=0;stage<problem.counts.size();++stage) {
        int begin=problem.graph.stage_offsets[stage],end=problem.graph.stage_offsets[stage+1];
        if(begin==end)continue;
        double minimum=*std::min_element(problem.task_ns.begin()+begin,problem.task_ns.begin()+end);
        // Some worker must execute this many tasks of the stage, even if
        // all other stages can be placed arbitrarily.
        queue=std::max(queue,minimum*double((end-begin+grid_upper-1)/grid_upper));
      }
      double priority=std::max({work,bounds.critical_path_ns,queue});
      for (int kappa:{1,2,4}) {
        JointCandidate candidate;candidate.config=g;candidate.kappa=kappa;
        candidate.ctas_per_sm=1;candidate.priority_ns=priority;
        candidate.work_lb_ns=work;candidate.cp_lb_ns=bounds.critical_path_ns;
        candidate.queue_lb_lb_ns=queue;
        candidate.key=std::to_string(g.tile_m)+"x"+std::to_string(g.tile_n)+"x"+
            std::to_string(g.tile_k)+"s"+std::to_string(g.stages)+"split"+
            std::to_string(split)+"kappa"+std::to_string(kappa);
        candidates.push_back(std::move(candidate));
      }
    }
  }
  auto task_prices=std::make_shared<dialect::PlacementTaskPriceCache>();
  JointCandidate winner;int winner_limit=0;
  result.outer_candidates=candidates;
  double best=std::numeric_limits<double>::infinity(),best_sim=best;
  evidence << "candidate\tplacement\tstatus\tfloor_ns\tcp_ns\tqueue_lb_ns\tpredicted_ns\tgrid\tkappa\n";
  // Rejected numerical geometries are outside the admitted domain, so their
  // kappa variants must not consume the finite evaluation budget.
  candidates.erase(std::remove_if(candidates.begin(),candidates.end(),[&](auto const& candidate) {
    for(auto const& [rejected,reason]:options.numerical_rejections) {
      auto const& g=candidate.config;
      if(std::tie(g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k)==
          std::tie(rejected.tile_m,rejected.tile_n,rejected.tile_k,rejected.stages,rejected.split_k)) {
        evidence<<candidate.key<<"\t-\tnumerical exclusion: "<<reason<<"\t0\t0\t0\t0\t0\t"<<candidate.kappa<<'\n';
        return true;
      }
    }
    return false;
  }),candidates.end());
  result.ranking=SearchL2Configurations(std::move(candidates),options.capacity,
      [&](JointCandidate const& candidate) {
    std::vector<JointEvaluation> evaluations;
    try {
      for(auto const& [rejected,reason]:options.numerical_rejections) {
        auto const& g=candidate.config;
        if(std::tie(g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k)==
           std::tie(rejected.tile_m,rejected.tile_n,rejected.tile_k,rejected.stages,rejected.split_k))
          throw std::invalid_argument("numerical exclusion: "+reason);
      }
      frontend::ImportOptions import;
      auto const& g=candidate.config;
      import.gemms.assign(model.gemms.size(),{g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
      import.rope_tile_per_block=import.kv_tile_per_block=true;
      import.activation_tile_per_block=import.combiner_tile_per_block=true;
      frontend::ImportSummary candidate_summary;
      auto module=frontend::TorchExportImporter{}.ImportPlan(path,plan,context,&candidate_summary,import);
      int limit=options.query_residency ? options.query_residency(*module,candidate.kappa) : 1;
      if (limit<1) throw std::invalid_argument("compiled kernel has no resident CTA");
      double candidate_floor=std::numeric_limits<double>::infinity(),candidate_sim=candidate_floor;
      JointEvaluation candidate_best;mlir::OwningOpRef<mlir::ModuleOp> candidate_module;
      for (int residency=1;residency<=limit;++residency) {
        auto placed=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(module->clone()));
        auto placement=options.placement;placement.kappa=candidate.kappa;placement.task_price_cache=task_prices;
        placement.residency=residency;placement.verified_resident_limit=limit;
        auto solved=dialect::SolveAndWritePlacement(*placed,placement);
        for (auto const& plan:solved.candidates) {
          JointEvaluation e;e.candidate=candidate;e.candidate.ctas_per_sm=residency;
          e.candidate.key+="r"+std::to_string(residency);
          e.placement=plan.name;e.status=plan.error.empty() ? "ok" : plan.error;
          e.floor_ns=plan.bounds.lower_bound_ns;e.makespan_ns=plan.predicted_ns;e.simulated=plan.error.empty();
          evaluations.push_back(e);
          if (e.simulated && (result.shortlist.size()<3 ||
              std::tie(e.floor_ns,e.makespan_ns)<std::tie(result.shortlist.back().evaluation.floor_ns,
                                                       result.shortlist.back().evaluation.makespan_ns))) {
            auto copy=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(placed->clone()));
            dialect::WriteSolvedPlacement(*copy,plan,placement);
            result.shortlist.push_back({e,std::move(copy)});
            std::stable_sort(result.shortlist.begin(),result.shortlist.end(),[](auto const& a,auto const& b) {
              return std::tie(a.evaluation.floor_ns,a.evaluation.makespan_ns)<
                     std::tie(b.evaluation.floor_ns,b.evaluation.makespan_ns);
            });
            if (result.shortlist.size()>3) result.shortlist.pop_back();
          }
          if (options.keep_evaluated && e.simulated &&
              std::tie(e.floor_ns,e.makespan_ns)<std::tie(candidate_floor,candidate_sim)) {
            candidate_floor=e.floor_ns;candidate_sim=e.makespan_ns;candidate_best=e;
            auto copy=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(placed->clone()));
            dialect::WriteSolvedPlacement(*copy,plan,placement);
            candidate_module=std::move(copy);
          }
          evidence << e.candidate.key << '\t' << plan.name << '\t' << e.status << '\t'
              << e.floor_ns << '\t' << plan.bounds.critical_path_ns << '\t'
              << plan.bounds.queue_lb_ns << '\t' << e.makespan_ns << '\t'
              << solved.grid << '\t' << candidate.kappa << '\n';
        }
        auto const& chosen=solved.candidates.front();
        if (std::tie(chosen.bounds.lower_bound_ns,chosen.predicted_ns)<std::tie(best,best_sim)) {
          best=chosen.bounds.lower_bound_ns;best_sim=chosen.predicted_ns;
          winner=candidate;winner.ctas_per_sm=residency;winner_limit=limit;
          result.module=std::move(placed);if (summary) *summary=candidate_summary;
        }
      }
      if (candidate_module) result.evaluated.push_back({candidate_best,std::move(candidate_module)});

    } catch (std::exception const& e) {
      JointEvaluation failed;failed.status=e.what();evaluations.push_back(failed);
      evidence << candidate.key << "\t-\t" << e.what() << "\t0\t0\t0\t0\t0\t" << candidate.kappa << '\n';
    }
    evidence.flush();return evaluations;
  },&result.stats);
  if (!result.module) throw std::runtime_error("no legal solver configuration; inspect search evidence");
  // §6 B2: the uniform kappa the outer search selected is the start of a
  // coordinate descent over producer stages. A stage with one task has one
  // group at every kappa, so the sweep skips it; every trial is a full
  // placement solve, which is what bounds the descent to two passes.
  if (options.per_stage_kappa || !options.stage_kappa.empty()) {
    frontend::ImportOptions import;
    auto const& g=winner.config;
    import.gemms.assign(model.gemms.size(),{g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
    import.rope_tile_per_block=import.kv_tile_per_block=true;
    import.activation_tile_per_block=import.combiner_tile_per_block=true;
    auto base=frontend::TorchExportImporter{}.ImportPlan(path,plan,context,nullptr,import);
    auto placement=options.placement;
    placement.kappa=winner.kappa;placement.residency=winner.ctas_per_sm;
    placement.verified_resident_limit=winner_limit;placement.task_price_cache=task_prices;
    RelationBounds shape;
    auto const counts=dialect::PreparePlacementProblem(*base,placement,&shape).counts;
    mlir::OwningOpRef<mlir::ModuleOp> refined;
    auto evaluate=[&](std::vector<int> const& table,std::string const& label,
                      bool keep)->std::pair<double,double> {
      auto trial=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(base->clone()));
      auto trial_options=placement;trial_options.stage_kappa=table;
      auto solved=dialect::SolveAndWritePlacement(*trial,trial_options);
      auto const& chosen=solved.candidates.front();
      evidence << winner.key << label << '\t' << chosen.name << '\t'
          << (chosen.error.empty() ? "ok" : chosen.error) << '\t'
          << chosen.bounds.lower_bound_ns << '\t' << chosen.bounds.critical_path_ns
          << '\t' << chosen.bounds.queue_lb_ns << '\t' << chosen.predicted_ns
          << '\t' << solved.grid << '\t' << winner.kappa << '\n';
      evidence.flush();
      if (keep) refined=std::move(trial);
      return {chosen.bounds.lower_bound_ns,chosen.predicted_ns};
    };
    result.stage_count=counts.size();
    std::vector<int> table(counts.size(),winner.kappa);
    auto incumbent=evaluate(table,"-perstage-uniform",false);
    result.uniform_ns=incumbent.second;
    if (!options.stage_kappa.empty()) {
      // The placement is re-solved under the pinned table rather than
      // relabelled: a coarser event groups later producer tasks into the wait,
      // and a slot order solved for uniform kappa could put a consumer ahead of
      // one of them on its own worker.
      result.per_stage_ns=evaluate(options.stage_kappa,"-pinned",true).second;
      result.module=std::move(refined);result.stage_kappa=options.stage_kappa;
    } else {
      for (int pass=0;pass<2;++pass) {
        bool moved=false;
        for (std::size_t stage=0;stage<table.size();++stage) {
          if (counts[stage]<2) continue;
          for (int kappa:{1,2,4}) {
            if (kappa==table[stage]) continue;
            int const previous=table[stage];table[stage]=kappa;
            auto trial=evaluate(table,"-s"+std::to_string(stage)+"k"+std::to_string(kappa),false);
            if (trial<incumbent) {incumbent=trial;moved=true;++result.stage_kappa_moves;}
            else table[stage]=previous;
          }
        }
        if (!moved) break;
      }
      result.per_stage_ns=incumbent.second;
      // A table the descent never moved off uniform is the global kappa, so it
      // is not written back: B2's answer there is that global kappa sufficed.
      if (result.stage_kappa_moves) {
        evaluate(table,"-perstage-best",true);
        // The refined module is the winner's geometry re-imported, so the
        // summary the search already reported still describes it.
        result.module=std::move(refined);result.stage_kappa=table;
      }
    }
  }
  result.winner=winner;result.winner_resident_limit=winner_limit;
  mlir::OpBuilder b(&context);
  result.module->getOperation()->setAttr("tilemega.search_deferred",b.getI64IntegerAttr(result.stats.capacity_deferred));
  result.module->getOperation()->setAttr("tilemega.search_restricted_geometry",b.getBoolAttr(!options.geometry_domain.empty()));
  result.module->getOperation()->setAttr("tilemega.search_scope",b.getStringAttr(
      options.query_residency ? "uniform geometry; six placements; kappa 1,2,4; all compiled resident levels" :
      "uniform geometry; six placements; kappa 1,2,4; residency 1 pending compiled resource evidence"));
  return result;
}
} // namespace tilemega::solver
