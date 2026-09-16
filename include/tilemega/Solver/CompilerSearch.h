// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/CandidateGenerator.h>
#include <tilemega/Solver/JointSearch.h>
#include <ostream>

namespace tilemega::solver {
struct CompilerSearchOptions {
  dialect::PlacementSolveOptions placement;
  std::size_t capacity=8;
  std::function<int(mlir::ModuleOp,int)> query_residency;
};
struct CompilerSearchResult {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  JointSearchStats stats;
  std::vector<JointEvaluation> ranking;
};
/// The finite budget is disclosed as deferred work, never an optimality proof.
/// Reimport each geometry so ownership and CG relations change together.
inline CompilerSearchResult SolveExport(std::string const& path,
    mlir::MLIRContext& context,CompilerSearchOptions const& options,
    frontend::ImportSummary* summary,std::ostream& evidence) {
  auto seed=frontend::TorchExportImporter{}.Import(path,context);
  auto model=ModelDescription::FromCouplingGraph(*seed,options.placement.dims,"compile-search");
  CandidateGenerator generator(options.placement.target,model.dtype,{256,64,5});
  CostModel cost(options.placement.target,model.dtype);
  std::vector<JointCandidate> candidates;
  for (auto const& backend:generator.Enumerate()) {
    auto const& t=backend.traits();
    for (int split:{1,2,4,8,16,32}) {
      GemmConfig g{t.tile_m,t.tile_n,t.tile_k,t.stages,split};
      // This is a ranking heuristic, not a pruning bound. Exact task-domain
      // bounds are evaluated after importing the candidate's own CG below.
      double priority=cost.Evaluate(model,g,{1}).total_ns;
      for (int kappa:{1,2,4}) {
        JointCandidate candidate;candidate.config=g;candidate.kappa=kappa;
        candidate.ctas_per_sm=1;candidate.priority_ns=priority;
        candidate.key=std::to_string(g.tile_m)+"x"+std::to_string(g.tile_n)+"x"+
            std::to_string(g.tile_k)+"s"+std::to_string(g.stages)+"split"+
            std::to_string(split)+"kappa"+std::to_string(kappa);
        candidates.push_back(std::move(candidate));
      }
    }
  }
  CompilerSearchResult result;
  double best=std::numeric_limits<double>::infinity(),best_sim=best;
  evidence << "candidate\tplacement\tstatus\tfloor_ns\tcp_ns\tqueue_lb_ns\tpredicted_ns\tgrid\tkappa\n";
  result.ranking=SearchL2Configurations(std::move(candidates),options.capacity,
      [&](JointCandidate const& candidate) {
    std::vector<JointEvaluation> evaluations;
    try {
      frontend::ImportOptions import;
      auto const& g=candidate.config;
      import.gemms.assign(model.gemms.size(),{g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
      import.rope_tile_per_block=import.kv_tile_per_block=true;
      import.activation_tile_per_block=import.combiner_tile_per_block=true;
      frontend::ImportSummary candidate_summary;
      auto module=frontend::TorchExportImporter{}.Import(path,context,&candidate_summary,import);
      int limit=options.query_residency ? options.query_residency(*module,candidate.kappa) : 1;
      if (limit<1) throw std::invalid_argument("compiled kernel has no resident CTA");
      for (int residency=1;residency<=limit;++residency) {
        auto placed=mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(module->clone()));
        auto placement=options.placement;placement.kappa=candidate.kappa;
        placement.residency=residency;placement.verified_resident_limit=limit;
        auto solved=dialect::SolveAndWritePlacement(*placed,placement);
        for (auto const& plan:solved.candidates) {
          JointEvaluation e;e.candidate=candidate;e.candidate.ctas_per_sm=residency;
          e.candidate.key+="r"+std::to_string(residency);
          e.placement=plan.name;e.status=plan.error.empty() ? "ok" : plan.error;
          e.floor_ns=plan.bounds.lower_bound_ns;e.makespan_ns=plan.predicted_ns;e.simulated=plan.error.empty();
          evaluations.push_back(e);
          evidence << e.candidate.key << '\t' << plan.name << '\t' << e.status << '\t'
              << e.floor_ns << '\t' << plan.bounds.critical_path_ns << '\t'
              << plan.bounds.queue_lb_ns << '\t' << e.makespan_ns << '\t'
              << solved.grid << '\t' << candidate.kappa << '\n';
        }
        auto const& chosen=solved.candidates.front();
        if (std::tie(chosen.bounds.lower_bound_ns,chosen.predicted_ns)<std::tie(best,best_sim)) {
          best=chosen.bounds.lower_bound_ns;best_sim=chosen.predicted_ns;
          result.module=std::move(placed);if (summary) *summary=candidate_summary;
        }
      }

    } catch (std::exception const& e) {
      JointEvaluation failed;failed.status=e.what();evaluations.push_back(failed);
      evidence << candidate.key << "\t-\t" << e.what() << "\t0\t0\t0\t0\t0\t" << candidate.kappa << '\n';
    }
    evidence.flush();return evaluations;
  },&result.stats);
  if (!result.module) throw std::runtime_error("no legal solver configuration; inspect search evidence");
  mlir::OpBuilder b(&context);
  result.module->getOperation()->setAttr("tilemega.search_deferred",b.getI64IntegerAttr(result.stats.capacity_deferred));
  result.module->getOperation()->setAttr("tilemega.search_scope",b.getStringAttr(
      options.query_residency ? "uniform geometry; six placements; kappa 1,2,4; all compiled resident levels" :
      "uniform geometry; six placements; kappa 1,2,4; residency 1 pending compiled resource evidence"));
  return result;
}
} // namespace tilemega::solver
