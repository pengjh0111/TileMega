// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MoeStructureSearch.h>
#include <tilemega/Frontend/MoeRegionPlan.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Builders.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>

namespace tilemega::solver {
std::string MoeBindingChoice::Key() const {
  return grouped?"group;bm="+std::to_string(block_rows):"slot;bm=1";
}
frontend::ImportedSemantics RebuildMoeStructure(frontend::ImportedSemantics const& initial,
    MoeBindingChoice const& choice) {
  using namespace frontend;
  if(!initial.plan.dm || (!initial.plan.serving && !initial.plan.forward_token_axis) ||
      (choice.grouped?(choice.block_rows!=16 && choice.block_rows!=32 &&
          choice.block_rows!=64 && choice.block_rows!=128):choice.block_rows!=1))
    throw std::invalid_argument("invalid MoE binding structure");
  auto result=initial;
  auto const& bridge=initial.bridge;
  if(initial.plan.forward) {
    MoeRegionOptions options;options.tokens=initial.plan.serving_seq;
    options.grouped=choice.grouped;options.block_rows=choice.block_rows;
    for(auto const& stage:initial.plan.stages)if(stage.kind==PlanTaskKind::kMoECombine) {
      options.combine_token_tile=stage.group;options.combine_channel_tile=stage.width;
    }
    result.plan=BuildMoeRegion(bridge.nodes,bridge.inputs,bridge.outputs,options);
  } else {
    ServingOptions options;options.seq=initial.plan.serving_seq;
    options.capacity=initial.plan.serving_capacity;
    options.phase=options.seq==1?ServingOptions::Phase::kDecode:ServingOptions::Phase::kPrefill;
    options.deferred_norm=std::any_of(initial.plan.gemms.begin(),initial.plan.gemms.end(),
        [](auto const& g){return g.norm_ss!=codegen::kDmNoIndex;});
    options.moe_grouped=choice.grouped;options.moe_block_rows=choice.block_rows;
    bool binding=false;
    for(auto const& stage:initial.plan.stages) {
      if(stage.kind==PlanTaskKind::kFusedAttention) {
        options.kv_block=stage.attention_kv_block;options.query_rows=stage.attention_query_rows;
      }
      if(stage.moe.top_k && !binding) {
        if(stage.moe.row_capacity%stage.moe.top_k ||
            (stage.moe.row_capacity/stage.moe.top_k)%options.seq)
          throw std::invalid_argument("MoE binding structure has inconsistent token capacity");
        options.moe_batch=stage.moe.row_capacity/stage.moe.top_k/options.seq;binding=true;
      }
    }
    if(!binding)throw std::invalid_argument("decoder has no MoE binding structure");
    for(auto const& gemm:initial.plan.gemms)
      if(gemm.epilogue==PlanGemm::Epilogue::kArgmaxPartial)options.argmax_tile_n=gemm.partial_tile_n;
    result.plan=BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  }
  if(result.plan.gemms.size()!=initial.plan.gemms.size() ||
      result.plan.serving_seq!=initial.plan.serving_seq || result.plan.forward!=initial.plan.forward ||
      result.plan.serving_capacity!=initial.plan.serving_capacity)
    throw std::invalid_argument("MoE binding structure changed GEMM identities or phase");
  for(unsigned i=0;i<result.plan.gemms.size();++i)
    if(result.plan.gemms[i].n!=initial.plan.gemms[i].n || result.plan.gemms[i].k!=initial.plan.gemms[i].k)
      throw std::invalid_argument("MoE binding structure changed contraction geometry");
  result.lifted=LiftSemantics(result.plan,result.lift_options);return result;
}
MoeStructureSearchResult SearchMoeStructures(frontend::ImportedSemantics const& initial,
    std::vector<MoeBindingChoice> const& choices,mlir::MLIRContext& context,
    SkeletonSearchOptions const& search,std::ostream& evidence) {
  if(choices.empty() || !search.evaluation_cases.empty() || search.top_m<1)
    throw std::invalid_argument("invalid MoE structural search request");
  auto start=std::chrono::steady_clock::now();MoeStructureSearchResult result;
  std::set<std::string> seen;
  for(auto const& choice:choices) {
    if(!seen.insert(choice.Key()).second)throw std::invalid_argument("duplicate MoE binding choice");
    auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now()-start).count();
    if(search.search_budget_ms>0 && elapsed>=search.search_budget_ms) {result.budget_exhausted=true;break;}
    MoeStructureEvaluation record;record.choice=choice;
    try {
      auto rebuilt=RebuildMoeStructure(initial,choice);
      auto inner=search;inner.search_only=true;
      inner.artifact_prefix+=".binding"+std::to_string(result.evaluated.size());
      if(search.search_budget_ms>0)inner.search_budget_ms=std::max<long long>(1,
          (search.search_budget_ms-elapsed)/(choices.size()-result.evaluated.size()));
      auto solved=SolveSkeletonImported(rebuilt,context,inner,nullptr,evidence);
      for(auto& candidate:solved.evaluated)
        if(candidate.error.empty() && std::isfinite(candidate.score))record.top.push_back(std::move(candidate));
      std::stable_sort(record.top.begin(),record.top.end(),[](auto const& a,auto const& b){return a.score<b.score;});
      if(record.top.size()>std::size_t(search.top_m))record.top.resize(search.top_m);
      if(record.top.empty())throw std::runtime_error("MoE binding has no admitted geometry");
      record.score=record.top.front().score;
    }catch(std::exception const& error) {record.error=error.what();}
    evidence<<"MOE_STRUCTURE\t"<<choice.Key()<<'\t'<<record.score<<'\t'<<record.error<<'\n';evidence.flush();
    if(result.evaluated.empty() || record.score<result.evaluated[result.winner].score)
      result.winner=result.evaluated.size();
    result.evaluated.push_back(std::move(record));
  }
  llvm::json::Array entries;
  for(auto const& record:result.evaluated) {
    llvm::json::Array top;
    for(auto const& candidate:record.top)top.push_back(llvm::json::Object{
        {"key",candidate.key},{"flow_ns",candidate.score}});
    entries.push_back(llvm::json::Object{{"key",record.choice.Key()},
        {"score_ns",std::isfinite(record.score)?llvm::json::Value(record.score):llvm::json::Value(nullptr)},
        {"error",record.error},{"top",std::move(top)}});
  }
  std::ofstream file(search.artifact_prefix+".moe_structures.json");
  file<<llvm::formatv("{0:2}",llvm::json::Value(llvm::json::Object{
      {"scope","predicted flow; no latency measurement or optimality proof"},
      {"winner",int(result.winner)},{"budget_exhausted",result.budget_exhausted},
      {"evaluated",std::move(entries)}})).str()<<'\n';file.close();
  if(!file)throw std::runtime_error("cannot write MoE structural evidence");
  if(result.evaluated.empty() || !std::isfinite(result.evaluated[result.winner].score))
    throw std::runtime_error("no admitted MoE binding structure");
  return result;
}
SkeletonSearchResult SolveMoeStructures(frontend::ImportedSemantics const& initial,
    std::vector<MoeBindingChoice> const& choices,mlir::MLIRContext& context,
    SkeletonSearchOptions const& search,frontend::ImportSummary* summary,std::ostream& evidence) {
  auto structures=SearchMoeStructures(initial,choices,context,search,evidence);
  auto const& selected=structures.evaluated.at(structures.winner);
  auto imported=RebuildMoeStructure(initial,selected.choice);auto final=search;
  final.evaluation_cases.clear();
  for(auto const& candidate:selected.top) {
    SkeletonEvaluationCase item{candidate.config,candidate.kappa,candidate.residency};
    item.page_bytes=candidate.page_bytes;item.lookahead_bytes=candidate.lookahead_bytes;
    item.handoff_mask=candidate.handoff_mask;final.evaluation_cases.push_back(std::move(item));
  }
  auto result=SolveSkeletonImported(imported,context,final,summary,evidence);
  if(result.compiled.module) {
    mlir::Builder builder(&context);auto mark=[&](mlir::ModuleOp module) {
      module->setAttr("tilemega.moe_structure",builder.getStringAttr(selected.choice.Key()));
    };
    mark(*result.compiled.module);for(auto& entry:result.compiled.shortlist)mark(*entry.module);
  }
  return result;
}
}
