// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/DnnStructureSearch.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Builders.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>

namespace tilemega::solver {
std::string DnnStructureChoice::Key() const {
  std::ostringstream out;out<<"c="<<small_channels<<";reuse="<<reuse<<";fuse=";
  for(auto id:fused_gemms)out<<id<<',';
  out<<";ln=";for(auto const& [norm,id]:deferred_edges)
    out<<norm.size()<<':'<<norm<<':'<<id<<';';
  if(reduction_mask>=0)out<<";la="<<reduction_mask;
  return out.str();
}
frontend::ImportedSemantics RebuildDnnStructure(frontend::ImportedSemantics const& initial,
    frontend::DnnPlanOptions options,DnnStructureChoice const& choice) {
  options.dwpw_fuse_gemms=choice.fused_gemms;
  options.deferred_layernorm_edges=choice.deferred_edges;
  options.memory_reuse=choice.reuse;options.small_input_channels=choice.small_channels;
  auto result=initial;
  result.plan=frontend::BuildDnnModelPlan(initial.bridge.nodes,initial.bridge.inputs,
      initial.bridge.outputs,options);
  if(result.plan.gemms.size()!=initial.plan.gemms.size() ||
      result.plan.serving_seq!=initial.plan.serving_seq)
    throw std::invalid_argument("DNN structure changed GEMM identities or workload");
  result.plan.dm_reduction_mask=choice.reduction_mask;
  result.lifted=frontend::LiftSemantics(result.plan,result.lift_options);
  return result;
}
DnnStructureSearchResult SearchDnnStructures(frontend::ImportedSemantics const& initial,
    frontend::DnnPlanOptions const& options,std::vector<std::string> const& reuse_choices,
    mlir::MLIRContext& context,SkeletonSearchOptions const& search,std::ostream& evidence) {
  if(!initial.plan.dm || !initial.plan.forward || initial.plan.forward_token_axis ||
      !search.evaluation_cases.empty() || search.top_m<1 || search.passes<1 || reuse_choices.empty())
    throw std::invalid_argument("invalid DNN structural search request");
  for(auto const& policy:reuse_choices)
    if(policy!="none" && policy!="greedy" && policy!="l2")
      throw std::invalid_argument("invalid DNN structural reuse policy");
  if(std::find(reuse_choices.begin(),reuse_choices.end(),options.memory_reuse)==reuse_choices.end())
    throw std::invalid_argument("DNN initial reuse policy is outside its candidate domain");
  if(options.batch!=unsigned(search.common.placement.dims.batch))
    throw std::invalid_argument("DNN structural batch differs from search binding");
  auto const start=std::chrono::steady_clock::now();
  auto remaining=[&] {
    if(search.search_budget_ms<=0)return 0;
    auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now()-start).count();
    return int(std::max<long long>(0,search.search_budget_ms-elapsed));
  };
  DnnStructureChoice enabled;enabled.reuse=options.memory_reuse;
  enabled.small_channels=options.small_input_channels;
  enabled.reduction_mask=initial.plan.dm_reduction_mask;
  for(auto const& stage:initial.plan.stages)
    if(stage.kind==frontend::PlanTaskKind::kDwPwFused)enabled.fused_gemms.insert(stage.gemm);
  enabled.deferred_edges.insert(initial.plan.deferred_layernorm_edges.begin(),
      initial.plan.deferred_layernorm_edges.end());
  std::vector<unsigned> channels{enabled.small_channels};
  for(auto const& buffer:initial.plan.buffers)
    if(buffer.layout.kind==codegen::DmLayout::kNHWC && buffer.layout.rank==4 &&
        buffer.layout.logical[3]<=4) {channels={4,8};break;}
  auto coordinates=enabled.fused_gemms.size()+enabled.deferred_edges.size()+
      reuse_choices.size()+channels.size()+(enabled.reduction_mask>0);
  auto slice=search.search_budget_ms>0?std::max<std::size_t>(1,
      search.search_budget_ms/(2+2*search.passes*coordinates)):0;
  DnnStructureSearchResult result;std::map<std::string,std::size_t> memo;
  auto admitted=[&](std::size_t index){return std::isfinite(result.evaluated.at(index).score);};
  auto better=[&](std::size_t a,std::size_t b) {
    return result.evaluated.at(a).score<result.evaluated.at(b).score;
  };
  auto evaluate=[&](DnnStructureChoice choice) {
    auto key=choice.Key();auto found=memo.find(key);if(found!=memo.end())return found->second;
    auto index=result.evaluated.size();memo.emplace(key,index);
    DnnStructureEvaluation record;record.choice=std::move(choice);
    try {
      auto rebuilt=RebuildDnnStructure(initial,options,record.choice);
      auto inner=search;inner.search_only=true;
      inner.artifact_prefix+=".structure"+std::to_string(index);
      if(search.search_budget_ms>0)inner.search_budget_ms=std::max(1,
          std::min(remaining(),int(slice)));
      auto priced=SolveSkeletonImported(rebuilt,context,inner,nullptr,evidence);
      record.classes=std::move(priced.classes);
      for(auto& candidate:priced.evaluated)
        if(candidate.error.empty() && std::isfinite(candidate.score))record.top.push_back(std::move(candidate));
      std::stable_sort(record.top.begin(),record.top.end(),[](auto const& a,auto const& b){return a.score<b.score;});
      if(record.top.size()>std::size_t(search.top_m))record.top.resize(search.top_m);
      if(record.top.empty())throw std::runtime_error("DNN structure has no admitted geometry");
      record.score=record.top.front().score;
    } catch(std::exception const& error) {record.error=error.what();}
    evidence<<"DNN_STRUCTURE\t"<<index<<'\t'<<key<<'\t'<<record.score<<'\t'<<record.error<<'\n';
    evidence.flush();result.evaluated.push_back(std::move(record));
    if(index==0 || better(index,result.winner))result.winner=index;
    return index;
  };
  auto expired=[&] {
    if(search.search_budget_ms>0 && !remaining())result.budget_exhausted=true;
    return result.budget_exhausted;
  };
  auto off=enabled;off.fused_gemms.clear();off.deferred_edges.clear();
  if(off.reduction_mask>=0)off.reduction_mask=0;
  for(auto const& seed:{enabled,off}) {
    if(!result.evaluated.empty() && expired())break;
    auto incumbent=evaluate(seed);
    if(!admitted(incumbent))continue;
    for(int pass=0;pass<search.passes && !expired();++pass) {
      bool moved=false;++result.rounds;
      auto attempt=[&](DnnStructureChoice const& choice) {
        if(expired())return;
        auto candidate=evaluate(choice);
        if(better(candidate,incumbent)){incumbent=candidate;moved=true;}
      };
      for(auto id:enabled.fused_gemms) {
        auto choice=result.evaluated[incumbent].choice;
        if(!choice.fused_gemms.erase(id))choice.fused_gemms.insert(id);
        attempt(choice);
      }
      for(auto id:enabled.deferred_edges) {
        auto choice=result.evaluated[incumbent].choice;
        if(!choice.deferred_edges.erase(id))choice.deferred_edges.insert(id);
        attempt(choice);
      }
      for(auto const& policy:reuse_choices) {
        auto choice=result.evaluated[incumbent].choice;choice.reuse=policy;attempt(choice);
      }
      for(auto channel:channels) {
        auto choice=result.evaluated[incumbent].choice;choice.small_channels=channel;attempt(choice);
      }
      if(enabled.reduction_mask>0) {
        auto choice=result.evaluated[incumbent].choice;
        choice.reduction_mask^=enabled.reduction_mask;attempt(choice);
      }
      if(!moved)break;
    }
  }
  llvm::json::Array records;
  for(auto const& record:result.evaluated) {
    llvm::json::Array top;
    for(auto const& candidate:record.top)top.push_back(llvm::json::Object{
        {"key",candidate.key},{"flow_ns",candidate.score}});
    records.push_back(llvm::json::Object{{"key",record.choice.Key()},
        {"score_ns",std::isfinite(record.score)?llvm::json::Value(record.score):llvm::json::Value(nullptr)},
        {"error",record.error},{"top",std::move(top)}});
  }
  std::ofstream file(search.artifact_prefix+".structures.json");
  file<<llvm::formatv("{0:2}",llvm::json::Value(llvm::json::Object{
      {"scope","predicted flow; no latency measurement or optimality proof"},
      {"winner",int(result.winner)},{"budget_exhausted",result.budget_exhausted},
      {"rounds",result.rounds},{"evaluated",std::move(records)}})).str()<<'\n';
  file.close();
  if(!file)throw std::runtime_error("cannot write DNN structure evidence");
  if(result.evaluated.empty() || !admitted(result.winner)) {
    std::string errors;for(auto const& item:result.evaluated)
      errors+="\n"+item.choice.Key()+": "+item.error;
    throw std::runtime_error("no admitted DNN structure"+errors);
  }
  return result;
}
SkeletonSearchResult SolveDnnStructures(frontend::ImportedSemantics const& initial,
    frontend::DnnPlanOptions const& options,std::vector<std::string> const& reuse_choices,
    mlir::MLIRContext& context,SkeletonSearchOptions const& search,
    frontend::ImportSummary* summary,std::ostream& evidence) {
  auto structures=SearchDnnStructures(initial,options,reuse_choices,context,search,evidence);
  auto const& selected=structures.evaluated.at(structures.winner);
  auto imported=RebuildDnnStructure(initial,options,selected.choice);
  auto final=search;final.evaluation_cases.clear();
  for(auto const& candidate:selected.top) {
    SkeletonEvaluationCase item{candidate.config,candidate.kappa,candidate.residency};
    item.page_bytes=candidate.page_bytes;item.lookahead_bytes=candidate.lookahead_bytes;
    item.handoff_mask=candidate.handoff_mask;
    final.evaluation_cases.push_back(std::move(item));
  }
  auto result=SolveSkeletonImported(imported,context,final,summary,evidence);
  if(result.compiled.module) {
    mlir::Builder builder(&context);
    auto mark=[&](mlir::ModuleOp module) {module->setAttr("tilemega.dnn_structure",
        builder.getStringAttr(selected.choice.Key()));};
    mark(*result.compiled.module);
    for(auto& entry:result.compiled.shortlist)mark(*entry.module);
  }
  return result;
}
}
