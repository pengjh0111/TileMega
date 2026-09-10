// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ChainDP.h>
#include <tilemega/Solver/FusionResources.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/AttentionWork.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <set>
#include <stdexcept>

#ifndef TILEMEGA_FUSION_INTERVAL_DP
#define TILEMEGA_FUSION_INTERVAL_DP 1
#endif

namespace tilemega::solver {
ChainDpSolution ChainDP::SolveFusionIntervals(ModelDescription const& model,
    FusionDpDomain const& domain,ChainDpOptions options,
    std::vector<FusionDpAlternative>* alternatives) const {
  analysis::IslReferenceAudit audit(__func__);
#if !TILEMEGA_FUSION_INTERVAL_DP
  throw std::invalid_argument("fusion interval DP disabled");
#endif
  if (model.dims.IsSymbolic() || !cost_->options().unified_task_cost ||
      cost_->options().cg_interface || !cost_->options().l2_events || !cost_->options().sync)
    throw std::invalid_argument("fusion interval DP requires concrete unified L2 prices without legacy interface terms");
  if (domain.plan.gemms.size()!=model.gemms.size() || domain.pairs.empty() ||
      domain.stage_traits.size()!=model.stages.size() ||
      domain.stage_registers.size()!=model.stages.size() || domain.static_shared_bytes<0 ||
      domain.resident_shared_floor<0 ||
      options.max_ctas_per_sm<=0 || !options.per_operator_candidates.empty())
    throw std::invalid_argument("incomplete fusion interval implementation domain");
  (void)GemmStages(model);
  int threads=cost_->dtype()==ScalarType::kBF16 ? kTensorBF16Threads : kSimtF32Threads;
  std::vector<GemmConfig> configs;
  for (auto const& g:domain.plan.gemms)
    configs.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
  for (std::size_t i=0;i<domain.stage_traits.size();++i)
    if (domain.stage_traits[i].threads!=threads || domain.stage_traits[i].smem_bytes<0 || domain.stage_registers[i]<=0)
      throw std::invalid_argument("fusion interval requires complete phase resource evidence");
  struct Pair {
    int producer,consumer;
    ModelFusionCandidate runtime;
    analysis::CouplingRelation queue_mapping;
    FusionResources resources;
    std::pair<std::string,std::string> names;
  };
  std::vector<Pair> pairs;
  std::set<std::pair<std::string,std::string>> unique;
  for (auto const& names:domain.pairs) {
    if (!unique.insert(names).second) throw std::invalid_argument("duplicate fusion interval candidate");
    (void)DeriveLogicalFusionCandidate(model,configs,names.first,names.second);
    int p=-1,c=-1;
    for (auto const& task:model.task_semantics) {
      if (task.op.name==names.first) p=task.stage;
      if (task.op.name==names.second) c=task.stage;
    }
    auto candidate=DeriveModelFusionCandidate(model,configs,p,c);
    auto ownership=[&](int stage,DerivedTaskInput const& input) {
      if (input.scalar_access)
        return analysis::CouplingRelation::FromIslText("{ [q] -> [q] }");
      auto semantic=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),
          [&](auto const& entry) { return entry.stage==stage; });
      if (semantic==model.task_semantics.end())
        throw std::invalid_argument("fusion phase has no ownership semantics");
      return ProjectTaskOwnership(*semantic,input.task,model.stages.at(stage),threads)
          .BindParams(model.MetricBindings());
    };
    auto queue_mapping=ownership(c,candidate.consumer)
        .ApplyRange(candidate.accesses.consumer_to_producer)
        .ApplyRange(ownership(p,candidate.producer).Reverse());
    std::map<std::string,int> types;
    for (auto const& [tensor,map]:candidate.accesses.intermediate_tiles)
      types.emplace(tensor,cost_->dtype()==ScalarType::kBF16 ? 2 : 4);
    long allocated=0;
    if (model.stages[p].kind==StageKind::kGemm) {
      auto const& g=configs.at(model.stages[p].gemm);
      allocated=static_cast<long>(g.tile_m)*g.tile_n*(cost_->dtype()==ScalarType::kBF16 ? 2 : 4);
    }
    auto resources=DeriveFusionResources(candidate.accesses,model.MetricBindings(),types,
        domain.stage_traits.at(p),domain.stage_registers.at(p),
        domain.stage_traits.at(c),domain.stage_registers.at(c),allocated);
    pairs.push_back({p,c,std::move(candidate),std::move(queue_mapping),resources,names});
  }
  // An interval transition consumes one stage or a selected adjacent pair.
  // Keep the fusion choices until the terminal state: poll image deduplication
  // is global and cannot be replaced by additive per-edge rebates.
  struct State { std::vector<int> pairs; int shared=0,registers=0; };
  std::vector<std::vector<State>> states(model.stages.size()+1);
  states[0].push_back({{},domain.resident_shared_floor,0});
  for (std::size_t stage=0;stage<model.stages.size();++stage) {
    for (auto const& state:states[stage]) {
      auto single=state;
      single.shared=std::max(single.shared,domain.stage_traits[stage].smem_bytes);
      single.registers=std::max(single.registers,domain.stage_registers[stage]);
      states[stage+1].push_back(std::move(single));
      for (std::size_t pair=0;pair<pairs.size();++pair) if (pairs[pair].producer==int(stage)) {
        auto fused=state; fused.pairs.push_back(pair);
        fused.shared=std::max(fused.shared,pairs[pair].resources.shared_bytes);
        fused.registers=std::max(fused.registers,pairs[pair].resources.registers);
        states[stage+2].push_back(std::move(fused));
      }
    }
    states[stage].clear();
  }
  if (alternatives) alternatives->clear();
  for (auto& state:states.back()) {
    std::vector<std::pair<std::string,std::string>> pattern;
    for (int pair:state.pairs) pattern.push_back(pairs[pair].names);
    std::sort(pattern.begin(),pattern.end());
    auto evidence=domain.compiled_registers.find(pattern);
    if (evidence!=domain.compiled_registers.end()) {
      if (evidence->second<=0) throw std::invalid_argument("invalid fused kernel register evidence");
      state.registers=evidence->second;
    } else if (domain.require_compiled_registers) {
      throw std::invalid_argument("fusion pattern lacks whole-kernel register evidence");
    }
  }
  ChainDpSolution best;
  auto resident=[&](State const& state) {
    long long shared=static_cast<long long>(state.shared)+domain.static_shared_bytes;
    if (state.shared>cost_->target().res.max_dynamic_smem_per_cta ||
        shared>cost_->target().res.max_smem_per_sm ||
        static_cast<long long>(state.registers)*threads>cost_->target().res.regs_per_sm)
      return 0;
    return CtasPerSm(static_cast<int>(shared),state.registers);
  };
  for (int r=1;r<=options.max_ctas_per_sm;++r) {
    if (std::none_of(states.back().begin(),states.back().end(),[&](auto const& state) { return resident(state)==r; })) continue;
    RuntimeProjectionOptions projection_options{cost_->target().res.num_sms*r,threads,cost_->options().kappa};
    auto original=ProjectRuntimeQueues(model,domain.plan,projection_options);
    auto priced=model;
    AttachProjectedEventMetrics(priced,domain.plan,original);
    auto baseline=cost_->Evaluate(priced,configs,{r});
    std::vector<FusionTaskPrice> prices;
    for (auto const& pair:pairs)
      prices.push_back(PriceFusionTasks(pair.runtime,*cost_,domain.stage_traits[pair.producer],
          domain.stage_traits[pair.consumer],{r},model));
    for (auto const& state:states.back()) {
      if (resident(state)!=r) continue;
      auto projection=original;
      ChainDpSolution solution;
      solution.feasible=true; solution.configs=configs; solution.residency={r};
      solution.attention=model.attention_plan ? model.attention_plan->choices : std::vector<codegen::AttentionRuntimeRecord>{};
      solution.max_smem_bytes=state.shared; solution.max_registers=state.registers;
      solution.cost=baseline;
      for (int index:state.pairs) {
        auto const& pair=pairs[index];
        int p=-1,c=-1;
        for (int stage=0;stage<int(projection.stages.size());++stage) {
          if (projection.stages[stage].logical_stage==pair.producer) p=stage;
          if (projection.stages[stage].logical_stage==pair.consumer) c=stage;
        }
        projection=FuseProjectedQueues(projection,p,c,pair.queue_mapping,projection_options).projection;
        solution.fusion.push_back(pair.names);
        // Replace the actual two baseline stage prices, not a second estimate
        // of them, so the unfused branch stays bit-identical to Evaluate.
        auto config=[&](int stage) { int gemm=model.stages[stage].gemm;
          return gemm<0 ? (configs.empty() ? GemmConfig{} : configs.front()) : configs[gemm]; };
        solution.cost.task_ns_sum-=cost_->TaskStageNs(model,pair.producer,config(pair.producer),{r});
        solution.cost.task_ns_sum-=cost_->TaskStageNs(model,pair.consumer,config(pair.consumer),{r});
        solution.cost.task_ns_sum+=prices[index].fused_ns;
      }
      solution.cost.stage_count=projection.stages.size();
      AttachProjectedEventMetrics(priced,domain.plan,projection);
      solution.cost.event_ns=cost_->EventNs(priced,configs,{r},solution.cost.stage_count);
      solution.cost.total_ns=solution.cost.task_ns_sum+solution.cost.combine_ns+
          solution.cost.barrier_ns+solution.cost.event_ns;
      if (alternatives) alternatives->push_back({solution,
          projection.runtime_task_refs.Eval(model.MetricBindings()),
          projection.runtime_wait_entries.Eval(model.MetricBindings())});
      if (!best.feasible || solution.cost.total_ns<best.cost.total_ns) best=std::move(solution);
    }
  }
  return best;
}
}  // namespace tilemega::solver
