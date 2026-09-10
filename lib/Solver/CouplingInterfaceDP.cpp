// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ChainDP.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
ChainDpSolution ChainDP::SolveCouplingInterfaces(ModelDescription const& model,
    ChainDpOptions options,ChainDpStats* stats) const {
  analysis::IslReferenceAudit audit(__func__);
  if (!options.interface_term || !options.general_transitions)
    throw std::invalid_argument("CG interfaces require exact transitions; use the historical cost switch for control");
  if (model.task_semantics.empty()) throw std::invalid_argument("CG interface DP requires semantic access inputs");
  auto started=std::chrono::steady_clock::now();
  auto gemm_stages=GemmStages(model);
  int layers=gemm_stages.size();
  ChainDpStats local;
  local.candidates=candidates_.size(); local.operators=layers;
  ChainDpSolution best;
  double best_total=std::numeric_limits<double>::infinity();
  if (!layers || candidates_.empty()) { if (stats) *stats=local; return best; }
  std::vector<int> ordinal(model.gemms.size(),-1);
  for (int i=0;i<layers;++i) ordinal.at(model.stages.at(gemm_stages[i]).gemm)=i;
  struct Factor {
    int producer,consumer;
    std::vector<int> variables;
    std::map<std::vector<int>,double> values;
  };
  std::set<std::pair<int,int>> pairs;
  for (auto const& edge:model.coupling_metrics.edges) pairs.emplace(edge.producer,edge.consumer);
  // A split adds an internal partial->combine edge absent from unsplit CG.
  for (int stage:gemm_stages) pairs.emplace(stage,stage);
  std::vector<Factor> factors;
  std::vector<int> last(layers);
  for (int i=0;i<layers;++i) last[i]=i;
  for (auto const& [producer,consumer]:pairs) {
    Factor f{producer,consumer,{},{}};
    for (int stage:{producer,consumer}) {
      int gemm=model.stages.at(stage).gemm;
      if (gemm>=0) f.variables.push_back(ordinal.at(gemm));
    }
    std::sort(f.variables.begin(),f.variables.end());
    f.variables.erase(std::unique(f.variables.begin(),f.variables.end()),f.variables.end());
    if (!f.variables.empty()) for (int v:f.variables) last[v]=std::max(last[v],f.variables.back());
    factors.push_back(std::move(f));
  }
  auto price=[&](Factor& f,std::vector<int> const& choices) {
    std::vector<int> key;
    for (int v:f.variables) key.push_back(choices.at(v));
    auto found=f.values.find(key);
    if (found!=f.values.end()) return found->second;
    std::vector<GemmConfig> configs(model.gemms.size(),candidates_.front().config);
    for (int v:f.variables) configs.at(model.stages.at(gemm_stages[v]).gemm)=candidates_.at(choices.at(v)).config;
    double total=0;
    for (auto const& edge:InstantiateModelCouplings(model,configs,std::pair{f.producer,f.consumer}))
      total+=cost_->InterfaceEdgeNs(edge,model);
    f.values.emplace(std::move(key),total);
    return total;
  };
  std::vector<int> ctas;
  for (auto const& candidate:candidates_) ctas.push_back(CtasPerSm(candidate.smem_bytes,candidate.registers));
  for (int r=1;r<=options.max_ctas_per_sm;++r) {
    std::vector<int> admitted;
    for (int c=0;c<int(candidates_.size());++c) if (ctas[c]>=r) admitted.push_back(c);
    if (admitted.empty()) continue;
    std::vector<std::vector<char>> allowed(layers,std::vector<char>(candidates_.size(),1));
    if (!options.per_operator_candidates.empty()) {
      if (int(options.per_operator_candidates.size())!=layers)
        throw std::invalid_argument("CG interface DP: one admissibility set per GEMM is required");
      for (int i=0;i<layers;++i) {
        std::fill(allowed[i].begin(),allowed[i].end(),0);
        for (int c:options.per_operator_candidates[i]) {
          if (c<0 || c>=int(candidates_.size())) throw std::invalid_argument("invalid CG interface candidate id");
          allowed[i][c]=1;
        }
      }
    }
    bool exact=false,complete=true;
    for (int i=0;i<layers;++i) {
      bool any=false;
      for (int c:admitted) if (allowed[i][c]) { any=true; exact|=ctas[c]==r; }
      complete&=any;
    }
    if (!exact || !complete) continue;
    ++local.residency_levels;
    Residency residency{r};
    double barrier=cost_->BarrierNs(residency),fixed=0;
    for (auto const& stage:model.stages) if (stage.kind!=StageKind::kGemm)
      fixed+=(cost_->options().unified_task_cost ? cost_->TaskStageNs(model,int(&stage-model.stages.data()),
          candidates_.front().config,residency) : cost_->NonGemmStageNs(stage,model.dims,residency))+barrier;
    std::vector<int> empty(layers,0);
    for (auto& f:factors) if (f.variables.empty()) fixed+=price(f,empty);
    std::vector<std::vector<double>> unary(layers,std::vector<double>(candidates_.size()));
    for (int i=0;i<layers;++i) for (int c:admitted) {
      int chunks=0;
      auto const& gemm=model.gemms.at(model.stages.at(gemm_stages[i]).gemm);
      double ns;
      if (cost_->options().unified_task_cost) {
        chunks=cost_->Chunks(gemm,candidates_[c].config);
        ns=cost_->TaskStageNs(model,gemm_stages[i],candidates_[c].config,residency)+barrier;
      } else ns=cost_->GemmStageNs(gemm,candidates_[c].config,residency,model,&chunks)+barrier;
      if (chunks>1) ns+=cost_->CombineStageNs(gemm,chunks,model.dims)+barrier;
      unary[i][c]=ns;
    }
    std::vector<std::vector<int>> groups;
    if (options.mode==DpMode::kPerOperator) groups.push_back(admitted);
    else if (options.mode==DpMode::kUniform) for (int c:admitted) groups.push_back({c});
    else {
      std::map<std::array<int,4>,std::vector<int>> shapes;
      for (int c:admitted) {
        auto const& g=candidates_[c].config;
        shapes[{g.tile_m,g.tile_n,g.tile_k,g.stages}].push_back(c);
      }
      for (auto const& [shape,group]:shapes) groups.push_back(group);
    }
    for (auto const& group:groups) {
      struct State { double ns; std::vector<int> choice; bool exact; };
      std::map<std::vector<int>,State> prev;
      prev.emplace(std::vector<int>{0},State{fixed,std::vector<int>(layers,-1),false});
      for (int i=0;i<layers;++i) {
        std::vector<int> live;
        for (int v=0;v<=i;++v) if (last[v]>i) live.push_back(v);
        local.interface_frontier_width=std::max(local.interface_frontier_width,int(live.size()));
        std::map<std::vector<int>,State> next;
        for (auto const& [old_key,state]:prev) for (int c:group) {
          if (!allowed[i][c]) continue;
          ++local.transitions;
          State step=state; step.choice[i]=c; step.exact|=ctas[c]==r;
          step.ns+=unary[i][c];
          for (auto& f:factors) if (!f.variables.empty() && f.variables.back()==i)
            step.ns+=price(f,step.choice);
          std::vector<int> key{int(step.exact)};
          for (int v:live) key.push_back(step.choice[v]);
          auto found=next.find(key);
          if (found==next.end() || step.ns<found->second.ns ||
              (step.ns==found->second.ns && step.choice<found->second.choice)) next[key]=std::move(step);
        }
        prev=std::move(next);
      }
      for (auto const& [key,state]:prev) {
        if (!state.exact || !(state.ns<best_total)) continue;
        ChainDpSolution solution; solution.feasible=true; solution.residency=residency;
        solution.configs.resize(model.gemms.size());
        for (int i=0;i<layers;++i) {
          auto const& c=candidates_[state.choice[i]];
          solution.configs.at(model.stages.at(gemm_stages[i]).gemm)=c.config;
          solution.max_smem_bytes=std::max(solution.max_smem_bytes,c.smem_bytes);
          solution.max_registers=std::max(solution.max_registers,c.registers);
        }
        solution.cost=cost_->Evaluate(model,solution.configs,residency);
        local.decomposition_error_ns=std::max(local.decomposition_error_ns,std::fabs(state.ns-solution.cost.total_ns));
        best=std::move(solution); best_total=state.ns;
      }
    }
  }
  for (auto const& f:factors) if (!f.values.empty()) {
    double lo=std::numeric_limits<double>::infinity(),hi=0;
    for (auto const& [key,value]:f.values) { lo=std::min(lo,value); hi=std::max(hi,value); }
    local.interface_spread_ns=std::max(local.interface_spread_ns,hi-lo);
  }
  local.solve_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
  if (stats) *stats=local;
  return best;
}
}  // namespace tilemega::solver
