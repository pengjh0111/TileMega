// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ChainDP.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Analysis/ISLContext.h>
#include <gmpxx.h>
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
namespace {
using QP=analysis::QuasiPolynomial;
QP Constant(std::string const& p,long first,long last,double value) {
  return QP::FromIslText("["+p+"] -> { "+mpq_class(value).get_str()+" : "+
      std::to_string(first)+"<="+p+"<="+std::to_string(last)+" }");
}
struct State {
  long begin,end;
  QP ns;
  std::vector<int> choice;
  int residency;
  bool exact;
};
// Feasibility stays outside the polynomial. In particular absent support is
// never used as a zero-cost infeasible DP state.
std::vector<State> Minimize(std::vector<State> const& states,std::string const& parameter) {
  std::set<long> boundaries;
  for (auto const& s:states) { boundaries.insert(s.begin); boundaries.insert(s.end+1); }
  std::vector<long> cuts(boundaries.begin(),boundaries.end());
  std::vector<State> result;
  for (std::size_t i=1;i<cuts.size();++i) {
    long begin=cuts[i-1],end=cuts[i]-1;
    std::vector<State const*> present;
    for (auto const& s:states) if (s.begin<=begin && end<=s.end) present.push_back(&s);
    if (present.empty()) continue;
    std::stable_sort(present.begin(),present.end(),[](auto a,auto b) {
      return std::tie(a->residency,a->choice)<std::tie(b->residency,b->choice);
    });
    std::vector<QP> prices;
    for (auto s:present) prices.push_back(s->ns);
    for (auto const& region:QuadraticEnvelope(prices,parameter,begin,end,false)) {
      auto chosen=*present.at(region.choice);
      chosen.begin=region.begin; chosen.end=region.end;
      chosen.ns=QP::FromIslText("["+parameter+"] -> { ("+region.coefficients[0]+")+("+
          region.coefficients[1]+")*"+parameter+"+("+region.coefficients[2]+")*"+parameter+"^2 : "+
          std::to_string(region.begin)+"<="+parameter+"<="+std::to_string(region.end)+" }");
      result.push_back(std::move(chosen));
    }
  }
  return result;
}
}

SymbolicDpSolution ChainDP::SolveSymbolicParameter(ModelDescription const& symbolic,
    FiniteParameterDomain const& domain,ChainDpOptions options) const {
  analysis::IslReferenceAudit audit(__func__);
  if (domain.begin<=0 || domain.end<domain.begin || domain.parameter!=symbolic.dims.seq_parameter ||
      !symbolic.dims.past_parameter.empty() || !cost_->options().unified_task_cost ||
      !cost_->options().cg_interface || cost_->options().l2_events || symbolic.attention_plan ||
      !options.interface_term || !options.general_transitions)
    throw std::invalid_argument("symbolic DP currently requires unified CG-interface L1 with fixed-past unchunked attention");
  auto model=symbolic;
  // Instantiate geometry symbolically; dims.seq is not read by symbolic costs.
  model.dims.seq_parameter.clear();
  std::string const& parameter=model.seq_metric_parameter;
  if (parameter.empty()) throw std::invalid_argument("symbolic DP requires canonical CG seq role");
  for (auto const& [name,value]:domain.fixed.values) {
    if (name==domain.parameter || name==parameter) throw std::invalid_argument("seq cannot be fixed inside symbolic domain");
    model.metric_bindings.Bind(name,value);
  }
  auto stages=GemmStages(model);
  int layers=stages.size();
  SymbolicDpSolution result; result.parameter=parameter;
  if (!layers || candidates_.empty()) return result;
  std::vector<int> ordinal(model.gemms.size(),-1),last(layers);
  for (int i=0;i<layers;++i) { ordinal.at(model.stages.at(stages[i]).gemm)=i; last[i]=i; }
  struct Factor { int producer,consumer; std::vector<int> variables; std::map<std::vector<int>,QP> prices; };
  std::set<std::pair<int,int>> pairs;
  for (auto const& edge:model.coupling_metrics.edges) pairs.emplace(edge.producer,edge.consumer);
  for (int stage:stages) pairs.emplace(stage,stage);
  std::vector<Factor> factors;
  for (auto const& [p,c]:pairs) {
    Factor f{p,c,{},{}};
    for (int stage:{p,c}) if (model.stages.at(stage).gemm>=0) f.variables.push_back(ordinal.at(model.stages.at(stage).gemm));
    std::sort(f.variables.begin(),f.variables.end());
    f.variables.erase(std::unique(f.variables.begin(),f.variables.end()),f.variables.end());
    if (!f.variables.empty()) for (int v:f.variables) last[v]=std::max(last[v],f.variables.back());
    factors.push_back(std::move(f));
  }
  auto price=[&](Factor& f,std::vector<int> const& choice) {
    std::vector<int> key;
    for (int v:f.variables) key.push_back(choice.at(v));
    auto found=f.prices.find(key);
    if (found!=f.prices.end()) return found->second;
    std::vector<GemmConfig> configs(model.gemms.size(),candidates_.front().config);
    for (int v:f.variables) configs.at(model.stages.at(stages[v]).gemm)=candidates_.at(choice.at(v)).config;
    std::vector<QP> pieces;
    for (auto const& edge:InstantiateModelCouplings(model,configs,std::pair{f.producer,f.consumer}))
      pieces.push_back(cost_->SymbolicInterfaceEdgeNs(edge,model,parameter,domain.begin,domain.end));
    auto value=QP::Sum(pieces); f.prices.emplace(key,value); return value;
  };
  std::vector<int> ctas;
  for (auto const& candidate:candidates_) {
    if (candidate.registers<=0 || candidate.smem_bytes<=0)
      throw std::invalid_argument("symbolic DP requires compiled resource evidence");
    ctas.push_back(CtasPerSm(std::max(candidate.smem_bytes,model.NonGemmSharedBytes()),candidate.registers));
  }
  std::vector<std::vector<bool>> allowed(layers,std::vector<bool>(candidates_.size(),true));
  if (!options.per_operator_candidates.empty()) {
    if (options.per_operator_candidates.size()!=stages.size()) throw std::invalid_argument("invalid symbolic admissibility rank");
    for (int i=0;i<layers;++i) {
      std::fill(allowed[i].begin(),allowed[i].end(),false);
      for (int c:options.per_operator_candidates[i]) {
        if (c<0 || c>=int(candidates_.size())) throw std::invalid_argument("invalid symbolic candidate id");
        allowed[i][c]=true;
      }
    }
  }
  std::vector<State> final;
  for (int r=1;r<=options.max_ctas_per_sm;++r) {
    std::vector<int> admitted;
    for (int c=0;c<int(candidates_.size());++c) if (ctas[c]>=r) admitted.push_back(c);
    if (std::none_of(admitted.begin(),admitted.end(),[&](int c) { return ctas[c]==r; })) continue;
    auto barrier=Constant(parameter,domain.begin,domain.end,cost_->BarrierNs({r}));
    std::vector<QP> fixed;
    for (int s=0;s<int(model.stages.size());++s) if (model.stages[s].kind!=StageKind::kGemm)
      fixed.push_back(cost_->SymbolicStageNs(model,s,candidates_.front().config,{r},parameter,domain.begin,domain.end)
          .Add(barrier.Scale(model.RuntimeStages(s))));
    for (auto& f:factors) if (f.variables.empty()) fixed.push_back(price(f,{}));
    auto prefix=QP::Sum(fixed);
    std::vector<std::map<int,QP>> unary(layers);
    for (int i=0;i<layers;++i) for (int c:admitted) if (allowed[i][c]) {
      auto const& config=candidates_[c].config;
      auto const& gemm=model.gemms.at(model.stages.at(stages[i]).gemm);
      int chunks=cost_->Chunks(gemm,config);
      auto value=cost_->SymbolicStageNs(model,stages[i],config,{r},parameter,domain.begin,domain.end).Add(barrier);
      if (chunks>1) value=value.Add(cost_->SymbolicCombineNs(gemm,chunks,parameter,domain.begin,domain.end)).Add(barrier);
      unary[i].emplace(c,std::move(value));
    }
    std::vector<std::vector<int>> groups;
    if (options.mode==DpMode::kPerOperator) groups.push_back(admitted);
    else if (options.mode==DpMode::kUniform) for (int c:admitted) groups.push_back({c});
    else {
      std::map<std::array<int,4>,std::vector<int>> shapes;
      for (int c:admitted) { auto const& g=candidates_[c].config; shapes[{g.tile_m,g.tile_n,g.tile_k,g.stages}].push_back(c); }
      for (auto const& [shape,group]:shapes) groups.push_back(group);
    }
    for (auto const& group:groups) {
      std::map<std::vector<int>,std::vector<State>> prev;
      prev[{0}].push_back({domain.begin,domain.end,prefix,std::vector<int>(layers,-1),r,false});
      for (int i=0;i<layers;++i) {
        std::map<std::vector<int>,std::vector<State>> next;
        for (auto const& [key,states]:prev) for (auto const& state:states) for (int c:group) {
          if (!allowed[i][c]) continue;
          ++result.transitions;
          auto step=state; step.choice[i]=c; step.exact|=ctas[c]==r;
          step.ns=step.ns.Add(unary[i].at(c));
          for (auto& f:factors) if (!f.variables.empty() && f.variables.back()==i) step.ns=step.ns.Add(price(f,step.choice));
          std::vector<int> new_key{int(step.exact)};
          for (int v=0;v<=i;++v) if (last[v]>i) new_key.push_back(step.choice[v]);
          next[new_key].push_back(std::move(step));
        }
        for (auto& [key,states]:next) states=Minimize(states,parameter);
        prev=std::move(next);
      }
      for (auto const& [key,states]:prev) for (auto const& state:states) if (state.exact) final.push_back(state);
    }
  }
  for (auto const& state:Minimize(final,parameter)) {
    SymbolicDpPiece piece{state.begin,state.end,state.ns,std::vector<GemmConfig>(model.gemms.size()),{state.residency}};
    for (int i=0;i<layers;++i) piece.configs.at(model.stages.at(stages[i]).gemm)=candidates_.at(state.choice[i]).config;
    result.pieces.push_back(std::move(piece));
  }
  return result;
}
}  // namespace tilemega::solver
