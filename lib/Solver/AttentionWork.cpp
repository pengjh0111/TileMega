// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/AttentionWork.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <limits>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
void ApplyAttentionCostPlan(ModelDescription& model,
    std::vector<codegen::AttentionRuntimeRecord> const& choices,int threads) {
  analysis::IslReferenceAudit audit(__func__);
  if (threads<=0 || (threads&(threads-1)))
    throw std::invalid_argument("attention cost plan requires valid CTA width");
  if (choices.empty()) {
    model.attention_plan.reset(); model.coupling_metrics.runtime.reset(); return;
  }
  if (choices.size()!=model.stages.size()) throw std::invalid_argument("attention cost plan length mismatch");
  auto plan=std::make_shared<AttentionCostPlan>(); plan->choices=choices;
  std::vector<analysis::QuasiPolynomial> workspace;
  for (std::size_t i=0;i<model.stages.size();++i) {
    auto const& stage=model.stages[i]; auto const& choice=choices[i];
    auto extent=choice.chunk_extent ? choice.chunk_extent : TILEMEGA_ATTENTION_MAX_TOTAL;
    if (choice.chunks==0) throw std::invalid_argument("attention chunk count must be positive");
    if (extent<=0 || extent>std::numeric_limits<int>::max()/sizeof(float))
      throw std::invalid_argument("attention storage extent overflow");
    plan->shared_bytes=std::max(plan->shared_bytes,int(sizeof(float))*codegen::SimtSharedElements(
        static_cast<codegen::TaskKind>(stage.kind),threads,extent));
    if (choice.chunks<=1) continue;
    auto found=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),
        [&](auto const& semantic) { return semantic.stage==static_cast<int>(i); });
    if (found==model.task_semantics.end()) throw std::invalid_argument("attention plan has no CG semantic task");
    auto phases=DeriveAttentionPhaseWork(model,*found,choice,threads);
    for (std::string const& name:{found->op.name+".scores",found->op.name+".partials"}) {
      analysis::CouplingRelation writes;
      for (auto const& phase:phases) {
        auto output=phase.accesses.writes.find(name);
        if (output!=phase.accesses.writes.end())
          writes=writes.empty() ? output->second : writes.Union(output->second);
      }
      if (writes.empty()) throw std::invalid_argument("attention workspace has no writer");
      workspace.push_back(writes.ImageCard().Scale(sizeof(float)));
    }
    plan->stages.emplace(i,std::move(phases));
  }
  plan->workspace_bytes=analysis::QuasiPolynomial::Sum(workspace);
  model.attention_plan=std::move(plan);
  // Counts for the previous expansion cannot price a changed chunk plan.
  model.coupling_metrics.runtime.reset();
}

std::vector<AttentionPhaseWork> DeriveAttentionPhaseWork(
    ModelDescription const& model,ModelTaskSemantics const& semantic,
    codegen::AttentionRuntimeRecord choice,int threads) {
  analysis::IslReferenceAudit audit(__func__);
#if !TILEMEGA_ATTENTION_PHASE_WORK
  throw std::runtime_error("attention phase work disabled");
#endif
  using namespace analysis;
  using codegen::AttentionPhase;
  auto const& stage=model.stages.at(semantic.stage);
  auto const& op=semantic.op;
  if (stage.kind!=StageKind::kAttention || op.arithmetic!="attention" ||
      op.result.axes.size()!=2 || op.operands.size()!=3 ||
      op.operands[1].tensor.axes.size()!=2 || stage.width<=0 || stage.extent<=0 ||
      stage.group<=0 || stage.extent%stage.group || choice.chunks<=1 ||
      choice.chunks>std::numeric_limits<int>::max() ||
      choice.chunk_extent==0 || choice.chunk_extent>std::numeric_limits<int>::max()/sizeof(float) ||
      threads<=0 || (threads&(threads-1)))
    throw std::invalid_argument("incomplete chunk attention semantic/resource contract");
  if (!model.dims.IsSymbolic() && (model.dims.total<=0 ||
      (static_cast<long>(model.dims.total)+choice.chunks-1)/choice.chunks>choice.chunk_extent))
    throw std::invalid_argument("attention chunk scratch capacity is insufficient");
  auto seq=op.result.axes[0].extent;
  auto total=op.operands[1].tensor.axes[0].extent;
  auto past=total+seq*ClosedForm::Constant(-1);
  auto heads=stage.extent,width=stage.width;
  if (static_cast<long>(heads)*width>std::numeric_limits<int>::max())
    throw std::invalid_argument("attention head layout exceeds runtime coordinate width");
  if (op.result.axes[1].extent.Eval({},{})!=static_cast<long>(heads)*width)
    throw std::invalid_argument("attention output extent differs from semantic head layout");
  std::set<std::string> symbols;
  for (auto const& expression:{seq,total})
    for (auto const& name:expression.FreeSymbols()) symbols.insert(name);
  std::string prefix;
  for (auto const& name:symbols) prefix+=(prefix.empty() ? "" : ",")+name;
  if (!prefix.empty()) prefix="["+prefix+"] -> ";
  auto S=seq.ToIslText(),T=total.ToIslText(),P=past.ToIslText();
  std::string domain="0<=s<("+S+") and ("+T+")>=("+S+") and ("+S+")>0";
  auto score_name=op.name+".scores",partial_name=op.name+".partials";
  int model_bytes=model.dtype==ScalarType::kBF16 ? 2 : 4;
  std::vector<AttentionPhaseWork> result;
  for (auto phase:codegen::kAttentionExpandedPhases) {
    AttentionPhaseWork work; work.phase=phase;
    bool partitioned=phase==AttentionPhase::kScores || phase==AttentionPhase::kPartialValue;
    int chunks=partitioned ? choice.chunks : 1;
    work.tasks=CouplingRelation::FromIslText(prefix+"{ [q] -> [] : 0<=q<("+S+")*"+
        std::to_string(heads*static_cast<long>(chunks))+" and ("+T+")>=("+S+") and ("+S+")>0 }");
    work.task_count=work.tasks.Card().SumDomain();
    CouplingRelation arithmetic_outputs;
    auto append=[&](auto& table,std::string const& tensor,int bytes,CouplingRelation access) {
      auto [it,inserted]=table.emplace(tensor,access);
      if (!inserted) it->second=it->second.Union(access);
      auto [size,fresh]=work.element_bytes.emplace(tensor,bytes);
      if (!fresh && size->second!=bytes) throw std::invalid_argument("attention tensor dtype conflict");
    };
    for (int h=0;h<heads;++h) for (int chunk=0;chunk<chunks;++chunk) {
      auto key="floord(("+T+")*"+std::to_string(chunk)+","+std::to_string(chunks)+")<=k and k<floord(("+
          T+")*"+std::to_string(chunk+1)+","+std::to_string(chunks)+")";
      auto owned="q=(s*"+std::to_string(heads)+"+"+std::to_string(h)+")*"+
          std::to_string(chunks)+"+"+std::to_string(chunk)+" and "+domain;
      auto relation=[&](std::vector<std::string> const& range,std::string const& condition) {
        std::string tuple="[",equalities;
        for (std::size_t i=0;i<range.size();++i) {
          auto dim="e"+std::to_string(i);
          tuple+=(i ? "," : "")+dim;
          equalities+=" and "+dim+"=("+range[i]+")";
        }
        return CouplingRelation::FromIslText(prefix+"{ [q] -> "+tuple+"] : exists s,k,d: "+owned+
            " and "+condition+equalities+" }");
      };
      auto query="s*"+std::to_string(heads)+"+"+std::to_string(h);
      std::vector<std::string> score{query,"k"};
      std::vector<std::string> vector{"s",std::to_string(h*width)+"+d"};
      std::vector<std::string> cache{"k",std::to_string((h/stage.group)*width)+"+d"};
      std::vector<std::string> partial{query,std::to_string(chunk),"d"};
      auto components="0<=d<"+std::to_string(width);
      if (phase==AttentionPhase::kScores) {
        auto valid=key+" and k<=("+P+")+s";
        append(work.accesses.reads,op.operands[0].tensor.name,model_bytes,relation(vector,valid+" and "+components));
        append(work.accesses.reads,op.operands[1].tensor.name,model_bytes,relation(cache,valid+" and "+components));
        append(work.accesses.writes,score_name,sizeof(float),relation(score,key));
        auto active=relation(score,valid);
        arithmetic_outputs=arithmetic_outputs.empty() ? active : arithmetic_outputs.Union(active);
      } else if (phase==AttentionPhase::kNormalize) {
        auto scores=relation(score,key);
        append(work.accesses.reads,score_name,sizeof(float),scores);
        append(work.accesses.writes,score_name,sizeof(float),scores);
        arithmetic_outputs=arithmetic_outputs.empty() ? scores : arithmetic_outputs.Union(scores);
      } else if (phase==AttentionPhase::kPartialValue) {
        append(work.accesses.reads,score_name,sizeof(float),relation(score,key));
        append(work.accesses.reads,op.operands[2].tensor.name,model_bytes,relation(cache,key+" and "+components));
        append(work.accesses.writes,partial_name,sizeof(float),relation(partial,components));
        auto mac=relation({"s",std::to_string(h),"k","d"},key+" and "+components);
        arithmetic_outputs=arithmetic_outputs.empty() ? mac : arithmetic_outputs.Union(mac);
      } else {
        append(work.accesses.reads,partial_name,sizeof(float),relation({query,"k","d"},
            "0<=k<"+std::to_string(choice.chunks)+" and "+components));
        auto outputs=relation(vector,components);
        append(work.accesses.writes,op.result.name,model_bytes,outputs);
        arithmetic_outputs=arithmetic_outputs.empty() ? outputs : arithmetic_outputs.Union(outputs);
      }
    }
    auto bytes=[&](auto const& accesses) {
      std::vector<QuasiPolynomial> terms;
      for (auto const& [name,relation]:accesses)
        terms.push_back(relation.Card().Scale(work.element_bytes.at(name)));
      return QuasiPolynomial::Sum(terms);
    };
    work.read_bytes=bytes(work.accesses.reads); work.write_bytes=bytes(work.accesses.writes);
    ArithmeticInputs inputs;
    inputs.width=width; inputs.reduction=QuasiPolynomial::Constant(choice.chunks);
    inputs.dtype=op.dtype;
    auto name=phase==AttentionPhase::kScores ? "attention_scores" :
              phase==AttentionPhase::kNormalize ? "attention_normalize" :
              phase==AttentionPhase::kPartialValue ? "attention_mac" : "attention_sum";
    auto signature=InstantiateArithmetic(name,inputs);
    auto outputs=arithmetic_outputs.Card();
    work.flops=outputs.Scale(signature.flops_per_output_element.numerator.Eval({}))
        .ScaleRational("1/"+std::to_string(signature.flops_per_output_element.denominator));
    work.transcendental=outputs.Scale(signature.transcendental_per_output_element.numerator.Eval({}))
        .ScaleRational("1/"+std::to_string(signature.transcendental_per_output_element.denominator));
    int load=work.flow.Add(codegen::ScalarPhase::kLoad);
    int compute=work.flow.Add(codegen::ScalarPhase::kArithmetic,{load});
    if (phase==AttentionPhase::kScores) {
      compute=work.flow.Add(codegen::ScalarPhase::kPublish,{compute});
      work.shared_bytes=choice.chunk_extent*sizeof(float);
    }
    work.flow.Add(codegen::ScalarPhase::kStore,{compute});
    result.push_back(std::move(work));
  }
  return result;
}
}  // namespace tilemega::solver
