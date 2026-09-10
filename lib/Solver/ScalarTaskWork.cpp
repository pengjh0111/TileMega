// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <map>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
analysis::TaskWork DeriveRuntimeScalarWork(ModelDescription const& model,
    ModelTaskSemantics const& semantic,analysis::OperatorNode const& task,
    analysis::TaskWork work,int threads,RuntimeScalarAccess* accesses) {
  analysis::IslReferenceAudit audit(__func__);
#if !TILEMEGA_SCALAR_TASK_WORK
  throw std::runtime_error("scalar ownership-derived work is disabled");
#endif
  using namespace analysis;
  auto const& stage=model.stages.at(semantic.stage);
  auto ownership=ProjectScalarTaskOwnership(semantic,task,stage,threads);
  auto writes=ownership.ApplyRange(ElementAccess(task,BuildWriteMap(task),{},AccessDomain::kPhysicalTensor));
  std::map<std::string,CouplingRelation> reads;
  if (!semantic.op.element_reads.empty()) {
    for (auto const& read:semantic.op.element_reads)
      reads[read.tensor.name]=reads[read.tensor.name].Union(
          ownership.ApplyRange(ExactElementRead(semantic.op,task,read,{})));
  } else {
    for (std::size_t i=0;i<task.operands.size();++i) {
      auto read=BuildReadMap(task,i);
      reads[read.tensor.name]=reads[read.tensor.name].Union(
          ownership.ApplyRange(ElementAccess(task,read,{},AccessDomain::kPhysicalTensor)));
    }
  }
  if (stage.kind==StageKind::kKVAppend && semantic.element_chunk) {
    // Tile ownership preloads the retained prefix on the host; element
    // ownership copies it in this same task. Its range comes from the CG
    // append origin. Count the additional region, not a handwritten byte
    // estimate, and keep the two output regions in one address-space union.
    if (stage.operands.size()!=3 || semantic.op.result.axes.size()!=2 ||
        semantic.op.result_effect.state_object!="kv_cache")
      throw std::invalid_argument("KV prefix copy has no CG state/operand contract");
    auto past=semantic.op.result.axes[0].origin;
    long cols=semantic.op.result.axes[1].extent.Eval({},{});
    if (stage.width<=0 || cols%stage.width) throw std::invalid_argument("invalid KV head layout");
    std::string parameters;
    for (auto const& p:past.FreeSymbols()) parameters+=(parameters.empty() ? "" : ",")+p;
    if (!parameters.empty()) parameters="["+parameters+"] -> ";
    auto prefix=CouplingRelation::FromIslText(parameters+"{ [q] -> [i] : 0 <= i < ("+
        past.ToIslText()+")*"+std::to_string(cols)+" and q=floord(i,"+std::to_string(threads)+") }");
    reads["cg_buffer_"+std::to_string(stage.operands[1])]=prefix;
    // Literal head enumeration avoids a parameter-times-coordinate product:
    // the number of heads is a fixed structural CG dimension, not sampled theta.
    for (long head=0;head<cols/stage.width;++head) {
      auto map=CouplingRelation::FromIslText(parameters+"{ [q] -> [element0,element1] : "
          "0 <= element0 < ("+past.ToIslText()+") and "+std::to_string(head*stage.width)+
          " <= element1 < "+std::to_string((head+1)*stage.width)+" and q=floord(("+
          std::to_string(head*stage.width)+"*("+past.ToIslText()+")+element0*"+
          std::to_string(stage.width)+"+element1-"+std::to_string(head*stage.width)+"),"+
          std::to_string(threads)+") }");
      writes=writes.Union(map);
    }
  }
  std::vector<QuasiPolynomial> counts;
  for (auto const& [name,relation]:reads) counts.push_back(relation.Card());
  work.read_elements=QuasiPolynomial::Sum(counts);
  work.write_elements=writes.Card();
  work.task_count=writes.Reverse().ImageCard();
  if (accesses) *accesses={std::move(ownership),std::move(writes),std::move(reads)};
  return work;
}
}  // namespace tilemega::solver
