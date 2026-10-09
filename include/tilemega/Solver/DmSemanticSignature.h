// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingCache.h>
#include <map>

namespace tilemega::solver {
// The legacy key predates independent ownership spaces, side stores and
// named bindings. Canonicalize their identities together for DM classes.
inline std::string DmSemanticSignature(analysis::SemanticOp op) {
  std::map<std::string,std::string> tensors,producers,aliases,states,layouts;
  auto rename=[](auto& names,std::string const& name,char const* prefix) {
    if(name.empty())return name;
    auto [found,added]=names.emplace(name,std::string(prefix)+std::to_string(names.size()));
    return found->second;
  };
  auto tensor=[&](analysis::TensorSpace& space) {
    space.name=rename(tensors,space.name,"tensor");
    space.layout_id=rename(layouts,space.layout_id,"layout");
  };
  auto indexing=[&](analysis::IndexingMap& map) {
    for(auto& index:map.results)
      index.binding_source=rename(tensors,index.binding_source,"tensor");
  };
  auto predicates=[&](std::vector<analysis::IndexResult>& values) {
    for(auto& index:values)
      index.binding_source=rename(tensors,index.binding_source,"tensor");
  };
  auto effect=[&](analysis::MemoryEffect& memory) {
    memory.alias_set=rename(aliases,memory.alias_set,"alias");
    memory.state_object=rename(states,memory.state_object,"state");
  };
  op.name="op";tensor(op.result);indexing(op.result_map);effect(op.result_effect);
  if(op.exact_task_access) {tensor(op.task_space);indexing(op.task_map);}
  for(auto& dim:op.domain)dim.binding_source=rename(tensors,dim.binding_source,"tensor");
  for(auto& operand:op.operands) {
    operand.producer=rename(producers,operand.producer,"producer");
    tensor(operand.tensor);indexing(operand.map);effect(operand.effect);
  }
  for(auto& read:op.element_reads) {
    tensor(read.tensor);indexing(read.map);predicates(read.nonnegative);
  }
  for(auto& write:op.additional_writes) {
    tensor(write.tensor);indexing(write.map);predicates(write.nonnegative);effect(write.effect);
  }
  op.reduction.partial_tensor=rename(tensors,op.reduction.partial_tensor,"tensor");
  if(!op.reduction.combiner.empty())op.reduction.combiner="combiner";
  return op.Serialize();
}
} // namespace tilemega::solver
