// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CouplingCache.h>
#include <sstream>
#include <stdexcept>

namespace tilemega::analysis {
std::string SemanticSignature(SemanticOp const& original) {
  SemanticOp op=original;
  std::map<std::string,std::string> tensors,producers,aliases,states;
  auto rename=[](std::map<std::string,std::string>& names,std::string const& name,char const* prefix) {
    if(name.empty())return name;
    auto it=names.find(name);if(it!=names.end())return it->second;
    return names.emplace(name,std::string(prefix)+std::to_string(names.size())).first->second;
  };
  auto tensor=[&](TensorSpace& t) {t.name=rename(tensors,t.name,"tensor");};
  auto effect=[&](MemoryEffect& e) {
    e.alias_set=rename(aliases,e.alias_set,"alias");e.state_object=rename(states,e.state_object,"state");
  };
  op.name="op";tensor(op.result);effect(op.result_effect);
  for(auto& operand:op.operands) {
    operand.producer=rename(producers,operand.producer,"producer");
    tensor(operand.tensor);effect(operand.effect);
  }
  for(auto& read:op.element_reads)tensor(read.tensor);
  op.reduction.partial_tensor=rename(tensors,op.reduction.partial_tensor,"tensor");
  if(!op.reduction.combiner.empty())op.reduction.combiner="combiner";
  return op.Serialize();
}
namespace {
std::pair<SemanticOp const*,std::string> semanticOf(SemanticGraph const& graph,std::string name) {
  if(auto* op=graph.Find(name))return {op,""};
  for(std::size_t at=name.rfind('.');at!=std::string::npos;at=name.rfind('.',at-1)) {
    if(auto* op=graph.Find(name.substr(0,at)))return {op,name.substr(at)};
    if(at==0)break;
  }
  throw std::invalid_argument("instantiated task has no semantic origin: "+name);
}
std::string geometry(OperatorNode const& node,SemanticOp const& op,Granularity const& g,ParamBinding const& known) {
  std::ostringstream out;
  if(auto it=g.tiles.find(op.name);it!=g.tiles.end())
    for(auto const& [dim,value]:it->second)out<<dim<<'='<<value.Substitute(known).ToString()<<';';
  if(auto it=g.reduction_chunk.find(op.name);it!=g.reduction_chunk.end())out<<"split="<<it->second.Substitute(known).ToString()<<';';
  for(auto const& value:node.tile)out<<value.Substitute(known).ToString()<<',';
  return out.str();
}
}
std::shared_ptr<OraclePair> CouplingCache::OracleFor(std::string const& relation) {
  auto it=oracle_entries.find(relation);
  if(it==oracle_entries.end())it=oracle_entries.emplace(relation,std::make_shared<OraclePair>(relation)).first;
  return it->second;
}
std::vector<CouplingEdge> CouplingCache::Derive(SemanticGraph const& semantics,
    OperatorGraph const& tasks,Granularity const& granularity,ParamBinding const& known) {
  std::vector<CouplingEdge> edges;
  for(auto const& consumer:tasks.nodes)for(std::size_t k=0;k<consumer.operands.size();++k) {
    auto* producer=tasks.Find(consumer.operands[k].producer);if(!producer)continue;
    auto [ps,pr]=semanticOf(semantics,producer->name);auto [cs,cr]=semanticOf(semantics,consumer.name);
    Key key{SemanticSignature(*ps)+pr,SemanticSignature(*cs)+cr,k,
            geometry(*producer,*ps,granularity,known),geometry(consumer,*cs,granularity,known)};
    auto it=entries.find(key);
    if(it==entries.end()) {
      ++misses;
      OperatorGraph pair;pair.nodes={*producer,consumer};
      pair.nodes[0].operands.clear();
      for(std::size_t j=0;j<pair.nodes[1].operands.size();++j)
        if(j!=k)pair.nodes[1].operands[j].producer.clear();
      auto derived=CouplingDerivation{}.Derive(pair,known);
      if(derived.size()!=1)throw std::logic_error("cached pair must derive exactly one coupling");
      auto value=std::move(derived.front());value.src.name.clear();value.dst.name.clear();
      it=entries.emplace(std::move(key),Value{std::move(value),{}}).first;
    } else ++hits;
    if(!it->second.oracle)it->second.oracle=OracleFor(it->second.edge.C.Reverse().ToString());
    auto edge=it->second.edge;edge.src.name=producer->name;edge.dst.name=consumer.name;
    edges.push_back(std::move(edge));
  }
  return edges;
}
}
