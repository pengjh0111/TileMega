// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <map>
#include <tuple>

namespace tilemega::analysis {
/// Alpha-renames identities, retaining access maps, predicates and effects.
std::string SemanticSignature(SemanticOp const& op);
struct CouplingCache {
  using Key=std::tuple<std::string,std::string,std::size_t,std::string,std::string>;
  struct Value {
    CouplingEdge edge;
    // Oracle descriptions are immutable symbolic text, never concrete tile edges.
    std::string forward_oracle,reverse_oracle,structure;
  };
  std::map<Key,Value> entries;
  std::size_t hits=0,misses=0;
  std::vector<CouplingEdge> Derive(SemanticGraph const& semantics,
      OperatorGraph const& tasks,Granularity const& granularity,ParamBinding const& known);
};
}
