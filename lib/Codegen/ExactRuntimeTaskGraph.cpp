// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/ListScheduler.h>
#include <numeric>
#include <stdexcept>

namespace tilemega::codegen {
RuntimeTaskGraph MaterializeExactRuntimeTaskGraph(std::vector<int> const& counts,
    RuntimeExactDependencyDesc const& description,int seq,int past,int workers) {
  analysis::IslContext context;
  analysis::IslReferenceAudit audit(__func__);
  if (!description.tasks || !description.dependencies || !description.seq_parameter ||
      !description.past_parameter || seq<=0 || past<0 ||
      std::string(description.seq_parameter).empty() || std::string(description.past_parameter).empty() ||
      std::string(description.seq_parameter)==description.past_parameter)
    throw std::invalid_argument("incomplete exact runtime dependency descriptor");
  auto graph=MaterializeRuntimeTaskGraph(counts,{},workers);
  analysis::ParamBinding theta;
  theta.Bind(description.seq_parameter,seq).Bind(description.past_parameter,past);
  auto coordinate=[&](std::vector<long> const& point) {
    if (point.size()!=2 || point[0]<0 || point[0]>=long(counts.size()) ||
        point[1]<0 || point[1]>=counts[point[0]])
      throw std::invalid_argument("exact runtime relation escapes physical task ownership");
    return graph.stage_offsets[point[0]]+int(point[1]);
  };
  std::vector<bool> present(graph.successors.size());
  for (auto const& [domain,point]:analysis::CouplingRelation::FromIslText(description.tasks).BindParams(theta).Points()) {
    int id=coordinate(point);
    if (!domain.empty() || present[id])
      throw std::invalid_argument("exact runtime task domain is not an identity set");
    present[id]=true;
  }
  for (bool found:present) if (!found)
    throw std::invalid_argument("runtime dimensions or task counts differ from generated fusion domain");
  for (auto const& [consumer,producer]:analysis::CouplingRelation::FromIslText(description.dependencies).BindParams(theta).Points()) {
    int p=coordinate(producer),c=coordinate(consumer);
    if (producer[0]>=consumer[0])
      throw std::invalid_argument("exact runtime dependency violates stage order");
    graph.successors[p].push_back(c);
  }
  std::vector<int> stage_order(graph.successors.size());
  std::iota(stage_order.begin(),stage_order.end(),0);
  solver::ListScheduler{}.Validate(graph.successors,stage_order);
  return graph;
}
}  // namespace tilemega::codegen
