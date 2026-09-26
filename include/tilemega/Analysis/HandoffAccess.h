// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/FusionAccess.h>
namespace tilemega::analysis {
struct HandoffAccessProof {
  CouplingRelation consumer_to_producer;
  TaskAccesses composed;
};
// Availability is indexed by consumer task, just like its reads. The caller
// derives it from no-producer elements and the consumer's dependency closure.
HandoffAccessProof ProveRecompute(TaskAccesses const& producer,TaskAccesses const& consumer,
    std::set<std::string> const& intermediate,TaskAccesses const& available);
// Each collapse maps the partial tensor's elements to final output elements.
// Equality checks ensure complete reduction fibres, not bounding boxes.
HandoffAccessProof ProveLastArriver(TaskAccesses const& producer,TaskAccesses const& consumer,
    std::map<std::string,CouplingRelation> const& collapse,std::string const& output);
HandoffAccessProof ProveDirectHandoff(TaskAccesses const& producer,TaskAccesses const& consumer,
    std::set<std::string> const& intermediate,CouplingRelation const& producer_position,
    CouplingRelation const& consumer_position);
} // namespace tilemega::analysis
