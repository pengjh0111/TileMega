// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <map>
#include <set>

#ifndef TILEMEGA_FUSION_ACCESS
#define TILEMEGA_FUSION_ACCESS 1
#endif

namespace tilemega::analysis {
struct TaskAccesses {
  std::map<std::string, CouplingRelation> reads, writes;
};
struct FusionAccesses {
  TaskAccesses task;
  CouplingRelation consumer_to_producer;
  QuasiPolynomial fanout, recompute_tasks, task_count;
  std::set<std::string> retained_intermediates;
};

// Exact L-task composition, indexed by consumer coordinates. A multi-tile
// producer footprint is a failed granularity constraint, not a free fusion.
// Callers must enumerate a compatible producer tile before pricing this pair.
FusionAccesses ComposeFusionAccesses(TaskAccesses const& producer,
    TaskAccesses const& consumer, std::set<std::string> const& internal_tensors,
    std::set<std::string> const& externally_read_tensors);
}  // namespace tilemega::analysis
