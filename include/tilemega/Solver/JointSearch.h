// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/ExecutionSimulator.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace tilemega::solver {
/// Opt-in L2 outer search. ChainDP remains the independent L1 initializer.
/// Bounds must be admissible; priority is only a heuristic and never a bound.
struct JointCandidate {
  GemmConfig config;
  int kappa = 1, ctas_per_sm = 1;
  double work_lb_ns = 0, cp_lb_ns = 0, priority_ns = 0;
  std::string key;
  double queue_lb_lb_ns = 0;
};
struct JointEvaluation {
  JointCandidate candidate;
  std::string placement, status;
  double makespan_ns = 0, floor_ns = 0;
  bool simulated = false;
};
struct JointSearchStats {
  std::size_t considered = 0, pruned = 0, evaluated = 0, capacity_deferred = 0;
};
/// A callback materializes the candidate's own CG, legal resident Plan and
/// event grouping. The binding bound is the selection objective; simulation
/// breaks bound ties and selects candidates for measurement. It returns all
/// inner placement evaluations, including
/// rejected ones. A finite capacity is an explicit degraded search, not proof
/// that the unevaluated configurations are dominated.
inline std::vector<JointEvaluation> SearchL2Configurations(
    std::vector<JointCandidate> candidates, std::size_t capacity,
    std::function<std::vector<JointEvaluation>(JointCandidate const&)> evaluate,
    JointSearchStats* stats) {
  if (!stats || !capacity) throw std::invalid_argument("joint search needs capacity and stats");
  *stats = {};
  for (auto const& c : candidates)
    if(c.kappa<=0 || c.ctas_per_sm<=0 || !std::isfinite(c.work_lb_ns) ||
       !std::isfinite(c.cp_lb_ns) || !std::isfinite(c.queue_lb_lb_ns) || !std::isfinite(c.priority_ns) ||
       c.work_lb_ns<0 || c.cp_lb_ns<0 || c.queue_lb_lb_ns<0)
      throw std::invalid_argument("invalid joint candidate");
  std::stable_sort(candidates.begin(),candidates.end(),[](auto const&a,auto const&b){
    return a.priority_ns < b.priority_ns;
  });
  double incumbent=std::numeric_limits<double>::infinity();
  std::vector<JointEvaluation> results;
  for(auto const& c:candidates) {
    ++stats->considered;
    if(std::max({c.work_lb_ns,c.cp_lb_ns,c.queue_lb_lb_ns})>incumbent) {++stats->pruned;continue;}
    if(stats->evaluated==capacity) {++stats->capacity_deferred;continue;}
    auto inner=evaluate(c);++stats->evaluated;
    for(auto& r:inner) {
      if (r.candidate.key.empty()) r.candidate=c;
      if(r.status=="ok" && r.simulated) {
        if (!std::isfinite(r.floor_ns) || r.floor_ns<0 || !std::isfinite(r.makespan_ns))
          throw std::invalid_argument("invalid binding-bound evaluation");
        incumbent=std::min(incumbent,r.floor_ns);
      }
      results.push_back(std::move(r));
    }
  }
  std::stable_sort(results.begin(),results.end(),[](auto const&a,auto const&b){
    bool av=a.status=="ok"&&a.simulated,bv=b.status=="ok"&&b.simulated;
    if (av!=bv) return av;
    if (a.floor_ns!=b.floor_ns) return a.floor_ns<b.floor_ns;
    return a.makespan_ns<b.makespan_ns;
  });
  return results;
}
} // namespace tilemega::solver
