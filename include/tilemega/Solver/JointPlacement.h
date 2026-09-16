// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/EftPlacement.h>
#include <tilemega/Solver/ChainPlacement.h>
#include <tilemega/Solver/ExecutionSimulator.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace tilemega::solver {
struct PlacementEvaluation {
  std::string name,error;
  dialect::PlacementMode mode=dialect::PlacementMode::kLegacyGridStride;
  std::vector<std::int64_t> params;
  MaterializedPlan plan;
  PlanBounds bounds;
  double predicted_ns=0;
};
/// Every geometry invokes the entire placement catalog. Legality failures
/// remain individual candidate results, and never become an admissible plan.
inline std::vector<PlacementEvaluation> SolvePlacementCatalog(
    SimulatorInput const& input,PlanRequest request,SimulatorOptions const& options,
    HopCurve const& hop) {
  std::vector<PlacementEvaluation> results(6);
  results[0].name="legacy_grid_stride";
  results[1].name="rotate";results[1].mode=dialect::PlacementMode::kRotate;
  results[2].name="balanced";results[2].mode=dialect::PlacementMode::kBalanced;
  results[3].name="eft";results[3].mode=dialect::PlacementMode::kEft;
  results[4].name="wavefront";results[4].mode=dialect::PlacementMode::kTemplate;results[4].params={1};
  results[5].name="chain";results[5].mode=dialect::PlacementMode::kEft;
  PreparedPlanBounds prepared;std::string error;
  if (!PreparePlanBounds(input,&prepared,&error)) throw std::invalid_argument(error);
  auto priced=input;priced.prepared_graph=&prepared.graph;
  for (auto& result:results) {
    request.mode=result.mode;request.params=result.params;
    request.eft_worker.clear();request.eft_slot.clear();
    if (result.mode==dialect::PlacementMode::kEft) {
      EftRequest e;e.graph=input.graph;e.task_ns=input.task_ns;e.task_lanes=input.task_lanes;
      e.grid=request.grid;e.sms=request.grid;e.ctas_per_sm=1;e.hop=hop;
      if (result.name=="eft") {
        EftSchedule schedule;
        if (!ScheduleByEarliestFinish(e,&schedule,&result.error)) continue;
        request.eft_worker=std::move(schedule.worker);request.eft_slot=std::move(schedule.slot);
      } else {
        ChainRequest c;c.graph=input.graph;c.task_ns=input.task_ns;c.grid=request.grid;
        c.sms=request.grid;c.ctas_per_sm=1;c.hop=hop;c.cost_aware_extend=true;
        ChainSchedule schedule;
        if (!ScheduleByCriticalChain(c,&schedule,&result.error)) continue;
        request.eft_worker=std::move(schedule.worker);request.eft_slot=std::move(schedule.slot);
      }
    }
    if (!MaterializePlanPlacement(request,&result.plan,&result.error) ||
        !CheckPlanLegality(*input.graph,result.plan,&result.error) ||
        !EvaluatePlanBounds(prepared,result.plan,&result.bounds,&result.error)) continue;
    SimulatorResult sim;
    if (!SimulateExecution(priced,result.plan,options,hop,&sim,&result.error)) continue;
    result.predicted_ns=sim.makespan_ns;
  }
  std::stable_sort(results.begin(),results.end(),[](auto const& a,auto const& b) {
    if (a.error.empty()!=b.error.empty()) return a.error.empty();
    if (a.bounds.lower_bound_ns!=b.bounds.lower_bound_ns) return a.bounds.lower_bound_ns<b.bounds.lower_bound_ns;
    return a.predicted_ns<b.predicted_ns;
  });
  if (!results.front().error.empty()) throw std::runtime_error("all placement candidates failed legality");
  return results;
}
} // namespace tilemega::solver
