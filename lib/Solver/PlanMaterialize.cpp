// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/PlanMaterialize.h>

#include <algorithm>
#include <numeric>

#include <tilemega/Solver/BalancedPlacement.h>
#include <tilemega/Solver/ListScheduler.h>

namespace tilemega::solver {
namespace {

using dialect::PlacementMode;

int StageOfNode(codegen::RuntimeTaskGraph const& graph, int node) {
  return static_cast<int>(
      std::upper_bound(graph.stage_offsets.begin(), graph.stage_offsets.end(),
                       node) - graph.stage_offsets.begin() - 1);
}

/// sigma for every mode whose queues stay stage-major: walk the stage order,
/// then the tasks of each stage, handing out the next free slot on the owner.
/// This is exactly the order the pre-plan host materialized in, which is what
/// makes H2 and H3 byte identities hold by construction rather than by test.
void NumberStageMajor(PlanRequest const& request, MaterializedPlan* out) {
  std::vector<int> next(request.grid, 0);
  for (std::uint32_t stage : request.stage_order)
    for (int task = 0; task < request.counts[stage]; ++task)
      out->slot[stage][task] = next[out->owner[stage][task]]++;
}

}  // namespace

bool MaterializePlanPlacement(PlanRequest const& request, MaterializedPlan* out,
                              std::string* error) {
  auto fail = [&](std::string message) {
    if (error) *error = std::move(message);
    return false;
  };
  if (request.grid <= 0) return fail("plan materialization needs a positive grid");
  if (request.physical_worker.size() != static_cast<std::size_t>(request.grid))
    return fail("the physical worker map does not cover the grid");
  if (request.params.size() != dialect::PlacementModeParamCount(request.mode))
    return fail(std::string("placement mode ") +
                dialect::PlacementModeName(request.mode) +
                " got the wrong parameter count");

  std::size_t const stages = request.counts.size();
  out->owner.assign(stages, {});
  out->slot.assign(stages, {});
  for (std::size_t stage = 0; stage < stages; ++stage) {
    out->owner[stage].assign(request.counts[stage], 0);
    out->slot[stage].assign(request.counts[stage], 0);
  }

  switch (request.mode) {
    case PlacementMode::kLegacyGridStride:
      for (std::size_t stage = 0; stage < stages; ++stage)
        for (int task = 0; task < request.counts[stage]; ++task)
          out->owner[stage][task] = request.physical_worker[task % request.grid];
      break;

    case PlacementMode::kRotate: {
      // base[stage] is the prefix sum of the active task counts along the
      // stage order, mod grid, so stage s starts where s - 1 stopped.
      std::vector<int> base(stages, 0);
      long long running = 0;
      for (std::uint32_t stage : request.stage_order) {
        base[stage] = static_cast<int>(running % request.grid);
        running += request.counts[stage];
      }
      for (std::size_t stage = 0; stage < stages; ++stage)
        for (int task = 0; task < request.counts[stage]; ++task)
          out->owner[stage][task] =
              request.physical_worker[(task + base[stage]) % request.grid];
      out->rotate_base = std::move(base);
      break;
    }

    case PlacementMode::kBalanced: {
      if (!request.graph) return fail("balanced placement needs the task DAG");
      auto const order = ListScheduler{}.Schedule(request.graph->successors);
      auto const placed = BalanceTaskPlacement(
          request.graph->successors, order, request.graph->preferred_worker,
          request.grid, request.graph->baseline_max_queue);
      for (std::size_t stage = 0; stage < stages; ++stage)
        for (int task = 0; task < request.counts[stage]; ++task)
          out->owner[stage][task] =
              placed.worker[request.graph->stage_offsets[stage] + task];
      out->balanced = {placed.max_queue, request.graph->baseline_max_queue,
                       placed.same_worker_edges, placed.fence_free_producers};
      out->has_balanced_stats = true;
      break;
    }

    case PlacementMode::kEft:
    case PlacementMode::kTemplate:
      return fail(std::string("placement mode ") +
                  dialect::PlacementModeName(request.mode) +
                  " is not materialized yet (EX-S2)");
  }

  NumberStageMajor(request, out);
  return BuildPlanQueues(request.counts, request.grid, out, error);
}

bool BuildPlanQueues(std::vector<int> const& counts, int grid,
                     MaterializedPlan* plan, std::string* error) {
  auto fail = [&](std::string message) {
    if (error) *error = std::move(message);
    return false;
  };
  plan->queue.assign(grid, {});
  std::vector<std::vector<int>> slot_seen(grid);
  for (std::size_t stage = 0; stage < counts.size(); ++stage)
    for (int task = 0; task < counts[stage]; ++task) {
      int const worker = plan->owner[stage][task];
      if (worker < 0 || worker >= grid)
        return fail("pi sent stage " + std::to_string(stage) + " task " +
                    std::to_string(task) + " to worker " +
                    std::to_string(worker) + ", outside the grid");
      plan->queue[worker].push_back(
          {static_cast<std::uint32_t>(stage), task});
      slot_seen[worker].push_back(plan->slot[stage][task]);
    }
  for (int worker = 0; worker < grid; ++worker) {
    auto& queue = plan->queue[worker];
    std::vector<std::size_t> order(queue.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return slot_seen[worker][a] < slot_seen[worker][b];
    });
    auto sorted_slots = slot_seen[worker];
    std::sort(sorted_slots.begin(), sorted_slots.end());
    for (std::size_t i = 0; i < sorted_slots.size(); ++i)
      if (sorted_slots[i] != static_cast<int>(i))
        return fail("sigma on worker " + std::to_string(worker) +
                    " is not a dense order: slot " +
                    std::to_string(sorted_slots[i]) + " where " +
                    std::to_string(i) + " was expected");
    std::vector<PlanQueueItem> ordered;
    ordered.reserve(queue.size());
    for (std::size_t index : order) ordered.push_back(queue[index]);
    queue = std::move(ordered);
  }
  return true;
}

bool CheckPlanLegality(codegen::RuntimeTaskGraph const& graph,
                       MaterializedPlan const& plan, std::string* error) {
  auto fail = [&](std::string message) {
    if (error) *error = std::move(message);
    return false;
  };
  std::size_t const nodes = graph.successors.size();
  auto owner_of = [&](int node) {
    int const stage = StageOfNode(graph, node);
    return plan.owner[stage][node - graph.stage_offsets[stage]];
  };
  auto slot_of = [&](int node) {
    int const stage = StageOfNode(graph, node);
    return plan.slot[stage][node - graph.stage_offsets[stage]];
  };

  // L-c first: a same-worker producer whose sigma is not strictly smaller would
  // otherwise be silently elided from the consumer's polls (§5.7.3, EX-E1).
  for (std::size_t producer = 0; producer < nodes; ++producer)
    for (int consumer : graph.successors[producer]) {
      int const p = static_cast<int>(producer);
      if (owner_of(p) != owner_of(consumer)) continue;
      if (slot_of(p) >= slot_of(consumer))
        return fail("L-c: worker " + std::to_string(owner_of(p)) +
                    " runs its producer at slot " + std::to_string(slot_of(p)) +
                    " but its consumer at slot " +
                    std::to_string(slot_of(consumer)));
    }

  // L-a on the union of task edges and the same-worker queue edges.  W = 1, so
  // a worker's window edges are exactly slot i -> slot i + 1.
  std::vector<int> indegree(nodes, 0);
  for (std::size_t producer = 0; producer < nodes; ++producer)
    for (int consumer : graph.successors[producer]) ++indegree[consumer];
  std::vector<int> node_of_slot;
  std::vector<int> queue_successor(nodes, -1);
  for (auto const& queue : plan.queue) {
    node_of_slot.clear();
    for (auto const& item : queue)
      node_of_slot.push_back(graph.stage_offsets[item.stage] + item.logical);
    for (std::size_t i = 0; i + 1 < node_of_slot.size(); ++i) {
      queue_successor[node_of_slot[i]] = node_of_slot[i + 1];
      ++indegree[node_of_slot[i + 1]];
    }
  }
  std::vector<int> ready;
  for (std::size_t node = 0; node < nodes; ++node)
    if (indegree[node] == 0) ready.push_back(static_cast<int>(node));
  std::size_t emitted = 0;
  while (!ready.empty()) {
    int const node = ready.back();
    ready.pop_back();
    ++emitted;
    auto relax = [&](int next) {
      if (--indegree[next] == 0) ready.push_back(next);
    };
    for (int consumer : graph.successors[node]) relax(consumer);
    if (queue_successor[node] >= 0) relax(queue_successor[node]);
  }
  if (emitted != nodes)
    return fail("L-a: the task edges and the queue order form a cycle over " +
                std::to_string(nodes - emitted) + " tasks");
  return true;
}

}  // namespace tilemega::solver
