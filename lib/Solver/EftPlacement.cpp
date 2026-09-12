// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/EftPlacement.h>

#include <tilemega/Solver/CoResidency.h>

#include <algorithm>
#include <queue>

namespace tilemega::solver {

bool ScheduleByEarliestFinish(EftRequest const& request, EftSchedule* out,
                              std::string* error) {
  auto fail = [&](std::string message) {
    if (error) *error = std::move(message);
    return false;
  };
  if (!request.graph) return fail("no runtime task graph");
  codegen::RuntimeTaskGraph const& graph = *request.graph;
  int const nodes =
      graph.stage_offsets.empty() ? 0 : graph.stage_offsets.back();
  if (static_cast<int>(graph.successors.size()) != nodes)
    return fail("the task graph has " + std::to_string(graph.successors.size()) +
                " adjacency rows for " + std::to_string(nodes) + " nodes");
  if (static_cast<int>(request.task_ns.size()) != nodes)
    return fail("task_ns has " + std::to_string(request.task_ns.size()) +
                " entries for " + std::to_string(nodes) + " nodes");
  if (!request.task_lanes.empty() &&
      static_cast<int>(request.task_lanes.size()) != nodes)
    return fail("task_lanes is neither empty nor one row per node");
  int const grid = request.grid;
  if (grid <= 0) return fail("earliest-finish scheduling needs a positive grid");

  std::vector<int> worker_sm = request.worker_sm;
  if (worker_sm.empty()) {
    int const sms = request.sms > 0 ? request.sms : grid;
    worker_sm.resize(grid);
    for (int w = 0; w < grid; ++w) worker_sm[w] = w % sms;
  }
  if (static_cast<int>(worker_sm.size()) != grid)
    return fail("worker_sm has " + std::to_string(worker_sm.size()) +
                " entries for a grid of " + std::to_string(grid));
  int sm_count = 0;
  for (int sm : worker_sm) {
    if (sm < 0) return fail("worker_sm names a negative SM");
    sm_count = std::max(sm_count, sm + 1);
  }
  if (request.ctas_per_sm > 0 &&
      static_cast<long>(sm_count) * request.ctas_per_sm < grid)
    return fail("a grid of " + std::to_string(grid) + " does not fit in " +
                std::to_string(sm_count) + " SMs at " +
                std::to_string(request.ctas_per_sm) + " CTAs each");
  std::vector<std::vector<int>> sm_members(sm_count);
  for (int w = 0; w < grid; ++w) sm_members[worker_sm[w]].push_back(w);

  out->worker.assign(nodes, -1);
  out->slot.assign(nodes, -1);
  out->start_ns.assign(nodes, 0.0);
  out->end_ns.assign(nodes, 0.0);
  out->makespan_ns = 0.0;
  if (nodes == 0) return true;

  std::vector<std::vector<int>> predecessors(nodes);
  std::vector<int> indegree(nodes, 0);
  for (int node = 0; node < nodes; ++node)
    for (int succ : graph.successors[node]) {
      if (succ < 0 || succ >= nodes)
        return fail("the task graph names successor " + std::to_string(succ) +
                    ", outside [0, " + std::to_string(nodes) + ")");
      predecessors[succ].push_back(node);
      ++indegree[succ];
    }

  // The price of publishing this node's event to a consumer elsewhere.  N is
  // the node's whole fan-out and R is one, both pessimistic: the scheduler has
  // to price an edge before pi exists, so it cannot know how many workers will
  // end up polling the row.  The simulator, which sees the finished plan, uses
  // the exact cross-worker fan-out instead -- the two therefore disagree on a
  // hop by at most `c1 * log2(1 + N)`, which is zero inside one standard error
  // on sm_89 (SIMULATOR/README.md).  Making the estimate exact would need a
  // per-node worker set over 34M edges at mha4 seq 512.
  std::vector<double> hop_cost(nodes, 0.0);
  for (int node = 0; node < nodes; ++node)
    if (!graph.successors[node].empty())
      hop_cost[node] =
          request.hop.Ns(static_cast<int>(graph.successors[node].size()), 1);

  std::vector<int> topo;
  topo.reserve(nodes);
  {
    std::vector<int> degree = indegree, stack;
    for (int node = 0; node < nodes; ++node)
      if (degree[node] == 0) stack.push_back(node);
    while (!stack.empty()) {
      int const node = stack.back();
      stack.pop_back();
      topo.push_back(node);
      for (int succ : graph.successors[node])
        if (--degree[succ] == 0) stack.push_back(succ);
    }
    if (static_cast<int>(topo.size()) != nodes)
      return fail("the task DAG has a cycle over " +
                  std::to_string(nodes - topo.size()) + " tasks");
  }

  // Upward rank in ns: node weight from `task_ns`, edge weight from the
  // measured hop curve.  Round one's `ListScheduler` ranked by edge-count
  // height instead, which cannot tell a 200 ns elementwise task from a 40 us
  // GEMM and so ranks the axis modes 0 and 5 already sit at the ends of.
  std::vector<double> rank(nodes, 0.0);
  for (int i = nodes - 1; i >= 0; --i) {
    int const node = topo[i];
    double tail = 0.0;
    for (int succ : graph.successors[node])
      tail = std::max(tail, hop_cost[node] + rank[succ]);
    rank[node] = request.task_ns[node] + tail;
  }

  struct Ready {
    double rank;
    int node;
    /// Larger rank first; the node id breaks ties so the schedule is a
    /// function of the request and not of heap order.
    bool operator<(Ready const& other) const {
      return rank != other.rank ? rank < other.rank : node > other.node;
    }
  };
  std::priority_queue<Ready> ready;
  std::vector<int> degree = indegree;
  for (int node = 0; node < nodes; ++node)
    if (degree[node] == 0) ready.push({rank[node], node});

  std::vector<ResourceVector> const& lanes = request.task_lanes;
  bool const use_lanes = !lanes.empty();
  std::vector<double> worker_free(grid, 0.0);
  std::vector<int> last_of_worker(grid, -1), next_slot(grid, 0);
  std::vector<int> members;

  int placed = 0;
  while (!ready.empty()) {
    int const node = ready.top().node;
    ready.pop();

    // A predecessor's result reaches a worker at `end`, plus a hop unless that
    // worker is the one that produced it.  Reducing the scan to (best, second
    // best by owner) is what keeps a kAll fan-in of 256 from costing
    // 256 * grid: for a candidate w, the largest arrival among predecessors
    // *not* on w is `best` unless w owns `best`, in which case it is `second`.
    // A same-worker predecessor needs no arrival term of its own: starts are
    // non-decreasing per worker and W = 1, so `worker_free[w]` is already the
    // end of every task placed on w, that predecessor included.
    double best = 0.0, second = 0.0;
    int best_owner = -1;
    for (int pred : predecessors[node]) {
      int const owner = out->worker[pred];
      double const arrival = out->end_ns[pred] + hop_cost[pred];
      if (arrival > best) {
        // `second` becomes the old best, whose owner differs from the new one
        // and which dominates every arrival seen so far, so it is exactly the
        // largest arrival not produced by the new `best_owner`.
        if (owner != best_owner) {
          second = best;
          best_owner = owner;
        }
        best = arrival;
      } else if (owner != best_owner && arrival > second) {
        second = arrival;
      }
    }

    int chosen = -1;
    double chosen_start = 0.0, chosen_end = 0.0;
    for (int w = 0; w < grid; ++w) {
      double const arrival = best_owner == w ? second : best;
      double const start = std::max(worker_free[w], arrival);

      // Co-residency is charged against the siblings still running at `start`,
      // which is a causal approximation: a sibling task placed later can
      // overlap this one and is not visible here.  The simulator, not this
      // greedy, is the arbiter of the resulting makespan, and the two disagree
      // exactly where this approximation does.
      members.clear();
      members.push_back(node);
      for (int sibling : sm_members[worker_sm[w]]) {
        if (sibling == w) continue;
        int const last = last_of_worker[sibling];
        if (last >= 0 && out->end_ns[last] > start) members.push_back(last);
      }
      double const stretch =
          members.size() == 1
              ? 1.0
              : std::max(1.0, use_lanes ? LaneStretch(lanes, members)
                                        : static_cast<double>(members.size()));
      double const finish = start + request.task_ns[node] * stretch;
      if (chosen < 0 || finish < chosen_end) {
        chosen = w;
        chosen_start = start;
        chosen_end = finish;
      }
    }

    out->worker[node] = chosen;
    out->slot[node] = next_slot[chosen]++;
    out->start_ns[node] = chosen_start;
    out->end_ns[node] = chosen_end;
    worker_free[chosen] = chosen_end;
    last_of_worker[chosen] = node;
    out->makespan_ns = std::max(out->makespan_ns, chosen_end);
    ++placed;

    for (int succ : graph.successors[node])
      if (--degree[succ] == 0) ready.push({rank[succ], succ});
  }
  if (placed != nodes)
    return fail("placed " + std::to_string(placed) + " of " +
                std::to_string(nodes) + " tasks");

  // sigma is the order tasks were handed to a worker, which is already its
  // start-time order: a task starts no earlier than `worker_free`, so starts
  // are non-decreasing per worker.  It is also why L-c holds by construction --
  // the ready list is topological, so a same-worker predecessor took an earlier
  // slot -- and the caller still runs `CheckPlanLegality` rather than trust it.
  return true;
}

}  // namespace tilemega::solver
