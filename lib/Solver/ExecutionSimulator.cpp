// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ExecutionSimulator.h>

#include <tilemega/Solver/CoResidency.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <queue>
#include <sstream>

namespace tilemega::solver {
namespace {

int StageOfNode(codegen::RuntimeTaskGraph const& graph, int node) {
  auto it = std::upper_bound(graph.stage_offsets.begin(),
                             graph.stage_offsets.end(), node);
  return static_cast<int>(it - graph.stage_offsets.begin()) - 1;
}

}  // namespace

double HopCurve::Ns(int consumers, int rows) const {
  double const n = std::max(1, consumers), r = std::max(1, rows);
  return c0 + c1 * std::log2(1.0 + n / r) + c2 * std::log2(r);
}

bool HopCurve::FromTsv(std::string const& path, HopCurve* out, std::string* error) {
  std::ifstream in(path);
  if (!in) {
    if (error) *error = "cannot read " + path;
    return false;
  }
  bool seen[3] = {false, false, false};
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::string key, value;
    if (!std::getline(fields, key, '\t') || !std::getline(fields, value, '\t'))
      continue;
    if (key == "c0") { out->c0 = std::stod(value); seen[0] = true; }
    else if (key == "c1") { out->c1 = std::stod(value); seen[1] = true; }
    else if (key == "c2") { out->c2 = std::stod(value); seen[2] = true; }
  }
  if (!(seen[0] && seen[1] && seen[2])) {
    if (error) *error = path + " does not carry c0, c1 and c2";
    return false;
  }
  return true;
}

bool SimulateExecution(SimulatorInput const& input, MaterializedPlan const& plan,
                       SimulatorOptions const& options, HopCurve const& hop,
                       SimulatorResult* out, std::string* error) {
  auto fail = [&](std::string message) {
    if (error) *error = std::move(message);
    return false;
  };
  if (!input.graph) return fail("no runtime task graph");
  codegen::RuntimeTaskGraph const& graph = *input.graph;
  if (options.window != 1)
    return fail("window W = " + std::to_string(options.window) +
                " is out of scope this round (EX-E2); only W = 1 is simulated");

  int const nodes = graph.stage_offsets.empty()
                        ? 0 : graph.stage_offsets.back();
  if (static_cast<int>(input.task_ns.size()) != nodes)
    return fail("task_ns has " + std::to_string(input.task_ns.size()) +
                " entries for " + std::to_string(nodes) + " nodes");
  if (!input.task_lanes.empty() &&
      static_cast<int>(input.task_lanes.size()) != nodes)
    return fail("task_lanes is neither empty nor one row per node");
  int const grid = static_cast<int>(plan.queue.size());
  if (grid <= 0) return fail("the plan has no workers");

  // Worker -> SM.  `sms == 0` gives every worker its own SM, which makes the
  // co-residency model inert; that is the control, not the default.
  std::vector<int> worker_sm = input.worker_sm;
  if (worker_sm.empty()) {
    worker_sm.resize(grid);
    int const sms = options.sms > 0 ? options.sms : grid;
    for (int w = 0; w < grid; ++w) worker_sm[w] = w % sms;
  }
  if (static_cast<int>(worker_sm.size()) != grid)
    return fail("worker_sm has " + std::to_string(worker_sm.size()) +
                " entries for a grid of " + std::to_string(grid));
  int sm_count = 0;
  for (int sm : worker_sm) sm_count = std::max(sm_count, sm + 1);

  // Flatten the plan's queues onto node ids once; everything below indexes by
  // node, so sigma never has to be consulted again.
  std::vector<int> owner_of(nodes, -1), position(nodes, -1);
  std::vector<std::vector<int>> queue(grid);
  for (int w = 0; w < grid; ++w) {
    queue[w].reserve(plan.queue[w].size());
    for (std::size_t i = 0; i < plan.queue[w].size(); ++i) {
      PlanQueueItem const& item = plan.queue[w][i];
      int const stage = static_cast<int>(item.stage);
      if (stage < 0 || stage + 1 >= static_cast<int>(graph.stage_offsets.size()))
        return fail("the plan names stage " + std::to_string(stage) +
                    ", which the task graph does not have");
      int const node = graph.stage_offsets[stage] + item.logical;
      if (node < graph.stage_offsets[stage] ||
          node >= graph.stage_offsets[stage + 1])
        return fail("the plan names task " + std::to_string(item.logical) +
                    " of stage " + std::to_string(stage) + ", out of range");
      if (owner_of[node] != -1)
        return fail("stage " + std::to_string(stage) + " task " +
                    std::to_string(item.logical) + " is queued twice");
      owner_of[node] = w;
      position[node] = static_cast<int>(i);
      queue[w].push_back(node);
    }
  }
  for (int node = 0; node < nodes; ++node)
    if (owner_of[node] == -1)
      return fail("stage " + std::to_string(StageOfNode(graph, node)) +
                  " task " + std::to_string(node - graph.stage_offsets[StageOfNode(graph, node)]) +
                  " is in no worker's queue");

  std::vector<int> unmet(nodes, 0);
  std::vector<int> cross_fanout(nodes, 0);
  long cross_edges = 0, same_edges = 0;
  // Counted branchlessly: on a plan like mode 5 almost every edge is cross, on
  // mode 0 the mix is uneven, and a mispredicted branch per edge is a third of
  // this sweep on a graph with half a million of them.
  for (int node = 0; node < nodes; ++node) {
    int const mine = owner_of[node];
    long same = 0;
    for (int succ : graph.successors[node]) {
      ++unmet[succ];
      same += owner_of[succ] == mine;
    }
    long const total = static_cast<long>(graph.successors[node].size());
    same_edges += same;
    cross_edges += total - same;
    cross_fanout[node] = static_cast<int>(total - same);
  }

  out->tasks.assign(nodes, SimulatedTask{});
  std::vector<double> ready(nodes, 0.0);
  std::vector<int> head(grid, 0);
  std::vector<double> free_at(grid, 0.0);
  // W = 1 means a worker runs one task at a time, and `free_at` alone cannot
  // enforce that: it lags a task's whole duration behind, so a foreign
  // completion arriving mid-task would find the head ready and start it
  // alongside the running one.  The queue lower bound then exceeds the
  // makespan, which is how this was caught.
  std::vector<char> busy(grid, 0);
  std::vector<std::vector<int>> in_flight(sm_count);
  for (auto& set : in_flight) set.reserve(std::max(1, options.ctas_per_sm));

  // Progress is rate-based, not fixed at start: when an SM's in-flight set
  // changes, every member still running on it is re-priced from that instant.
  // Pricing a task once, from the set it happened to see when it started, would
  // make the makespan depend on which co-resident task started first -- an
  // arbitrary tie-break deciding a number the solver ranks plans by.
  std::vector<double> remaining(nodes, 0.0), rate(nodes, 0.0), updated(nodes, 0.0);
  std::vector<int> version(nodes, 0);

  // Two event kinds in one heap.  A completion is obvious; a wake exists
  // because a head can become ready at a time strictly in the future -- the
  // predecessor has finished but its hop has not landed -- and nothing else
  // would ever look at that worker again.  Dropping wakes is the one way this
  // loop can silently stall a worker forever, so it is not an optimization.
  struct Event {
    double time;
    int target;        ///< completing node, or the worker to re-examine
    int version;       ///< 0 for a wake; a completion is stale if this is old
    bool completion;
    bool operator<(Event const& other) const { return time > other.time; }
  };
  std::priority_queue<Event> pending;
  int in_flight_total = 0;

  std::vector<ResourceVector> const& lanes = input.task_lanes;
  bool const use_lanes = !lanes.empty() && !options.proportional_sharing;

  // Drain every member of `sm` up to `now`, re-price the set, and re-publish
  // each member's completion.  Stale heap entries are left to be skipped.
  auto refresh = [&](int sm, double now) {
    auto& set = in_flight[sm];
    if (set.empty()) return;
    for (int m : set) {
      remaining[m] = std::max(0.0, remaining[m] - (now - updated[m]) * rate[m]);
      updated[m] = now;
    }
    double const stretch = std::max(1.0, use_lanes
        ? LaneStretch(lanes, set) : static_cast<double>(set.size()));
    for (int m : set) {
      rate[m] = 1.0 / stretch;
      pending.push({now + remaining[m] * stretch, m, ++version[m], true});
    }
  };

  auto try_start = [&](int w, double now) {
    if (busy[w]) return;
    if (head[w] >= static_cast<int>(queue[w].size())) return;
    int const node = queue[w][head[w]];
    if (unmet[node] != 0) return;
    double const start = std::max(free_at[w], ready[node]);
    if (start > now) { pending.push({start, w, 0, false}); return; }
    int const sm = worker_sm[w];
    SimulatedTask& task = out->tasks[node];
    task.worker = w;
    task.start_ns = start;
    task.block_ns = start - free_at[w];
    remaining[node] = input.task_ns[node];
    updated[node] = start;
    rate[node] = 0.0;
    ++head[w];
    busy[w] = 1;
    ++in_flight_total;
    in_flight[sm].push_back(node);
    refresh(sm, start);
  };

  double now = 0.0;
  std::vector<int> by_end;
  by_end.reserve(nodes);
  for (int w = 0; w < grid; ++w) try_start(w, now);
  int completed = 0;
  while (!pending.empty()) {
    Event const event = pending.top();
    pending.pop();
    now = std::max(now, event.time);
    if (!event.completion) {
      // A stale wake is harmless: try_start either starts the head, does
      // nothing, or re-pushes at a strictly later time, so time advances.
      try_start(event.target, now);
      continue;
    }
    int const node = event.target;
    if (event.version != version[node]) continue;  // re-priced since
    int const w = out->tasks[node].worker;
    int const sm = worker_sm[w];
    SimulatedTask& task = out->tasks[node];
    task.end_ns = now;
    task.stretch = input.task_ns[node] > 0.0
        ? (task.end_ns - task.start_ns) / input.task_ns[node] : 1.0;
    auto& set = in_flight[sm];
    set.erase(std::find(set.begin(), set.end(), node));
    --in_flight_total;
    free_at[w] = task.end_ns;
    busy[w] = 0;
    by_end.push_back(node);
    ++completed;
    refresh(sm, now);

    // The hop context: N is how many distinct workers poll this producer's
    // event, R the number of rows plausibly under contention at this instant,
    // taken as the tasks in flight device-wide.  Both coefficients are zero
    // inside one standard error on sm_89, so `flat_hop` exists to show that this
    // choice moves the makespan by less than the curve's own uncertainty.
    double const edge = cross_fanout[node] == 0
        ? 0.0
        : (options.flat_hop ? hop.c0
                            : hop.Ns(cross_fanout[node], std::max(1, in_flight_total)));
    for (int succ : graph.successors[node]) {
      double const arrival = task.end_ns + (owner_of[succ] == w ? 0.0 : edge);
      ready[succ] = std::max(ready[succ], arrival);
      if (--unmet[succ] == 0 && owner_of[succ] != w) try_start(owner_of[succ], now);
    }
    try_start(w, now);

  }
  if (completed != nodes) {
    // Nothing is in flight and some worker still has a queue, so no ready time
    // can ever advance: the queue order closed a cycle with the task edges.
    // That is an L-a violation in the plan under test, reported as one rather
    // than run out against a step budget.  The loop can also never have started
    // at all, which is the same cycle reaching back to slot 0.
    for (int v = 0; v < grid; ++v)
      if (head[v] < static_cast<int>(queue[v].size())) {
        int const stuck = queue[v][head[v]];
        int const stage = StageOfNode(graph, stuck);
        return fail("the queue order deadlocks: worker " + std::to_string(v) +
                    " is at stage " + std::to_string(stage) + " task " +
                    std::to_string(stuck - graph.stage_offsets[stage]) +
                    " with " + std::to_string(unmet[stuck]) +
                    " predecessors that can never run (L-a)");
      }
    return fail("simulated " + std::to_string(completed) + " of " +
                std::to_string(nodes) + " tasks");
  }


  out->makespan_ns = 0.0;
  out->total_work_ns = out->solo_work_ns = out->total_block_ns = 0.0;
  std::vector<double> worker_busy(grid, 0.0);
  for (int node = 0; node < nodes; ++node) {
    SimulatedTask const& task = out->tasks[node];
    out->makespan_ns = std::max(out->makespan_ns, task.end_ns);
    out->total_work_ns += task.end_ns - task.start_ns;
    out->solo_work_ns += input.task_ns[node];
    out->total_block_ns += task.block_ns;
    worker_busy[task.worker] += task.end_ns - task.start_ns;
  }
  out->busiest_worker = 0;
  for (int w = 0; w < grid; ++w)
    if (worker_busy[w] > worker_busy[out->busiest_worker]) out->busiest_worker = w;
  out->busiest_worker_ns = worker_busy[out->busiest_worker];
  out->cross_worker_edges = cross_edges;
  out->same_worker_edges = same_edges;

  // Longest end-to-end chain in the realized schedule, over task edges only:
  // the part of the makespan no placement can remove.  The sweep needs a
  // topological order and node ids are not guaranteed to be one, so it walks
  // `by_end`, the order the loop retired tasks in, which is topological by
  // construction -- a successor cannot end before its predecessor.
  std::vector<double> chain(nodes, 0.0);
  out->critical_path_ns = 0.0;
  for (int node : by_end) {
    chain[node] += out->tasks[node].end_ns - out->tasks[node].start_ns;
    out->critical_path_ns = std::max(out->critical_path_ns, chain[node]);
    for (int succ : graph.successors[node])
      chain[succ] = std::max(chain[succ], chain[node]);
  }
  return true;
}

}  // namespace tilemega::solver
