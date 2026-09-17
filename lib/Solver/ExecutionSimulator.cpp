// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ExecutionSimulator.h>

#include <tilemega/Solver/CoResidency.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <queue>
#include <sstream>
#include <unordered_map>

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

bool PrepareExecutionGraph(codegen::RuntimeTaskGraph const& graph,
    PreparedExecutionGraph* out,std::string* error) {
  if (graph.stage_offsets.empty() || graph.stage_offsets.back()<0 ||
      graph.successors.size()!=std::size_t(graph.stage_offsets.back())) {
    if (error) *error="prepared graph has inconsistent node counts";
    return false;
  }
  PreparedExecutionGraph result;
  result.source=&graph;
  result.forward_node_order=true;
  int source_node=0;
  std::unordered_map<std::size_t,std::vector<int>> buckets;
  for (auto const& row:graph.successors) {
    std::size_t hash=row.size();
    for (int succ:row) {
      result.forward_node_order &= succ>source_node;
      if (succ<0 || succ>=graph.stage_offsets.back()) {
        if (error) *error="prepared graph successor out of range";
        return false;
      }
      hash^=std::size_t(succ)+0x9e3779b9+(hash<<6)+(hash>>2);
    }
    int group=-1;
    for (int candidate:buckets[hash])
      if (result.successors[candidate]==row) {group=candidate;break;}
    if (group<0) {
      group=int(result.successors.size());
      result.successors.push_back(row);
      result.producer_count.push_back(0);result.producers.emplace_back();
      buckets[hash].push_back(group);
    }
    result.producers[group].push_back(int(result.group_of_node.size()));
    result.group_of_node.push_back(group);
    ++result.producer_count[group];
    ++source_node;
  }
  *out=std::move(result);
  return true;
}

bool PrepareExecutionPlan(PreparedExecutionGraph const& prepared,MaterializedPlan const& plan,
    PreparedExecutionPlan* out,std::string* error) {
  auto fail=[&](std::string message){if(error)*error=std::move(message);return false;};
  if(!prepared.source || prepared.source->stage_offsets.empty() || plan.queue.empty())
    return fail("readiness requires a prepared graph and worker queues");
  auto const& graph=*prepared.source;int nodes=graph.stage_offsets.back(),grid=plan.queue.size();
  // Flatten the plan's queues onto node ids once; everything below indexes by
  // node, so sigma never has to be consulted again.
  std::vector<int> owner_of(nodes, -1);
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
      queue[w].push_back(node);
    }
  }
  for (int node = 0; node < nodes; ++node)
    if (owner_of[node] == -1)
      return fail("stage " + std::to_string(StageOfNode(graph, node)) +
                  " task " + std::to_string(node - graph.stage_offsets[StageOfNode(graph, node)]) +
                  " is in no worker's queue");

  int const groups=int(prepared.successors.size());
  std::vector<int> unmet(nodes,0),cross_fanout(nodes,0);
  std::vector<unsigned char> cross_input(nodes,0);
  // Reuse graph membership and one worker histogram. Sorting owner lists for
  // each successor group was more expensive than the event recurrence itself.
  std::vector<int> consumer_counts(grid,0),touched;
  long cross_edges=0,same_edges=0;
  for (int g=0;g<groups;++g) {
    auto const& producers=prepared.producers[g];
    int first_owner=owner_of[producers.front()];bool mixed=false;
    for (int n:producers) mixed|=owner_of[n]!=first_owner;
    for (int succ:prepared.successors[g]) {
      ++unmet[succ];int w=owner_of[succ];
      if (!consumer_counts[w]++) touched.push_back(w);
      cross_input[succ]|=mixed || w!=first_owner;
    }
    for (int n:producers) {
      int same=consumer_counts[owner_of[n]];
      cross_fanout[n]=int(prepared.successors[g].size())-same;
      same_edges+=same;cross_edges+=cross_fanout[n];
    }
    for (int w:touched) consumer_counts[w]=0;
    touched.clear();
  }

  out->graph=&graph;out->plan=&plan;out->queue=std::move(queue);
  out->owner=std::move(owner_of);out->unmet=std::move(unmet);
  out->cross_fanout=std::move(cross_fanout);out->cross_input=std::move(cross_input);
  out->cross_edges=cross_edges;out->same_edges=same_edges;
  out->queue_next.assign(nodes,-1);
  out->forward_node_order=prepared.forward_node_order;
  for(auto const& row:out->queue)for(std::size_t i=1;i<row.size();++i) {
    out->queue_next[row[i-1]]=row[i];out->forward_node_order &= row[i]>row[i-1];
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

  PreparedExecutionGraph local_prepared;
  auto const* prepared=input.prepared_graph;
  if (!prepared) {
    if (!PrepareExecutionGraph(graph,&local_prepared,error)) return false;
    prepared=&local_prepared;
  }
  if (prepared->source!=&graph || prepared->group_of_node.size()!=std::size_t(nodes))
    return fail("prepared graph belongs to a different immutable graph");

  PreparedExecutionPlan local_plan;
  auto const* readiness=input.prepared_plan;
  if(!readiness) {
    if(!PrepareExecutionPlan(*prepared,plan,&local_plan,error))return false;
    readiness=&local_plan;
  }
  if(readiness->graph!=&graph || readiness->plan!=&plan || readiness->owner.size()!=std::size_t(nodes))
    return fail("prepared readiness belongs to a different immutable graph/plan");
  auto const& owner_of=readiness->owner;
  auto const& queue=readiness->queue;
  auto unmet=readiness->unmet;
  auto const& cross_fanout=readiness->cross_fanout;
  auto const& cross_input=readiness->cross_input;
  long cross_edges=readiness->cross_edges,same_edges=readiness->same_edges;
  int const groups=int(prepared->successors.size());
  struct GroupCompletion {
    int remaining;
    double cross_first=0,cross_second=0,chain=0,end_max=0;
    int first_owner=-1;
    void arrive(int owner,double end,double cross) {
      end_max=std::max(end_max,end);
      if (first_owner==owner) cross_first=std::max(cross_first,cross);
      else if (cross>=cross_first) {
        cross_second=cross_first;cross_first=cross;first_owner=owner;
      } else cross_second=std::max(cross_second,cross);
    }
    double ready_at(int owner) const {
      // A foreign producer's cross arrival is at least its end; therefore
      // the global end maximum plus the best foreign arrival also covers the
      // local-owner maximum, without a per-owner end-time table.
      return std::max(end_max,first_owner==owner ? cross_second : cross_first);
    }
  };
  std::vector<GroupCompletion> group_completion;
  group_completion.reserve(groups);
  for (int g=0;g<groups;++g) {
    group_completion.push_back({prepared->producer_count[g]});
  }
  std::vector<double> chain(nodes,0);
  double critical_path=0;

  if (!std::isfinite(options.publication_ns) || options.publication_ns < 0)
    return fail("publication cost must be finite and nonnegative");
  if (!input.publication_required.empty() && input.publication_required.size() != std::size_t(nodes))
    return fail("publication mask must have one entry per task");
  if (!std::isfinite(options.consumer_wait_ns) || options.consumer_wait_ns < 0)
    return fail("consumer wait cost must be finite and nonnegative");
  if (!input.consumer_wait_required.empty() && input.consumer_wait_required.size() != std::size_t(nodes))
    return fail("consumer wait mask must have one entry per task");
  std::vector<double> publication(nodes, 0.0), consumer_wait(nodes, 0.0);
  for (int node = 0; node < nodes; ++node) {
    bool const needed = input.publication_required.empty()
        ? cross_fanout[node] != 0 : input.publication_required[node] != 0;
    publication[node] = needed ? options.publication_ns : 0.0;
    bool const waits = input.consumer_wait_required.empty()
        ? cross_input[node] != 0 : input.consumer_wait_required[node] != 0;
    consumer_wait[node] = waits ? options.consumer_wait_ns : 0.0;
  }

  // With observed durations and a context-independent hop, event times are
  // exactly the weighted DAG recurrence. No resource rate can change in flight.
  if (options.observed_task_times && (options.flat_hop || (hop.c1==0 && hop.c2==0))) {
    if (!std::isfinite(hop.c0) || hop.c0<0) return fail("invalid flat hop cost");
    out->tasks.assign(nodes,SimulatedTask{});
    auto const& queue_next=readiness->queue_next;
    std::vector<int> ready_nodes;ready_nodes.reserve(nodes);
    std::vector<double> arrival(nodes,0),queue_end(nodes,0),worker_busy(grid,0);
    if(!readiness->forward_node_order)for(int next:queue_next)if(next>=0)++unmet[next];
    if(!readiness->forward_node_order)for (int n=0;n<nodes;++n) if (!unmet[n]) ready_nodes.push_back(n);
    std::size_t visited=0;
    out->makespan_ns=out->total_work_ns=out->solo_work_ns=out->total_block_ns=0;
    while (visited<(readiness->forward_node_order ? std::size_t(nodes) : ready_nodes.size())) {
      int n=readiness->forward_node_order ? int(visited) : ready_nodes[visited];++visited;
      int w=owner_of[n];
      auto& task=out->tasks[n];task.worker=w;
      task.start_ns=std::max(arrival[n],queue_end[n])+consumer_wait[n];
      task.end_ns=task.start_ns+input.task_ns[n];task.block_ns=task.start_ns-queue_end[n];
      task.stretch=1;
      out->makespan_ns=std::max(out->makespan_ns,task.end_ns+publication[n]);
      out->solo_work_ns+=input.task_ns[n];out->total_block_ns+=task.block_ns;
      worker_busy[w]+=input.task_ns[n];
      auto release=[&](int succ) {if (!readiness->forward_node_order && --unmet[succ]==0) ready_nodes.push_back(succ);};
      if (queue_next[n]>=0) {queue_end[queue_next[n]]=task.end_ns+publication[n];release(queue_next[n]);}
      auto& group=group_completion[prepared->group_of_node[n]];
      group.arrive(w,task.end_ns,task.end_ns+publication[n]+hop.c0);
      chain[n]+=input.task_ns[n];critical_path=std::max(critical_path,chain[n]);
      group.chain=std::max(group.chain,chain[n]);
      if (--group.remaining==0) for (int succ:prepared->successors[prepared->group_of_node[n]]) {
        arrival[succ]=std::max(arrival[succ],group.ready_at(owner_of[succ]));
        chain[succ]=std::max(chain[succ],group.chain);release(succ);
      }
    }
    if (visited!=std::size_t(nodes)) return fail("the queue order deadlocks (L-a)");
    out->total_work_ns=out->solo_work_ns;
    out->busiest_worker=int(std::max_element(worker_busy.begin(),worker_busy.end())-worker_busy.begin());
    out->busiest_worker_ns=worker_busy[out->busiest_worker];
    out->cross_worker_edges=cross_edges;out->same_worker_edges=same_edges;
    out->critical_path_ns=critical_path;
    return true;
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
    double const stretch = options.observed_task_times ? 1.0 : std::max(1.0, use_lanes
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
    double const start = std::max(free_at[w], ready[node]) + consumer_wait[node];
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
    free_at[w] = task.end_ns + publication[node];
    busy[w] = 0;
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
    if (!std::isfinite(edge) || edge<0) return fail("invalid contextual hop cost");
    auto& group=group_completion[prepared->group_of_node[node]];
    group.arrive(w,task.end_ns,task.end_ns+publication[node]+edge);
    chain[node]+=task.end_ns-task.start_ns;
    critical_path=std::max(critical_path,chain[node]);
    group.chain=std::max(group.chain,chain[node]);
    if (--group.remaining==0) {
      for (int succ:prepared->successors[prepared->group_of_node[node]]) {
        ready[succ]=std::max(ready[succ],group.ready_at(owner_of[succ]));
        chain[succ]=std::max(chain[succ],group.chain);
        if (--unmet[succ]==0 && owner_of[succ]!=w) try_start(owner_of[succ],now);
      }
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
    out->makespan_ns = std::max(out->makespan_ns, task.end_ns + publication[node]);
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

  out->critical_path_ns=critical_path;
  return true;
}

}  // namespace tilemega::solver

namespace tilemega::solver {

bool PreparePlanBounds(SimulatorInput const& input, PreparedPlanBounds* out,
                       std::string* error) {
  auto fail = [&](char const* message) {
    if (error) *error = message;
    return false;
  };
  if (!input.graph || input.graph->stage_offsets.empty())
    return fail("bounds require a runtime graph");
  auto const& graph = *input.graph;
  int const nodes = graph.stage_offsets.back();
  if (nodes < 0 || input.task_ns.size() != std::size_t(nodes) ||
      graph.successors.size() != std::size_t(nodes))
    return fail("bounds graph/cost size mismatch");
  std::vector<int> degree(nodes, 0), ready;
  for (auto const& row : graph.successors)
    for (int next : row) {
      if (next < 0 || next >= nodes) return fail("bounds invalid successor");
      ++degree[next];
    }
  std::vector<double> end(nodes, 0);
  double work = 0, cp = 0;
  for (int n = 0; n < nodes; ++n) {
    if (!std::isfinite(input.task_ns[n]) || input.task_ns[n] < 0)
      return fail("bounds require finite nonnegative node costs");
    work += input.task_ns[n];
    if (!degree[n]) ready.push_back(n);
  }
  std::size_t visited = 0;
  while (visited < ready.size()) {
    int const node = ready[visited++];
    end[node] += input.task_ns[node];
    cp = std::max(cp, end[node]);
    for (int next : graph.successors[node]) {
      end[next] = std::max(end[next], end[node]);
      if (--degree[next] == 0) ready.push_back(next);
    }
  }
  if (visited != std::size_t(nodes)) return fail("bounds task graph is cyclic");
  *out = {graph.stage_offsets, input.task_ns, work, cp};
  return PrepareExecutionGraph(graph,&out->graph,error);
}

bool EvaluatePlanBounds(PreparedPlanBounds const& input,
                        MaterializedPlan const& plan, PlanBounds* out,
                        std::string* error) {
  auto fail = [&](char const* message) {
    if (error) *error = message;
    return false;
  };
  if (plan.queue.empty()) return fail("bounds plan has no workers");
  std::vector<unsigned char> seen(input.task_ns.size(), 0);
  std::vector<int> queue_next(input.task_ns.size(),-1),degree(input.task_ns.size(),0);
  double queue = 0;
  std::size_t count = 0;
  for (auto const& worker : plan.queue) {
    double work = 0;
    int prior=-1;
    for (auto const& item : worker) {
      if (item.stage + 1 >= input.stage_offsets.size() || item.logical < 0)
        return fail("bounds invalid stage/task");
      int const node = input.stage_offsets[item.stage] + item.logical;
      if (node >= input.stage_offsets[item.stage + 1] || seen[node]++)
        return fail("bounds task outside stage or duplicated");
      if (prior>=0) {queue_next[prior]=node;++degree[node];}
      prior=node;
      work += input.task_ns[node];
      ++count;
    }
    queue = std::max(queue, work);
  }
  if (count != input.task_ns.size()) return fail("bounds plan omits tasks");
  auto const& graph=input.graph;
  if (graph.group_of_node.size()!=input.task_ns.size()) return fail("unprepared binding bound");
  auto remaining=graph.producer_count;
  std::vector<double> group_end(graph.successors.size(),0),end(input.task_ns.size(),0);
  for (auto const& row:graph.successors) for (int succ:row) ++degree[succ];
  std::vector<int> ready;
  for (int n=0;n<int(degree.size());++n) if (!degree[n]) ready.push_back(n);
  double binding=0;
  std::size_t visited=0;
  while (visited<ready.size()) {
    int n=ready[visited++];end[n]+=input.task_ns[n];binding=std::max(binding,end[n]);
    auto arrive=[&](int succ,double time) {
      end[succ]=std::max(end[succ],time);
      if (--degree[succ]==0) ready.push_back(succ);
    };
    if (queue_next[n]>=0) arrive(queue_next[n],end[n]);
    int group=graph.group_of_node[n];
    group_end[group]=std::max(group_end[group],end[n]);
    if (--remaining[group]==0)
      for (int succ:graph.successors[group]) arrive(succ,group_end[group]);
  }
  if (visited!=input.task_ns.size()) return fail("binding-bound queue order violates L-a");
  out->binding_path_ns=binding;
  out->work_lb_ns = input.work_ns / plan.queue.size();
  out->queue_lb_ns = queue;
  out->critical_path_ns = input.critical_path_ns;
  out->lower_bound_ns = std::max({out->work_lb_ns, queue, binding});
  return true;
}

bool RankPlans(SimulatorInput const& input, PreparedPlanBounds const& prepared,
               std::vector<MaterializedPlan const*> const& plans,
               SimulatorOptions const& options, HopCurve const& hop,
               std::size_t top_k, std::vector<RankedPlan>* out,
               std::string* error) {
  if (!top_k || plans.empty() || prepared.task_ns != input.task_ns ||
      !input.graph || prepared.stage_offsets != input.graph->stage_offsets) {
    if (error) *error = "invalid or stale hierarchical ranking input";
    return false;
  }
  auto shared_input=input;
  shared_input.prepared_graph=&prepared.graph;
  out->clear();
  for (std::size_t i = 0; i < plans.size(); ++i) {
    RankedPlan item;
    item.index = i;
    if (!plans[i] || !EvaluatePlanBounds(prepared, *plans[i], &item.bounds, error))
      return false;
    item.makespan_ns = item.bounds.lower_bound_ns;
    out->push_back(item);
  }
  std::stable_sort(out->begin(), out->end(), [](auto const& a, auto const& b) {
    return a.bounds.lower_bound_ns < b.bounds.lower_bound_ns;
  });
  double const cutoff = (*out)[std::min(top_k, out->size()) - 1].bounds.lower_bound_ns;
  for (auto& item : *out) {
    if (item.bounds.lower_bound_ns > cutoff) break;
    SimulatorResult result;
    if (!SimulateExecution(shared_input, *plans[item.index], options, hop, &result, error))
      return false;
    item.simulated = true;
    item.makespan_ns = result.makespan_ns;
  }
  std::stable_sort(out->begin(), out->end(), [](auto const& a, auto const& b) {
    if (a.simulated != b.simulated) return a.simulated > b.simulated;
    return a.makespan_ns < b.makespan_ns;
  });
  return true;
}

}  // namespace tilemega::solver
