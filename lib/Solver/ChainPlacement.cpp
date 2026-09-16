// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ChainPlacement.h>

#include <algorithm>
#include <numeric>
#include <utility>

namespace tilemega::solver {

namespace {

// One plain pass.  `rank_extra_ns` is added to a node's score in the extraction
// DP only -- never to the work the stop test, the caps or the simulation go on
// -- and is empty for the first pass.  `delay_out` reports, per node, the time
// this schedule has it waiting for its worker rather than for its data.
bool SchedulePass(ChainRequest const& request,
                  std::vector<double> const& rank_extra_ns, ChainSchedule* out,
                  std::vector<double>* delay_out, std::string* error) {
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
  int const grid = request.grid;
  if (grid <= 0) return fail("critical-chain scheduling needs a positive grid");
  if (!(request.chain_stop_ratio > 0.0))
    return fail("chain_stop_ratio must be positive");

  // Charged exactly as the earliest-finish path charges it, so the two
  // schedulers rank the same plan by the same hop. The cost-aware fill also
  // approximates sibling-CTA sharing; SimulateExecution remains the arbiter
  // of the makespan, including the existing feedback evaluations.
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

  out->worker.assign(nodes, -1);
  out->slot.assign(nodes, -1);
  out->chain_of.assign(nodes, -1);
  out->start_ns.assign(nodes, 0.0);
  out->end_ns.assign(nodes, 0.0);
  out->makespan_ns = 0.0;
  out->spine_length_ns = 0.0;
  out->max_queue_ns = 0.0;
  out->chain_count = 0;
  out->chain_interleaves = 0;
  out->split_count = 0;
  out->fill_overflows = 0;
  out->rejected_extensions.clear();
  if (delay_out) delay_out->assign(nodes, 0.0);
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
        if (--degree[succ] == 0) {
          stack.push_back(succ);
        }
    }
    if (static_cast<int>(topo.size()) != nodes)
      return fail("the task DAG has a cycle over " +
                  std::to_string(nodes - topo.size()) + " tasks");
  }
  std::vector<int> topo_index(nodes, 0);
  for (int i = 0; i < nodes; ++i) topo_index[topo[i]] = i;

  std::vector<double> hop_cost(nodes, 0.0);
  for (int node = 0; node < nodes; ++node)
    if (!graph.successors[node].empty())
      hop_cost[node] =
          request.hop.Ns(static_cast<int>(graph.successors[node].size()), 1);

  if (request.cost_aware_extend) {
    // Rank the fill's ready tasks by their remaining dependency path, so a
    // short off-path branch cannot occupy a queue ahead of a critical one.
    std::vector<double> rank(nodes, 0.0);
    for (int i = nodes - 1; i >= 0; --i) {
      int const node = topo[i];
      double tail = 0.0;
      for (int succ : graph.successors[node])
        tail = std::max(tail, hop_cost[node] + rank[succ]);
      rank[node] = request.task_ns[node] + tail;
    }
    auto lower_rank = [&](int a, int b) {
      return rank[a] != rank[b] ? rank[a] < rank[b] : a > b;
    };
    std::vector<int> ready, degree = indegree;
    for (int node = 0; node < nodes; ++node)
      if (degree[node] == 0) ready.push_back(node);
    std::make_heap(ready.begin(), ready.end(), lower_rank);
    topo.clear();
    while (!ready.empty()) {
      std::pop_heap(ready.begin(), ready.end(), lower_rank);
      int const node = ready.back();
      ready.pop_back();
      topo.push_back(node);
      for (int succ : graph.successors[node])
        if (--degree[succ] == 0) {
          ready.push_back(succ);
          std::push_heap(ready.begin(), ready.end(), lower_rank);
        }
    }
    for (int i = 0; i < nodes; ++i) topo_index[topo[i]] = i;
  }

  if (request.cost_aware_extend)
    for (int node = 0; node < nodes; ++node)
      for (int succ : graph.successors[node])
        if (hop_cost[node] <= request.task_ns[succ])
          out->rejected_extensions.push_back(
              {node, succ, hop_cost[node], request.task_ns[succ]});

  // Phase 1 and 2 (§6.1, §6.2).  A chain is the longest path through the nodes
  // no chain has claimed yet.  The ranking DP charges the hop on every edge for
  // the general case, where at 1235 ns a hop a path of many small tasks can
  // outrun a path of few large ones -- but in these graphs it is measured to
  // change nothing, and this comment is not allowed to imply otherwise: every
  // maximal path here spans the same stages, so a constant per-edge charge
  // shifts every candidate equally and cannot re-rank them.  Verified by
  // running both forms: `predicted.tsv` is byte-identical in all six cells and
  // `chain_weights.tsv` differs only in `solve_us`.  The capacity cap and the
  // stop test stay stated in work, which is what a queue costs once the hops
  // inside it are gone, so the two are different numbers.
  int spine_nodes = 0;
  std::vector<int> cluster(nodes, -1);
  std::vector<std::vector<int>> cluster_nodes;
  std::vector<double> cluster_ns;
  double const total_work_ns =
      std::accumulate(request.task_ns.begin(), request.task_ns.end(), 0.0);
  double remaining_ns = total_work_ns;
  {
    std::vector<double> down(nodes, 0.0);
    std::vector<int> next(nodes, -1);
    while (static_cast<int>(cluster_nodes.size()) < grid) {
      double best_ns = -1.0;
      int best_head = -1;
      for (int i = nodes - 1; i >= 0; --i) {
        int const node = topo[i];
        if (cluster[node] >= 0) continue;
        double tail = 0.0;
        int follow = -1;
        for (int succ : graph.successors[node]) {
          if (cluster[succ] >= 0) continue;
          // A zero-hop queue edge still serializes the successor's work.
          if (request.cost_aware_extend &&
              hop_cost[node] <= request.task_ns[succ])
            continue;
          double const reach = down[succ] + hop_cost[node];
          if (reach > tail || follow < 0) {
            tail = reach;
            follow = succ;
          }
        }
        down[node] = request.task_ns[node] +
                     (rank_extra_ns.empty() ? 0.0 : rank_extra_ns[node]) + tail;
        next[node] = follow;
        if (down[node] > best_ns) {
          best_ns = down[node];
          best_head = node;
        }
      }
      if (best_head < 0) break;

      std::vector<int> chain;
      double chain_ns = 0.0;
      for (int node = best_head; node >= 0; node = next[node]) {
        chain.push_back(node);
        chain_ns += request.task_ns[node];
      }

      // The average load a still-unplaced worker would otherwise carry.  Below
      // it a chain has stopped being a critical path and has become an ordinary
      // queue, and chaining it only removes scheduling freedom.  The spine
      // itself is always taken, whatever the ratio says.
      double const average = remaining_ns / grid;
      if (!cluster_nodes.empty() &&
          chain_ns <= request.chain_stop_ratio * average)
        break;

      for (int node : chain) cluster[node] = static_cast<int>(cluster_nodes.size());
      remaining_ns -= chain_ns;
      cluster_ns.push_back(chain_ns);
      cluster_nodes.push_back(std::move(chain));
      if (cluster_nodes.size() == 1) {
        out->spine_length_ns = chain_ns;
        spine_nodes = static_cast<int>(cluster_nodes.back().size());
      }
    }
  }
  out->chain_count = static_cast<int>(cluster_nodes.size());
  out->chain_of = cluster;

  // §6.3, and the one place this round's first attempt was wrong.  Every queue
  // below is ordered by topological index, so the union of task and queue edges
  // is a subset of a single topological order and L-a holds by construction.  A
  // chain needs no legality test of its own: a chain is a *path* in the task
  // DAG, so the queue edges between its consecutive members are task edges
  // already.  §6.3's hazard -- two chains whose cross edges run in both
  // directions -- is a cycle in the *contracted* graph, which is strictly
  // stronger than L-a and is not the condition that gates a Plan, because
  // a1->a2->b1->b2->a1 would already be a cycle in the task DAG itself.
  // Enforcing the contracted form instead cost the mechanism the whole of what
  // it was for: measured at gqa2 s4, 770 extensions refused over 200 nodes and
  // a 19.9 us spine against a 179.8 us longest path, because a bypass around a
  // path edge breaks convexity and these graphs are full of them.  What is left
  // to protect is contiguity, which is a performance property and is counted.
  std::vector<double> load(grid, 0.0);
  std::vector<std::vector<int>> queue(grid);
  std::vector<int> chain_lo(grid, -1), chain_hi(grid, -1);
  std::vector<int> order(cluster_nodes.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return cluster_ns[a] > cluster_ns[b];
  });
  for (int part : order) {
    int chosen = 0;
    for (int w = 1; w < grid; ++w)
      if (load[w] < load[chosen]) chosen = w;
    // A path rises in topological index, so a chain's span is its ends.
    int const lo = topo_index[cluster_nodes[part].front()];
    int const hi = topo_index[cluster_nodes[part].back()];
    if (chain_hi[chosen] >= 0 && lo <= chain_hi[chosen] &&
        chain_lo[chosen] <= hi)
      ++out->chain_interleaves;
    for (int node : cluster_nodes[part]) {
      out->worker[node] = chosen;
      queue[chosen].push_back(node);
    }
    chain_lo[chosen] =
        chain_lo[chosen] < 0 ? lo : std::min(chain_lo[chosen], lo);
    chain_hi[chosen] = std::max(chain_hi[chosen], hi);
    load[chosen] += cluster_ns[part];
  }

  // Phase 3 (§6.2): the rest goes where its producer already is, so long as
  // that worker stays under the spine -- the spine is the cap because no queue
  // shorter than the critical chain can be what decides the makespan.  Round
  // two capped on `baseline_max_queue` instead, which is the F-129 pathology: a
  // cap derived from the balanced plan admits queues the critical path never
  // justified.  A worker holding a chain is only offered a task that sorts
  // clear of the chain's span, so filling cannot drop a foreign task between
  // two chain members and block the chain behind it at a window of one.
  auto clear_of_chain = [&](int w, int node) {
    return chain_hi[w] < 0 || topo_index[node] < chain_lo[w] ||
           topo_index[node] > chain_hi[w];
  };
  // A queue is capped in count and in work, and the two caps answer two
  // different failures measured at mha4 s128.  In work alone the cap admitted
  // 864 of that cell's 419.84 ns filled tasks under 362813 ns of spine, and at
  // a window of one each is a FIFO step the tasks behind it wait through: 565 of
  // 614 critical-path edges became queue steps, makespan 627116 ns against
  // rotate's 437124.  Adding the count cap took that to 158 of 481 and 602997
  // ns -- better, still losing, because the spine is the wrong *work* bound for
  // a filled worker.  A makespan is no smaller than the total work over the grid
  // either, which is 73 us here against the spine's 363 us, so following a
  // producer up to the spine builds a queue five times heavier than balance
  // allows.  Fill follows its producer only while the worker stays inside the
  // work bound; the spine caps only the chains that earned it.  A task heavier
  // than the bound still has to land somewhere, so the bound is never below it.
  int const fill_cap = std::max(spine_nodes, (nodes + grid - 1) / grid);
  double const work_bound = total_work_ns / grid;
  // Fill is priced in time, not in queue position, and this is the third bound
  // the step has worn.  "Follow the producer, else the shortest queue" is what
  // §6.2 asks for and it is what loses: at mha4 s128 the extraction is clean --
  // spine contiguous at 362813 ns, 0 interleaves, 0 splits, every filled worker
  // inside the 37731 ns work bound -- and the schedule still simulates at
  // 618861 ns against rotate's 437124, with 49 critical-path hops against 35
  // and 150 critical-path queue edges against 32.  The load is balanced and the
  // makespan is not, because a queue is a FIFO at a window of one: following a
  // producer puts a task behind whatever that worker already holds, and the
  // critical path leaves the spine to thread 150 such steps.  Counting queue
  // elements cannot see that, since what a step costs is when the task ahead of
  // it finishes.  So the choice is the earliest finish this node can reach over
  // the workers the chains leave free and the caps still admit.  The hop is
  // inside that estimate, so a producer's worker is still preferred -- now when
  // it is actually sooner, which is the part the old rule assumed.
  std::vector<double> est_end(nodes, 0.0);
  std::vector<double> free_ns(grid, 0.0);
  std::vector<int> est_hops(nodes, 0), free_hops(grid, 0);
  std::vector<std::vector<int>> sm_workers(sm_count);
  if (request.cost_aware_extend)
    for (int w = 0; w < grid; ++w) sm_workers[worker_sm[w]].push_back(w);
  auto finish_on = [&](int node, int w, int* path_hops = nullptr) {
    double start = free_ns[w];
    int hops = free_hops[w];
    for (int pred : predecessors[node]) {
      double arrival = est_end[pred];
      if (out->worker[pred] != w) arrival += hop_cost[pred];
      if (arrival >= start) {
        int const incoming = est_hops[pred] + (out->worker[pred] != w);
        hops = arrival > start ? incoming : std::max(hops, incoming);
      }
      start = std::max(start, arrival);
    }
    double stretch = 1.0;
    if (request.cost_aware_extend)
      for (int sibling : sm_workers[worker_sm[w]])
        if (sibling != w && free_ns[sibling] > start) stretch += 1.0;
    if (path_hops) *path_hops = hops;
    return start + request.task_ns[node] * stretch;
  };
  for (int i = 0; i < nodes; ++i) {
    int const node = topo[i];
    if (out->worker[node] < 0) {
      double const work_limit = std::max(work_bound, request.task_ns[node]);
      (void)work_limit;
      int chosen = -1;
      double best = 0.0;
      int best_hops = 0;
      for (int w = 0; w < grid; ++w) {
        if (!clear_of_chain(w, node)) continue;
        if (request.cap_fill &&
            (static_cast<int>(queue[w].size()) >= fill_cap ||
             load[w] + request.task_ns[node] > work_limit))
          continue;
        int path_hops = 0;
        double const finish = finish_on(node, w, &path_hops);
        if (chosen < 0 || finish < best ||
            (request.cost_aware_extend && finish == best && path_hops < best_hops)) {
          best = finish;
          best_hops = path_hops;
          chosen = w;
        }
      }
      if (chosen < 0) {
        // Every worker clear of a chain is at a cap.  The task goes to the
        // earliest-finishing of them anyway and the overflow is counted, not
        // hidden.
        for (int w = 0; w < grid; ++w) {
          if (!clear_of_chain(w, node)) continue;
          int path_hops = 0;
          double const finish = finish_on(node, w, &path_hops);
          if (chosen < 0 || finish < best ||
              (request.cost_aware_extend && finish == best && path_hops < best_hops)) {
            best = finish;
            best_hops = path_hops;
            chosen = w;
          }
        }
        if (chosen >= 0) ++out->fill_overflows;
      }
      if (chosen < 0) {
        // Every worker holds a chain this task sorts inside.  Legality is not
        // at stake -- the queue stays topological either way -- so the task is
        // placed and the lost contiguity is recorded rather than hidden.
        for (int w = 0; w < grid; ++w) {
          int path_hops = 0;
          double const finish = finish_on(node, w, &path_hops);
          if (chosen < 0 || finish < best ||
              (request.cost_aware_extend && finish == best && path_hops < best_hops)) {
            best = finish;
            best_hops = path_hops;
            chosen = w;
          }
        }
        ++out->chain_interleaves;
      }
      out->worker[node] = chosen;
      queue[chosen].push_back(node);
      load[chosen] += request.task_ns[node];
    }
    // Priced in topological order, which is also the order phase 4 gives every
    // queue, so this estimate is the schedule phase 4 goes on to confirm.
    int const w = out->worker[node];
    est_end[node] = finish_on(node, w, &est_hops[node]);
    free_ns[w] = est_end[node];
    free_hops[w] = est_hops[node];
  }
  for (int w = 0; w < grid; ++w)
    out->max_queue_ns = std::max(out->max_queue_ns, load[w]);

  // Phase 4 (§6.4): sigma is topological order within each queue, which keeps
  // every chain contiguous wherever phase 3 could sort its filling clear of the
  // span, and leaves stages free to interleave otherwise.  Ordering tasks by an
  // estimated earliest start instead -- this round's first attempt -- dropped
  // filled tasks between chain members, and at a window of one the queue is
  // strict FIFO, so 732 to 2392 queue steps landed on the critical path.
  // `CheckPlanLegality` is still the arbiter; this is why it is expected to
  // pass, not a reason to skip it.
  for (int w = 0; w < grid; ++w) {
    std::sort(queue[w].begin(), queue[w].end(), [&](int a, int b) {
      return topo_index[a] < topo_index[b];
    });
    for (std::size_t s = 0; s < queue[w].size(); ++s)
      out->slot[queue[w][s]] = static_cast<int>(s);
  }
  // Queue order is topological order, so one pass over `topo` finds every
  // predecessor and every earlier slot on the same worker already priced.
  std::vector<double> worker_free(grid, 0.0);
  for (int i = 0; i < nodes; ++i) {
    int const node = topo[i];
    int const w = out->worker[node];
    double start = worker_free[w];
    for (int pred : predecessors[node]) {
      double arrival = out->end_ns[pred];
      if (out->worker[pred] != w) arrival += hop_cost[pred];
      start = std::max(start, arrival);
    }
    out->start_ns[node] = start;
    out->end_ns[node] = start + request.task_ns[node];
    worker_free[w] = out->end_ns[node];
    out->makespan_ns = std::max(out->makespan_ns, out->end_ns[node]);
  }
  if (delay_out) {
    for (int node = 0; node < nodes; ++node) {
      double ready = 0.0;
      for (int pred : predecessors[node]) {
        double arrival = out->end_ns[pred];
        if (out->worker[pred] != out->worker[node]) arrival += hop_cost[pred];
        ready = std::max(ready, arrival);
      }
      (*delay_out)[node] = std::max(0.0, out->start_ns[node] - ready);
    }
  }
  return true;
}

}  // namespace

// The static extraction of §6.1 is measured not to shorten the critical path at
// seq 128: at mha4 s128 the spine is 40 nodes and 362813 ns while the scheduled
// critical path is 87 steps, 28 of them queue steps, and admitting 21x more
// chains moves the hop count 39 -> 38 (`raw/stop_ratio_sweep.tsv`), while
// pricing the fill differently moves it not at all (`raw/fill_cap_sweep.tsv`).
// The cause is the DP above: it scores a path in work and hops and carries no
// queue term, so it optimises a path the fill then abandons.  Each feedback
// round hands that term back and re-extracts.
//
// Which model supplies that term decides whether the loop can help at all.
// Handing back the estimate the pass above ends with made the answer worse:
// that estimate and the simulator that scores the emitted plan are two models,
// and a loop tuned on the first while judged by the second walks away from the
// second (mha4 s128, 39 -> 41 critical-path hops).  So when the caller supplies
// an arbiter, it both ranks and scores, and the internal estimate is used for
// neither.  A round can then never lose by the measure that reports the result.
bool ScheduleByCriticalChain(ChainRequest const& request, ChainSchedule* out,
                             std::string* error) {
  int const rounds = request.feedback_rounds > 0 ? request.feedback_rounds : 0;
  std::vector<double> extra;
  ChainSchedule best;
  double best_score = 0.0;
  bool have_best = false;
  for (int round = 0; round <= rounds; ++round) {
    ChainSchedule pass;
    std::vector<double> delay;
    if (!SchedulePass(request, extra, &pass, &delay, error)) return false;
    double score = pass.makespan_ns;
    std::vector<double> blocked;
    if (request.evaluate && !request.evaluate(pass, &score, &blocked, error))
      return false;
    if (!have_best || score < best_score) {
      best = std::move(pass);
      best_score = score;
      have_best = true;
    }
    extra = request.evaluate ? std::move(blocked) : std::move(delay);
  }
  *out = std::move(best);
  return true;
}

}  // namespace tilemega::solver
