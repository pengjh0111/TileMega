// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativeFlow.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tilemega::solver {

char const* Sm90SyncKindName(Sm90SyncKind kind) {
  switch (kind) {
    case Sm90SyncKind::kAggregate: return "aggregate";
    case Sm90SyncKind::kGroup: return "group";
    case Sm90SyncKind::kHandoff: return "handoff";
    case Sm90SyncKind::kCounted: return "counted";
    case Sm90SyncKind::kData: return "data";
  }
  return "aggregate";
}

namespace {
// The release table of one boundary. `all_producer` is the V3 barrier: every
// producer task must finish. Otherwise each consumer is released by its own
// predecessors, coarsened to the event grouping, which is the construction
// FlowPreparation.cpp:635-686 performs from the coupling oracle.
std::shared_ptr<std::vector<std::pair<int, int>> const> ReleaseTable(
    int producer_tasks, int consumer_tasks, int kappa) {
  std::vector<std::pair<int, int>> sorted;
  sorted.reserve(consumer_tasks);
  double ratio = consumer_tasks > 0 ? double(producer_tasks) / consumer_tasks : 0.0;
  for (int consumer = 0; consumer < consumer_tasks; ++consumer) {
    // A linear coupling: consumer j reads the producer prefix that its own
    // index spans. CoarsenRelease then rounds that endpoint up to the end of
    // its event group, which is what kappa > 1 means.
    int last = ratio > 0 ? int(std::ceil((consumer + 1) * ratio)) - 1 : producer_tasks - 1;
    last = std::min(std::max(last, 0), std::max(producer_tasks - 1, 0));
    sorted.emplace_back(CoarsenRelease(last, producer_tasks, kappa), consumer);
  }
  std::stable_sort(sorted.begin(), sorted.end());
  return std::make_shared<std::vector<std::pair<int, int>> const>(std::move(sorted));
}
}  // namespace

std::vector<Sm90NativeStageNode> Sm90NativeStages(Sm90DenseWorkload const& work,
                                                  Sm90DenseCoordinates const& c,
                                                  Sm90DenseCostModel const& cost,bool isolated) {
  // The stage sequence and the `pre` rule are the cost model's own
  // (Sm90DenseCostModel.cpp:99-118); only the pricing and the boundary
  // semantics differ here.
  std::vector<Sm90NativeStageNode> nodes;
  auto resident = [&](unsigned f) {return c.gemms[f].FullyResident(work.shapes[f][1]);};
  unsigned last = c.moe ? 7u : 3u;
  std::array<bool, 8> pre{};
  pre[0] = c.weight_stream && resident(last);
  pre[4] = c.weight_stream && resident(last);
  if (c.moe) {
    pre[5] = c.weight_stream && resident(1);
    pre[7] = c.weight_stream && resident(6) && !c.expert_fuse;
  } else {
    pre[2] = c.weight_stream && resident(1);
    pre[3] = c.weight_stream && resident(2);
  }

  for (unsigned f : c.Active()) {
    Sm90NativeStageNode node;
    node.price = Sm90NativeGemmPrice(work, f, c.gemms[f], c.residency, cost,
                                     !isolated && c.weight_stream, !isolated && pre[f]);
    node.instances = f == 4 ? 1 : work.layers;
    // Appendix A. qkv releases attention by KV group when the coordinate asks
    // for the group wait (Coarsen); o and down reduce across ranks inside the
    // epilogue at tp > 1 (Realize, data); the fused expert stage waits on a
    // per-expert count; everything else is the V3 barrier.
    if (c.moe && c.expert_fuse && f == 7) {
      node.sync = Sm90SyncKind::kCounted;
      node.kappa = std::max(1, int(work.shapes[6][0] / c.gemms[6].tile_n));
      node.price.hop_ns=0;
    }
    nodes.push_back(std::move(node));
  }
  // Attention sits between qkv and o, where the cost model places it.
  Sm90NativeStageNode attention;
  attention.price = Sm90NativeAttentionPrice(work, c.attention, c.residency, cost);
  attention.instances = work.layers;
  attention.sync = c.group_wait ? Sm90SyncKind::kGroup : Sm90SyncKind::kAggregate;
  attention.kappa = c.group_wait ?
      std::max(1,work.shapes[0][0]/c.gemms[0].tile_n/work.hkv) : 1;
  if(c.group_wait)attention.price.hop_ns=std::max(0.,
      cost.Constants().Barrier(cost.Grid(c.residency))-
      cost.Constants().GroupWaitSaving(cost.Grid(c.residency)));
  if(isolated)for(auto* part:{&attention.price.middle,&attention.price.head,
      &attention.price.tail,&attention.price.head_and_tail})part->no_producer_dram_bytes=0;
  nodes.insert(nodes.begin() + 1, std::move(attention));
  return nodes;
}

double Sm90NativeOperatorSumScore(Sm90DenseWorkload const& work,
    Sm90DenseCoordinates const& c,Sm90DenseCostModel const& cost) {
  double sum=0;
  for(auto const& node:Sm90NativeStages(work,c,cost,true)) {
    auto isolated=Sm90NativeSingleStageProblem(node.price,cost,c.residency);
    sum+=node.instances*EvaluateFlow(isolated).makespan_ns;
  }
  return sum;
}

Sm90NativeFlowResult Sm90NativePoolScore(Sm90DenseWorkload const& work,
                                         Sm90DenseCoordinates const& c,
                                         Sm90DenseCostModel const& cost,
                                         analysis::CouplingCache* cache) {
  auto const& constants = cost.Constants();
  Sm90NativeFlowResult result;
  result.nodes = Sm90NativeStages(work, c, cost);

  auto& problem=result.problem;
  problem.workers = cost.Grid(c.residency);
  problem.dram_gbps = constants.dram_gbps;
  problem.hop_ns = constants.Barrier(problem.workers);

  // Eq. (8) is over T_theta, the complete tile-task set, so the program is
  // expanded: every layer contributes its stage sequence and the head runs
  // once. Multiplying one layer's makespan by the layer count would charge
  // the head 28 times and would also hide the boundary between layers.
  std::vector<std::size_t> node_of_space;
  int layers = 0;
  for (auto const& node : result.nodes) layers = std::max(layers, node.instances);
  for (int layer = 0; layer < layers; ++layer)
    for (std::size_t i = 0; i < result.nodes.size(); ++i)
      if (result.nodes[i].price.name!="gemm4" && layer < result.nodes[i].instances)
        node_of_space.push_back(i);
  for(std::size_t i=0;i<result.nodes.size();++i)
    if(result.nodes[i].price.name=="gemm4")node_of_space.push_back(i);

  for (std::size_t space_index = 0; space_index < node_of_space.size(); ++space_index) {
    std::size_t i = node_of_space[space_index];
    auto const& node = result.nodes[i];
    FlowSpace space;
    space.name = node.price.name;
    space.category = node.price.category;
    space.count = node.price.tasks;
    space.order = int(space_index);
    // A handoff's consumer is a reducer space: its tasks are the LA groups and
    // its release carries no hop (appendix A).
    space.handoff_reducer = node.sync == Sm90SyncKind::kHandoff;
    space.fused_reducer = node.sync == Sm90SyncKind::kCounted;
    enum {kBoth = 0, kHead = 1, kTail = 2, kMiddle = 3};
    space.pieces = {FlowPiece{0, node.price.head_and_tail}, FlowPiece{0, node.price.head},
                    FlowPiece{0, node.price.tail}, FlowPiece{0, node.price.middle}};
    space.piece_of_task.assign(node.price.tasks, kMiddle);
    for (int task = 0; task < node.price.tasks; ++task) {
      bool head = task < node.price.ends;
      bool tail = task >= node.price.tasks - node.price.ends;
      int piece = head && tail ? kBoth : head ? kHead : tail ? kTail : kMiddle;
      space.piece_of_task[task] = piece;
      ++space.pieces[piece].count;
    }
    problem.spaces.push_back(std::move(space));
  }

  // One boundary per consecutive pair of spaces, each with its own delay.
  for (std::size_t space_index = 1; space_index < node_of_space.size(); ++space_index) {
    auto const& node = result.nodes[node_of_space[space_index]];
    FlowRelease edge;
    edge.producer = int(space_index - 1);
    edge.consumer = int(space_index);
    edge.kappa = node.kappa;
    edge.all_producer = node.sync == Sm90SyncKind::kAggregate;
    edge.sorted = ReleaseTable(problem.spaces[space_index - 1].count,
                               problem.spaces[space_index].count, node.kappa);
    switch (node.sync) {
      case Sm90SyncKind::kAggregate:
        edge.hop_ns = constants.Barrier(problem.workers);
        break;
      case Sm90SyncKind::kGroup:
        // A group count is cheaper than the whole-grid barrier: the saving is
        // the measured constant the coordinate's group wait buys.
        edge.hop_ns = std::max(0.0, constants.Barrier(problem.workers) -
                                    constants.GroupWaitSaving(problem.workers));
        break;
      case Sm90SyncKind::kHandoff:
      case Sm90SyncKind::kCounted:
      case Sm90SyncKind::kData:
        // The last arriver, the per-expert count and the in-epilogue reduction
        // are paid inside the task's own price, not as a boundary delay.
        edge.hop_ns = 0;
        break;
    }
    problem.edges.push_back(std::move(edge));
  }

  result.evaluated=EvaluateFlow(problem);
  auto const& evaluated=result.evaluated;
  // The makespan of the whole step is the score; nothing is multiplied.
  result.pool_score_ns = evaluated.makespan_ns;
  result.step_ns = evaluated.makespan_ns;
  // Report per stage kind, summed over that kind's instances.
  result.stage_ns.assign(result.nodes.size(), 0.0);
  for (std::size_t space_index = 0; space_index < evaluated.spaces.size(); ++space_index) {
    auto const& space = evaluated.spaces[space_index];
    result.stage_ns[node_of_space[space_index]] +=
        space.last_end - std::max(space.first_start, 0.0);
  }
  if (cache) {
    result.cache_hits = cache->hits;
    result.cache_misses = cache->misses;
  }
  return result;
}
}  // namespace tilemega::solver
